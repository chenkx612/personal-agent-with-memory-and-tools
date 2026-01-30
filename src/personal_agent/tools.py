import json
import os
import requests
from datetime import datetime
from typing import Optional
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Memory file path
# Use absolute path relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEMORY_FILE = os.path.join(BASE_DIR, "data", "user_memory.json")

# Global cache for vector store
_vectorstore_cache = None
_last_memory_mtime = 0

def _load_memory():
    if not os.path.exists(MEMORY_FILE):
        return {}
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_memory(memory):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)

def _get_vectorstore():
    global _vectorstore_cache, _last_memory_mtime
    
    if not os.path.exists(MEMORY_FILE):
        return None
        
    current_mtime = os.path.getmtime(MEMORY_FILE)
    
    # If cache is valid, return it
    if _vectorstore_cache is not None and current_mtime == _last_memory_mtime:
        return _vectorstore_cache
        
    # Rebuild index
    memory = _load_memory()
    if not memory:
        return None
        
    documents = []
    for key, value in memory.items():
        doc = Document(
            page_content=f"{key}: {value}",
            metadata={"key": key}
        )
        documents.append(doc)
    
    if not documents:
        return None

    # Use a small, fast local model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    _vectorstore_cache = FAISS.from_documents(documents, embeddings)
    _last_memory_mtime = current_mtime
    
    return _vectorstore_cache

@tool
def search_memory(query: str, k: int = 3):
    """Search the user's long-term memory for relevant information.
    
    Use this tool when you need to recall specific details about the user that might be stored in memory,
    rather than guessing or asking the user again.
    
    Args:
        query: The search query (e.g., "user's favorite food", "birthday").
        k: Number of results to return (default: 3).
    """
    vectorstore = _get_vectorstore()
    if not vectorstore:
        return "Memory is empty."
        
    docs = vectorstore.similarity_search(query, k=k)
    if not docs:
        return "No relevant information found in memory."
        
    results = []
    for doc in docs:
        results.append(doc.page_content)
        
    return "\n".join(results)

@tool
def get_current_time():
    """Get the current time and date."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def get_weather(location: str):
    """Get the current weather for a specific location.
    
    Args:
        location: The name of the city or region (e.g., "Beijing", "New York").
    """
    try:
        # 1. Geocoding to get latitude and longitude
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en&format=json"
        geo_response = requests.get(geo_url, timeout=10)
        
        if geo_response.status_code != 200:
             return f"Failed to fetch location data. Status code: {geo_response.status_code}"
             
        geo_data = geo_response.json()
        
        if not geo_data.get("results"):
            return f"Could not find location: {location}"
            
        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]
        name = geo_data["results"][0]["name"]
        country = geo_data["results"][0].get("country", "")
        
        # 2. Fetch Weather data
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather_response = requests.get(weather_url, timeout=10)
        
        if weather_response.status_code != 200:
            return f"Failed to fetch weather data. Status code: {weather_response.status_code}"
            
        weather_data = weather_response.json()
        
        current = weather_data.get("current_weather", {})
        temp = current.get("temperature")
        wind = current.get("windspeed")
        
        location_str = f"{name}, {country}" if country else name
        return f"Weather in {location_str}: {temp}°C, Wind Speed: {wind} km/h"
        
    except Exception as e:
        return f"Error fetching weather for {location}: {str(e)}"

@tool
def update_user_memory(key: str, value: str):
    """Update or add a piece of information about the user in long-term memory.
    
    Args:
        key: The category or key of the information (e.g., "name", "preference", "hobby").
        value: The detailed information to store.
    """
    memory = _load_memory()
    memory[key] = value
    _save_memory(memory)
    return f"Successfully updated memory: {key} = {value}"

@tool
def get_user_memory(key: Optional[str] = None):
    """Retrieve information from the user's long-term memory.
    
    Args:
        key: The specific key to retrieve. If None, returns all memory.
    """
    memory = _load_memory()
    if key:
        return memory.get(key, "Information not found.")
    return json.dumps(memory, ensure_ascii=False)
