import sys
import os
import base64
import getpass
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MEMORY_FILE = os.path.join(BASE_DIR, "data", "user_memory.json")
ENC_FILE = os.path.join(BASE_DIR, "data", "user_memory.enc")

def derive_key(password: str, salt: bytes) -> bytes:
    """Derive a 32-byte URL-safe base64-encoded key from the password."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))

def encrypt(file_path, enc_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    print("Encrypting user_memory.json...")
    password = getpass.getpass("Enter encryption password: ")
    confirm = getpass.getpass("Confirm password: ")
    if password != confirm:
        print("Error: Passwords do not match.")
        return

    # Generate a random salt
    salt = os.urandom(16)
    key = derive_key(password, salt)
    f = Fernet(key)

    try:
        with open(file_path, "rb") as file:
            file_data = file.read()

        encrypted_data = f.encrypt(file_data)

        # Write salt + encrypted data
        with open(enc_path, "wb") as file:
            file.write(salt + encrypted_data)
        
        print(f"Success! Encrypted to {enc_path}")
        print(f"You can now safely commit {enc_path} to git.")
    except Exception as e:
        print(f"Encryption failed: {e}")

def decrypt(enc_path, file_path):
    if not os.path.exists(enc_path):
        print(f"Error: {enc_path} not found.")
        return

    print("Decrypting user_memory.enc...")
    password = getpass.getpass("Enter decryption password: ")

    try:
        with open(enc_path, "rb") as file:
            data = file.read()
        
        # Extract salt and data
        if len(data) < 16:
            print("Error: Invalid encrypted file format.")
            return
            
        salt = data[:16]
        encrypted_data = data[16:]

        key = derive_key(password, salt)
        f = Fernet(key)

        decrypted_data = f.decrypt(encrypted_data)
        
        # Write decrypted data (be careful not to overwrite if it exists? 
        # Actually user wants to restore, so overwrite is expected)
        if os.path.exists(file_path):
            overwrite = input(f"Warning: {file_path} already exists. Overwrite? (y/n): ")
            if overwrite.lower() != 'y':
                print("Aborted.")
                return

        with open(file_path, "wb") as file:
            file.write(decrypted_data)
        print(f"Success! Restored {file_path}")
        
    except Exception:
        print("Error: Decryption failed. Incorrect password or corrupted file.")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python run_security.py encrypt  # Encrypt user_memory.json -> user_memory.enc")
        print("  python run_security.py decrypt  # Decrypt user_memory.enc -> user_memory.json")
        sys.exit(1)
    
    action = sys.argv[1]
    
    if action == "encrypt":
        encrypt(MEMORY_FILE, ENC_FILE)
    elif action == "decrypt":
        decrypt(ENC_FILE, MEMORY_FILE)
    else:
        print(f"Unknown action: {action}")

if __name__ == "__main__":
    main()
