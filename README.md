# Personal Agent with Memory and Tools

个人 AI 助理，具备长期记忆管理和工具调用能力。本项目深度整合了 LLM 与 RAG 技术，实现智能对话与个性化服务。

## 核心算法与模型

本项目采用以下核心技术栈构建，侧重于轻量级、高效的本地化与云端混合架构：

### 1. 大语言模型 (LLM)
- **核心模型**: `deepseek-chat` (DeepSeek V3.2)
  - 通过 OpenAI 兼容接口调用 DeepSeek API。
  - **参数设置**: Temperature=0.7，保证回答的创造性与准确性的平衡。
- **Agent 框架**: `LangGraph` + `LangChain`
  - 实现了 **ReAct (Reasoning + Acting)** 范式。Agent 能够通过 "思考-行动-观察" 的循环自主决策，动态选择工具解决问题。

### 2. 检索增强生成 (RAG)与记忆系统
项目实现了一套混合记忆系统，结合了结构化存储与语义检索：

- **Embedding 算法**: `sentence-transformers/all-MiniLM-L6-v2`
  - **类型**: Dense Embedding (稠密向量嵌入)。
  - **特点**: 轻量级（约 80MB），速度快，运行于本地 CPU。将用户的文本记忆转化为 384 维向量。
- **向量检索 (Vector Search)**:
  - **算法**: `FAISS` (Facebook AI Similarity Search)。
  - **索引**: 使用 L2 距离（欧氏距离）进行相似度计算，支持 Top-K 语义召回。
- **存储架构**:
  - **持久化**: `user_memory.json` (JSON 格式) 作为单一事实来源 (Source of Truth)。
  - **运行时**: 每次启动或更新时，动态构建 FAISS 内存索引，保证检索的一致性。

### 3. 会话管理算法
- **Short-term Memory**: 使用 `langgraph.checkpoint.memory.MemorySaver`。
  - 维护当前会话的上下文窗口，确保多轮对话的连贯性。
  - 自动管理消息历史 (Message History) 的状态转换。

## 功能特性

- **长期记忆管理**:
  - **语义搜索**: 通过 `search_memory` 工具，利用向量相似度找到相关的用户偏好（如 "我喜欢吃什么？"）。
  - **直接查找**: 通过 `get_user_memory` 获取特定键值的精确信息。
  - **动态更新**: 通过 `update_user_memory` 实时写入新知识。
- **环境感知**:
  - **天气查询**: 集成 `Open-Meteo` API。
    - 算法流程：先调用 Geocoding API 将地名转换为经纬度，再调用 Forecast API 获取实时天气。
  - **时间感知**: 获取系统当前时间。

## 项目结构

- `src/`: 源代码目录
  - `agent.py`: Agent 构建逻辑，包含 LLM 初始化与 Graph 定义。
  - `main.py`: 主程序入口，包含 CLI 交互与会话管理。
  - `tools.py`: 工具函数实现（RAG 逻辑、天气 API、文件操作）。
- `tests/`: 测试目录
- `run.py`: 启动脚本

## 数据隐私与备份
本项目包含敏感数据文件 `user_memory.json`，该文件已被 git 忽略以保护隐私。

1. **初始化配置**:
   复制模板文件以创建初始配置：
   ```bash
   cp user_memory.json.template user_memory.json
   ```

2. **加密备份**:
   为了安全地备份你的数据到远程仓库，可以使用提供的脚本进行加密：
   ```bash
   # 加密 (生成 user_memory.enc)
   python secure_memory.py encrypt

   # 解密 (恢复 user_memory.json)
   python secure_memory.py decrypt
   ```
   加密后的 `user_memory.enc` 文件可以安全地提交到版本控制系统中。

## 如何运行

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 配置环境：
   确保 `.env` 文件中包含 `DEEPSEEK_API_KEY`。

3. 运行代理：
   ```bash
   python run.py
   ```

4. 运行测试：
   ```bash
   python tests/test_setup.py
   ```
