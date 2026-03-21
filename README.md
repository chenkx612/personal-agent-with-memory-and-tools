# Personal Agent with Memory and Tools

个人 AI 助理，具备长期记忆管理、笔记系统和工具调用能力。本项目深度整合了 LLM 与 RAG 技术，实现智能对话与个性化服务。

## 核心算法与模型

本项目采用以下核心技术栈构建，侧重于轻量级、高效的本地化与云端混合架构：

### 1. 大语言模型 (LLM)
- **核心模型**: `deepseek-chat` (DeepSeek V3.2)
  - 通过 OpenAI 兼容接口调用 DeepSeek API。
  - **参数设置**: Temperature=0.7，保证回答的创造性与准确性的平衡。
- **Agent 框架**: `LangGraph` + `LangChain`
  - 实现了 **ReAct (Reasoning + Acting)** 范式。Agent 能够通过 "思考-行动-观察" 的循环自主决策，动态选择工具解决问题。
  - 支持 **Human-in-the-Loop**：特定工具（如 `add_note`）需要用户确认后执行。

### 2. 检索增强生成 (RAG)与记忆系统
项目实现了两套独立的 RAG 系统，分别用于用户画像和笔记管理：

#### 用户画像记忆
- **Embedding 算法**: `sentence-transformers/all-MiniLM-L6-v2`
  - **类型**: Dense Embedding (稠密向量嵌入)。
  - **特点**: 轻量级（约 80MB），速度快，运行于本地 CPU。将用户的文本记忆转化为 384 维向量。
- **向量检索 (Vector Search)**:
  - **算法**: `FAISS` (Facebook AI Similarity Search)。
  - **索引**: 使用 L2 距离（欧氏距离）进行相似度计算，支持 Top-K 语义召回。
- **存储架构**:
  - **持久化**: `data/user_memory.json` (JSON 格式) 作为单一事实来源 (Source of Truth)。
  - **缓存**: 内存缓存 + 磁盘 FAISS 索引，带文件修改时间校验。

#### 笔记系统
- 独立的 FAISS 索引存储在 `data/notes_faiss_index/`
- 笔记数据存储在 `data/notes.json`
- 支持增量更新索引（添加笔记时无需重建整个索引）
- 按日期倒序浏览，支持标签管理

### 3. 会话管理算法
- **Short-term Memory**: 使用 `langgraph.checkpoint.memory.MemorySaver`。
  - 维护当前会话的上下文窗口，确保多轮对话的连贯性。
  - 自动管理消息历史 (Message History) 的状态转换。

## 功能特性

- **长期记忆管理**:
  - **语义搜索**: 通过 `search_memory` 工具，利用向量相似度找到相关的用户偏好（如 "我喜欢吃什么？"）。
  - **直接查找**: 通过 `get_memory` 获取特定键值的精确信息。
  - **动态更新**: 通过 `update_user_memory` 实时写入新知识，支持合并已有内容。
  - **智能整理**: `/tidy` 命令使用 LLM 自动整理合并记忆条目。

- **笔记系统**:
  - **记录笔记**: `add_note` 工具记录想法、会议要点等（需要用户确认）。
  - **语义搜索**: `search_notes` 搜索相关笔记。
  - **笔记浏览**: `/notes` 命令在 CLI 中按日期倒序列出笔记，支持查看详情。
  - **编辑支持**: 确认笔记时可使用系统编辑器（vim）修改内容。

- **环境感知**:
  - **天气查询**: 集成 `Open-Meteo` API + IP 地理位置。
    - 算法流程：先调用 ipinfo.io 获取位置，再调用 Open-Meteo Forecast API 获取实时天气。
  - **时间感知**: 获取系统当前日期、时间和星期。

- **网络搜索**:
  - 集成 DuckDuckGo 搜索，无需 API 密钥。

## 项目结构

```
.
├── main.py                 # CLI 启动入口
├── requirements.txt        # Python 依赖
├── config.yaml.template    # 配置文件模板
├── config.yaml            # 用户配置（git-ignored）
├── data/                  # 数据目录（git-ignored）
│   ├── user_memory.json   # 用户画像记忆
│   ├── notes.json         # 笔记数据
│   ├── memory_faiss_index/  # 记忆 FAISS 索引
│   └── notes_faiss_index/   # 笔记 FAISS 索引
└── src/
    ├── __init__.py
    ├── config.py          # 配置加载
    ├── core.py            # Agent 核心逻辑
    ├── llm.py             # 独立 LLM 实例
    ├── tools/             # 工具实现
    │   ├── __init__.py
    │   ├── base.py        # 共享工具函数
    │   ├── memory.py      # 用户记忆工具
    │   ├── notes.py       # 笔记工具
    │   ├── environment.py # 时间/天气工具
    │   └── web.py         # 网络搜索工具
    ├── interfaces/
    │   └── cli.py         # 命令行界面
    └── graph/             # LangGraph 实现
        ├── __init__.py
        ├── builder.py     # 图构建 + 审批配置
        ├── nodes.py       # Agent 和工具节点
        └── state.py       # 状态定义
```

## 如何运行

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 配置文件：
   ```bash
   cp config.yaml.template config.yaml
   # 编辑 config.yaml，设置 api_key 和其他配置
   ```

3. 运行命令行代理：
   ```bash
   python main.py
   ```

### CLI 命令

- `/notes` - 浏览笔记列表
- `/tidy` - 整理记忆（LLM 辅助）
- `/clear` - 清空会话上下文
- `/copy` - 复制上一轮回复
- `/exit` - 退出

## 配置说明

`config.yaml` 支持以下配置项：

```yaml
llm:
  api_key: "sk-..."          # API 密钥（必填）
  model: "deepseek-chat"      # 模型名称
  base_url: "https://api.deepseek.com"  # API 地址
  temperature: 0.7            # 温度参数

stream_output: true           # 是否流式输出

hf_endpoint: "https://hf-mirror.com"  # HuggingFace 镜像（可选，国内使用）

system_prompt: |              # 系统提示词
  你是我（用户）的专属个人秘书...
```

## 数据隐私

本项目包含敏感数据文件，均已被 git 忽略：
- `data/user_memory.json` - 用户画像记忆
- `data/notes.json` - 用户笔记
- `config.yaml` - 个人配置

如需备份到远程仓库，建议自行加密后提交。
