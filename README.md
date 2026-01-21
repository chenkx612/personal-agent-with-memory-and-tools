# ersonal-agent-with-memory-and-tools

个人ai助理，具有以下几个核心功能
- 长期记忆管理能力：可以通过对话动态管理长期记忆，了解用户的个人信息和偏好；
- 工具使用能力：可以调用工具，如获取时间，地区，天气等；

## 项目结构

- `src/`: 源代码目录
  - `agent.py`: Agent 逻辑
  - `main.py`: 主程序逻辑
  - `tools.py`: 工具函数
- `tests/`: 测试目录
- `run.py`: 启动脚本

## 如何运行

1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

2. 运行代理：
   ```bash
   python run.py
   ```

3. 运行测试：
   ```bash
   python tests/test_setup.py
   ```
