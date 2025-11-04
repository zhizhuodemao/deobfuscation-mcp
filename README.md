# 反混淆 MCP 服务器

一个基于模型上下文协议（MCP）的服务器，通过 HTTP JSON-RPC 接口提供 IDA Pro 逆向工程能力。该服务器使基于 LLM 的工具能够与 IDA Pro 交互，进行二进制分析、反编译和反混淆任务。

## 概述

此 MCP 服务器在 IDA Pro 与 AI 助手之间架起桥梁，提供全面的二进制分析和反混淆工具集。它支持交互式 IDA Pro 会话（通过插件）和独立的二进制分析（通过 idalib）。

## 架构

项目由三个主要组件组成：

1. **server.py** - 主 MCP 服务器，通过 HTTP JSON-RPC 与 IDA Pro 通信
2. **mcp-plugin.py** - 在 IDA Pro 内运行的插件，通过 JSON-RPC 暴露 API
3. **idalib_server.py** - 使用 idalib 进行非交互式分析的独立服务器

## 功能特性

### 核心能力

- 二进制元数据提取（哈希值、基地址、文件信息）
- 使用 Hex-Rays 反编译器进行函数反编译
- 带详细注释的汇编反汇编
- 内存读写
- 使用 Keystone Engine 进行代码修补（支持 ARM64）
- 基本块分析和控制流
- 注释管理
- 执行任意 IDA Python 代码

### 支持的架构

- x86/x64
- ARM64/AArch64（增强的修补支持）
- ARM32
- MIPS（32/64位）
- PowerPC（32/64位）
- SPARC（32/64位）

## 安装

### 前置要求

- Python 3.11 或更高版本
- 带 Python API 访问的 IDA Pro
- Hex-Rays 反编译器（可选，用于反编译功能）
- Keystone Engine（可选，用于汇编修补）

### 依赖项

安装所需的 Python 包：

```bash
pip install mcp fastmcp keystone-engine typing_inspection
```

或使用 uv：

```bash
uv pip install mcp fastmcp keystone-engine typing_inspection
```

### IDA 插件安装

使用 `--install` 标志运行服务器以自动安装：

```bash
python server.py --install
```

或使用 uv：

```bash
uv run server.py --install
```

这将会：
- 将 MCP 插件安装到 IDA Pro 的插件目录
- 配置 MCP 客户端（Cline、Claude Desktop、Cursor、Windsurf 等）

手动安装：
1. 将 `mcp-plugin.py` 复制到 IDA Pro 的插件目录：
   - Windows: `%APPDATA%\Hex-Rays\IDA Pro\plugins\`
   - macOS/Linux: `~/.idapro/plugins/`

## 使用方法

### 启动服务器

#### 交互模式（使用 IDA Pro 插件）

1. 启动 MCP 服务器：
```bash
python server.py
```

或使用 uv：
```bash
uv run server.py
```

2. 在 IDA Pro 中打开您的二进制文件

3. 激活插件：
   - 菜单：Edit -> Plugins -> MCP
   - 快捷键：Ctrl+Alt+M（Windows/Linux）或 Ctrl+Option+M（macOS）

4. 插件在 `127.0.0.1:13337` 启动 HTTP 服务器

#### 独立模式（使用 idalib）

```bash
python idalib_server.py /path/to/binary
```

此模式在没有 IDA Pro GUI 的情况下运行分析，并在端口 8745 上通过 HTTP 提供 MCP 服务。

### 命令行选项

```bash
# 安装插件并配置 MCP 客户端
python server.py --install

# 卸载插件并移除配置
python server.py --uninstall

# 生成 MCP 配置 JSON
python server.py --config

# 使用自定义 IDA RPC 服务器地址
python server.py --ida-rpc http://127.0.0.1:13337

# 启用不安全函数（危险）
python server.py --unsafe

# 使用 HTTP 传输进行调试
python server.py --transport http://127.0.0.1:8744
```

## 可用工具

### 安全函数（自动批准）

这些函数是只读的或具有最小副作用：

- `check_connection()` - 验证 IDA 插件是否运行
- `get_metadata()` - 获取二进制元数据（哈希值、基地址、大小）
- `get_function_by_address(address)` - 获取地址处的函数信息
- `get_current_address()` - 获取 IDA 中当前光标位置
- `get_current_function()` - 获取当前光标处的函数
- `convert_number(text, size)` - 在不同数字格式之间转换
- `decompile_function(address)` - 反编译为 C 伪代码
- `disassemble_function(start_address)` - 获取详细的汇编代码
- `read_memory_bytes(memory_address, size)` - 读取原始内存
- `get_basic_block(address)` - 获取基本块信息
- `get_blocks_referencing_block(address)` - 获取前驱基本块
- `get_blocks_referenced_by_block(address)` - 获取后继基本块

### 不安全函数（需要 `--unsafe` 标志）

这些函数会修改 IDA 数据库或执行任意代码：

- `set_comment(address, comment)` - 向反汇编/反编译器添加注释
- `patch_address_assembles(address, assembles)` - 使用汇编代码修补二进制文件
- `execute_ida_python_code(code)` - 执行任意 IDA Python 代码

## API 示例

### 获取函数信息

```python
# 获取函数元数据
result = get_function_by_address("0x401000")
# 返回: {"address": "0x401000", "name": "main", "size": "0x150"}
```

### 反编译函数

```python
# 反编译为 C 伪代码
pseudocode = decompile_function("0x401000")
# 返回带行号和地址的注释 C 代码
```

### 修补二进制文件（ARM64）

```python
# 修补多条指令
patch_address_assembles("0x184030", "b 0x184040; nop")
# 在连续地址处修补分支和 NOP
```

### 执行自定义分析

```python
# 运行任意 IDA Python 代码进行反混淆
code = """
encrypted_data = get_bytes(0x402000, 16)
key = 0xDEADBEEF
decrypted = []
for i, byte in enumerate(encrypted_data):
    decrypted.append(byte ^ ((key >> (i % 4) * 8) & 0xFF))
result = ''.join(chr(b) for b in decrypted if 32 <= b <= 126)
"""
result = execute_ida_python_code(code)
# 返回: {"success": True, "result": {"result": "decrypted_string"}}
```

## MCP 客户端配置

### Claude code/Cursor

```json
{
  "mcpServers": {
    "deobfuscation-mcp": {
      "command": "uv",
      "args": [
        "--directory",
        "D://Users//wenbochen.CN1//PycharmProjects//deobfuscation-mcp//deobfuscation-mcp-server",
        "run",
        "server.py",
        "--install-plugin"
      ],
      "timeout": 1800,
      "disabled": false
    }
  }
}
```

## 安全考虑

### 不安全函数

标记为不安全的函数可以：
- 永久修改 IDA 数据库
- 执行具有完整 IDA API 访问权限的任意 Python 代码
- 如果误用可能损坏二进制分析

在使用不安全函数之前，请始终备份您的 IDA 数据库。

### 代码执行

`execute_ida_python_code` 函数提供对以下内容的不受限制的访问：
- 完整的 IDA Python API
- 文件系统操作
- 网络操作
- 数据库修改

仅在必要时使用 `--unsafe` 标志启用，并了解风险。

## 架构详情

### 线程安全

插件使用 IDA 的同步装饰器进行线程安全操作：
- `@idaread` - 使用 `MFF_READ` 安全级别的读操作
- `@idawrite` - 使用 `MFF_WRITE` 安全级别的写操作

### JSON-RPC 协议

服务器通过 HTTP 实现 JSON-RPC 2.0：
- 端点：`http://127.0.0.1:13337/mcp`
- Content-Type：`application/json`
- 错误代码遵循 JSON-RPC 规范

### 自动生成的代码

`server_generated.py` 从 `mcp-plugin.py` 注释自动生成：
- 提取 `@jsonrpc` 装饰的函数
- 转换为 FastMCP 工具
- 从类型注释添加 Pydantic Field 描述

## 故障排除

### 插件无法启动

检查 IDA 的输出窗口以查看错误消息：
- 确保使用 Python 3.11+
- 验证所有依赖项已安装
- 检查端口 13337 是否可用

### 连接失败

```
Failed to connect to IDA Pro! Did you run Edit -> Plugins -> MCP (Ctrl+Alt+M)?
```

解决方案：
1. 确保 IDA Pro 正在运行
2. 激活 MCP 插件（Ctrl+Alt+M）
3. 检查防火墙设置

### 反编译器许可证错误

如果您没有 Hex-Rays 许可证：
- 使用 `disassemble_function()` 代替 `decompile_function()`
- 汇编视图提供了大部分所需信息

### 汇编修补错误

对于 ARM64 分支指令：
- 使用绝对地址格式：`b 0x184030`
- 确保 4 字节对齐
- 检查目标是否在 ±128MB 范围内

## 开发

### 添加新工具

1. 在 `mcp-plugin.py` 中添加带有 `@jsonrpc` 装饰器的函数：

```python
@jsonrpc
@idaread  # 对于修改使用 @idawrite
def my_new_tool(param: Annotated[str, "参数描述"]) -> ReturnType:
    """LLM 的工具描述"""
    # 实现
    return result
```

2. 重启服务器 - `server_generated.py` 会自动更新

3. 可选标记为不安全：

```python
@jsonrpc
@unsafe  # 需要 --unsafe 标志
@idawrite
def dangerous_operation():
    # 实现
```

### 测试

使用 MCP Inspector 进行调试：

```bash
# 使用 HTTP 传输启动服务器
python server.py --transport http://127.0.0.1:8744

# 连接 MCP Inspector
npx @modelcontextprotocol/inspector
```

## 常见问题

### Q: 如何在 Claude Desktop 中使用？

A:
1. 编辑 Claude Desktop 配置文件（`%APPDATA%\Claude\claude_desktop_config.json`）
2. 添加上述 MCP 服务器配置
3. 重启 Claude Desktop
4. 在 IDA Pro 中打开二进制文件并激活 MCP 插件（Ctrl+Alt+M）

### Q: uv 和 python 命令有什么区别？

A:
- `uv` 是一个更快的 Python 包管理器，可以自动管理虚拟环境
- `python` 是标准的 Python 解释器
- 两者都可以使用，`uv` 更推荐用于依赖管理

### Q: 如何启用不安全函数？

A: 在配置中的 args 数组中添加 `"--unsafe"`：

```json
"args": [
  "--directory",
  "D://path//to//server",
  "run",
  "server.py",
  "--unsafe"
]
```

### Q: 支持哪些 MCP 客户端？

A: 目前支持：
- Claude Desktop
- Cline（VSCode 扩展）
- Roo Code（VSCode 扩展）
- Cursor
- Windsurf
- Claude Code
- LM Studio

## 许可证

基于 [ida-pro-mcp](https://github.com/mrexodia/ida-pro-mcp)（MIT 许可证）

## 致谢

- 架构基于 mrexodia 的 ida-pro-mcp 项目
- 使用 FastMCP 进行 MCP 协议实现
- 使用 Keystone Engine 进行汇编修补
- 使用 IDA Pro SDK 提供逆向工程能力

## 贡献

欢迎贡献！改进方向：

- 额外的架构支持
- 更多反混淆工具
- 增强的修补能力
- 更好的错误处理
- 文档改进

## 支持

如有问题和功能请求，请参考原始 ida-pro-mcp 项目或在此仓库中提出问题。
