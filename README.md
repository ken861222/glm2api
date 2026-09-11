# ChatGLM 2 API (`glm2api`)

把 `chatglm.cn` 的网页接口转换成 **OpenAI / Anthropic 兼容接口** 的本地代理服务，方便直接接入 OpenAI SDK、Cherry Studio、Open WebUI、LobeChat、Claude Code、Codex 等工具。

> **本仓库是 [XxxXTeam/glm2api](https://github.com/XxxXTeam/glm2api) 的 Fork。**
> 上游作者：[@XxxXTeam](https://github.com/XxxXTeam)。原项目采用 GPL-3.0 授权，本 Fork 保留同一许可与完整提交历史。
>
> 本 Fork 相对上游新增了两件事：
> 1. **修复流式输出乱序 / 串字 / 丢字的 Bug**（详见 [§3](#3-流式输出-bug-修复本-fork-的主要改动)）
> 2. **补齐 Docker 部署能力**（`Dockerfile` + `docker-compose.yml`，上游没有）

---

## 目录

- [1. 功能概览](#1-功能概览)
- [2. 支持的接口](#2-支持的接口)
- [3. 流式输出 Bug 修复（本 Fork 的主要改动）](#3-流式输出-bug-修复本-fork-的主要改动)
- [4. 快速开始 —— Docker 部署（推荐）](#4-快速开始--docker-部署推荐)
- [5. 快速开始 —— 本地运行](#5-快速开始--本地运行)
- [6. 获取 `refresh_token`](#6-获取-refresh_token)
- [7. 配置项说明](#7-配置项说明)
- [8. 接口用法示例](#8-接口用法示例)
- [9. 鉴权](#9-鉴权)
- [10. 日志与排错](#10-日志与排错)
- [11. 回归测试](#11-回归测试)
- [12. 常见问题](#12-常见问题)
- [13. 风险与免责声明](#13-风险与免责声明)
- [14. 许可证](#14-许可证)

---

## 1. 功能概览

- OpenAI 兼容的 `/v1/chat/completions`（含 SSE 流式）
- OpenAI Responses API：`/v1/responses`
- Anthropic Messages API：`/v1/messages`（可给 Claude Code 类客户端用）
- 图片生成：`/v1/images/generations`
- 工具调用（Function Calling）：把 GLM 网页端的 `tool_calls` 与 DSML/XML 工具协议互相转换
- 多账号轮转 + 并发槽位排队
- 游客模式（不登录也能用，但额度与稳定性受限）
- 纯 Python 标准库实现，**运行时零第三方依赖**

## 2. 支持的接口

| 方法 | 路径 | 是否需要鉴权 | 说明 |
|---|---|---|---|
| `GET` | `/health` | 否 | 健康检查，返回 `{"status":"ok"}` |
| `GET` | `/v1/models` | 否 | 列出当前暴露的模型 |
| `POST` | `/v1/chat/completions` | 是 | OpenAI 聊天补全，支持 `stream` |
| `POST` | `/v1/responses` | 是 | OpenAI Responses API |
| `POST` | `/v1/messages` | 是 | Anthropic Messages API |
| `POST` | `/v1/images/generations` | 是 | 图片生成 |

> 鉴权规则：只有 `.env` 里的 `SERVER_API_KEYS` 非空时才校验 `Authorization: Bearer <key>`。
> `GET /health` 与 `GET /v1/models` 始终不校验（上游项目即如此设计）。

---

## 3. 流式输出 Bug 修复（本 Fork 的主要改动）

### 3.1 现象

同一个模型、同一个问题：

- **非流式**（`stream: false`）输出完全正常；
- **流式**（`stream: true`）输出串字、丢字、标点错位。

修复前的真实输出示例：

```text
用户：我帮您查一下杭州今天的天气
流式：我一下天的查一下杭州今天的天气        ← 语序错乱、字被吞

用户：请写一句七言诗并解释
流式：【turn0加大**：凉意渐显，起来，h4】   ← 括号内容错位、Markdown 结构损坏
```

### 3.2 根因

上游 `chatglm.cn` 对**同一个 `logic_id`** 会混发两种形态的数据：

| SSE `status` | 含义 | 示例 |
|---|---|---|
| `init` | **增量碎片**，每次只发一小段 | `"# "` → `"Pyth"` → `"on冒泡"` → … |
| `finish` | **完整快照**，重复整段内容（有时连发 2 遍） | `"# Python冒泡排序教程\n\n## 📚 目录…"` |

上游原实现的 `_compute_deltas()` 用**「长度差」**来切增量：

```python
# ❌ 上游原实现（有 Bug）
elif len(rendered_text) > prev_len:
    text_delta_parts.append(rendered_text[prev_len:])   # 把碎片当成累积文本
```

它假设「每次收到的是到目前为止的完整文本」，但上游 `init` 发的是**碎片**而不是累积值。
于是 `prev_len` 记录的是「上一次碎片的长度」，而新碎片按字符下标去切，就切出了乱序和丢字。

**为什么非流式正常**：非流式走 `build_response()` → `_render_full_output()`，只取每个 `logic_id` 的**最新 part**（也就是那条完整快照），天然绕开了这个 Bug。

### 3.3 修法

改为**折叠（fold）**策略，并新增两个模块级函数：

```python
def fold_incremental_into(sink, accumulators, logic_id, incoming, authoritative=False) -> None:
    """把 incoming 折叠进 accumulators[logic_id]，只把新增后缀推进 sink。"""
    previous = accumulators.get(logic_id, "")
    updated = fold_incremental(previous, incoming, authoritative=authoritative)
    if len(updated) > len(previous):
        sink.append(updated[len(previous):])
    accumulators[logic_id] = updated


def fold_incremental(accumulated: str, incoming: str, authoritative: bool = False) -> str:
    if not incoming:
        return accumulated
    if not accumulated:
        return incoming
    if accumulated.startswith(incoming):        # 陈旧回放，丢弃
        return accumulated
    if incoming.startswith(accumulated):        # 超集重写（"#" → "# 快"），替换
        return incoming
    if authoritative:                           # finish 快照与已发内容冲突：不追加
        return accumulated                      # （已流出的字收不回，只能维持现状）
    return accumulated + incoming               # 正常碎片：追加
```

三条关键规则：

1. **`init` 碎片追加**，但若新片段是当前累积值的超集（`"#"` → `"# 快"`）则替换，避免重复。
2. **`finish` 快照为权威值**；若它与已流出的内容冲突，则**不追加**——因为流式语义无法「收回」已经发出去的字符。这样既不会重复整段，也不会把答案搞乱。
3. **纯空白碎片必须保留**。上游会用一条独立的 `"\n\n"` 事件来分隔段落；如果按「去掉空白后为空就跳过」处理，段落会被粘成一坨：

```text
修复前： …随机排列的数组 |### 空间复杂度
修复后： …随机排列的数组 |
                         ### 空间复杂度
```

配套改动：

- 新增 `_accum_text` / `_accum_reasoning`，按 `logic_id` 记录折叠后的已发文本
- 新增 `_part_status`，记录每个 part 最近的 SSE `status`，用于判断 `authoritative`
- 新增 `_cached_part_raw_texts` / `_cached_part_raw_reasonings`：折叠必须基于**未 strip 的原文**，否则 `"关不住**\n\n"` 与 `"**解释：**"` 会被粘在一起，导致尾部的快照又被误判成新内容

### 3.4 效果

用真实上游抓包重放 6 组用例（`translator.py` 修复前后对比）：

| 用例 | 修复前 | 修复后 |
|---|---|---|
| 七言诗解释 | ✅ | ✅ |
| 杭州天气 | ✅ | ✅ |
| 量子纠缠 | ✅ | ✅ |
| 数到 5 | ✅ | ✅ |
| 冒泡排序（长文 + 工具调用） | ❌ 4311 字（应为 3780） | ✅ 3780 字 |
| 1–20 打油诗（碎片密集） | ❌ 92 字（应为 63，含重复段） | ✅ 63 字 |

修复后**流式输出与非流式输出逐字节一致**。

---

## 4. 快速开始 —— Docker 部署（推荐）

上游仓库**没有** `Dockerfile`，本 Fork 补了 `Dockerfile` 与 `docker-compose.yml`。

### 4.1 准备

- Docker 20.10+ / Docker Compose v2（`docker compose` 或 `docker-compose`）
- 一个可用的 `refresh_token`（见 [§6](#6-获取-refresh_token)）

```bash
git clone https://github.com/ken861222/glm2api.git
cd glm2api
```

### 4.2 配置

```bash
cp .env.example .env
mkdir -p data
```

编辑 `.env`，至少填这两项：

```env
SERVER_API_KEYS=sk-glm-换成你自己的随机字符串
PORT=8001                 # 宿主机映射端口；不填默认 8001
```

然后把你的 `refresh_token` 写进 `data/token.txt`（一行一个账号）：

```bash
echo "你的_refresh_token" > data/token.txt
```

> **权限注意**：容器内以 uid `10001` 运行。若宿主机上 `data/` 属主不对，容器启动会报 `Permission denied`。修正：
> ```bash
> sudo chown -R 10001:10001 data
> ```

### 4.3 启动

```bash
docker-compose up -d --build
docker-compose ps
curl http://127.0.0.1:8001/health
```

看到 `{"status":"ok"}` 即成功。

### 4.4 验证流式输出

```bash
curl -N http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-glm-换成你自己的随机字符串" \
  -d '{"model":"glm-4-flash","stream":true,
       "messages":[{"role":"user","content":"请写一句七言诗，并解释意思"}]}'
```

应当看到逐块输出、语义连贯、结尾无重复段落。

### 4.5 常用运维命令

```bash
docker-compose ps                                  # 查看状态
docker-compose logs -f glm2api                     # 跟踪日志
docker-compose restart                             # 重启
docker-compose down                                # 停止并删除容器
docker-compose up -d --build                       # ★ 改源码后必须用这个
```

> ⚠️ **源码是 `COPY` 进镜像的**，只改宿主机源码后执行 `restart` **不会生效**，必须 `up -d --build` 重新构建。

### 4.6 目录结构

```text
glm2api/
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── src/glm2api/            # 源码
├── tests/                  # 回归测试
├── data/                   # （挂载进容器）放 token.txt
└── log/                    # （挂载进容器）日志
```

---

## 5. 快速开始 —— 本地运行

需要 Python 3.10+（开发环境用的是 3.13 / 3.14）。

```bash
git clone https://github.com/ken861222/glm2api.git
cd glm2api
cp .env.example .env
uv sync
uv run main.py
```

或者不装 `uv`：

```bash
pip install -e .
python main.py
```

启动成功后日志类似：

```text
启动服务 host=127.0.0.1 port=8000 prefix=/v1 accounts=1 debug_dump_all=False models=...
```

> 想局域网访问，把 `.env` 里的 `HOST` 改成 `0.0.0.0`。

---

## 6. 获取 `refresh_token`

1. 浏览器登录 `https://chatglm.cn`
2. 按 `F12` 打开开发者工具
3. 进入 **Application → Local Storage**
4. 找到 `chatglm_refresh_token`，复制其值

> 也可以走游客模式，不登录：
> ```env
> GLM_USE_GUEST_REFRESH_TOKEN=true
> ```
> 游客模式会按 `GLM_MAX_CONCURRENCY` 预建同等数量的游客槽位，额度与稳定性都弱于登录账号。

---

## 7. 配置项说明

配置文件为 `.env`（不存在时会自动从 `.env.example` 复制一份）。

### 服务基础

| 变量 | 默认 | 说明 |
|---|---|---|
| `HOST` | `127.0.0.1` | 监听地址，局域网访问填 `0.0.0.0` |
| `PORT` | `8000` | 服务端口（Docker 里是容器内端口，宿主机映射见 `docker-compose.yml`） |
| `API_PREFIX` | `/v1` | OpenAI 兼容路径前缀 |
| `LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `DEBUG_DUMP_ALL` | `false` | 调试狂暴模式，打印入站请求、上游响应、SSE 原文等 |
| `REQUEST_TIMEOUT_SECONDS` | `120` | 上游请求超时（长回答 / 联网会较慢） |
| `CORS_ALLOW_ORIGIN` | `*` | CORS 允许来源 |
| `SERVER_API_KEYS` | 空 | 本服务自身的 Bearer 鉴权，多个用英文逗号分隔；留空则不校验 |

### 账号与并发

| 变量 | 默认 | 说明 |
|---|---|---|
| `GLM_TOKEN_FILE` | `token.txt` | 多账号文件，每行一个 `refresh_token` |
| `GLM_REFRESH_TOKEN` | 空 | 单账号兜底；上游返回新 token 时会自动写回 `.env` |
| `GLM_USE_GUEST_REFRESH_TOKEN` | `false` | 显式启用游客模式，忽略已配置账号 |
| `GLM_GUEST_MAX_RETRIES` | `3` | 游客 token 失败时重新拉取并重试的次数 |
| `GLM_MAX_CONCURRENCY` | `3` | 同时占用的上游执行槽位数（**同时决定预建多少游客槽位**） |
| `GLM_QUEUE_WAIT_TIMEOUT_SECONDS` | `600` | 超出并发时的排队等待上限 |
| `GLM_BUSY_MAX_RETRIES` | `30` | 上游返回「请等待其他对话生成完毕」时的重试次数 |
| `GLM_BUSY_RETRY_INTERVAL_SECONDS` | `2` | 上述重试的间隔秒数 |
| `GLM_DELETE_CONVERSATION` | `true` | 请求结束后是否删除上游会话记录 |

### 上游与工具

| 变量 | 默认 | 说明 |
|---|---|---|
| `GLM_BASE_URL` | `https://chatglm.cn/chatglm` | 上游地址，一般不改 |
| `GLM_ASSISTANT_ID` | `65940acf…` | 普通对话的 assistant id |
| `GLM_IMAGE_ASSISTANT_ID` | `65a232c0…` | 图片生成的 assistant id |
| `GLM_USER_AGENT` | Chrome UA | 自定义 User-Agent |
| `BLOCKED_TOOL_NAMES` | 见 `.env.example` | 工具黑名单，不注入提示词、服务端也丢弃 |

> `.env.example` 里 `GLM_MAX_CONCURRENCY` 的示例值写的是 `100`。**不建议照抄**：它会预建 100 个游客槽位，资源占用高、风控风险大。生产环境建议从 `3` 起步。

账号选择逻辑：

- 存在 `token.txt` → 优先用文件里的多账号，失败自动切下一个
- 设置了 `GLM_USE_GUEST_REFRESH_TOKEN=true` → 直接走游客
- 什么都没配 → 自动获取游客 token 兜底
- 多头模式下上游下发的新 `refresh_token` 会写回 `token.txt` 对应行；单账号模式写回 `.env`

---

## 8. 接口用法示例

### 8.1 聊天补全（curl）

```bash
curl http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-glm-xxx" \
  -d '{"model":"glm-4","messages":[{"role":"user","content":"你好，介绍一下你自己"}]}'
```

### 8.2 聊天补全（OpenAI SDK）

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8001/v1", api_key="sk-glm-xxx")

resp = client.chat.completions.create(
    model="glm-4",
    messages=[{"role": "user", "content": "你好，介绍一下你自己"}],
)
print(resp.choices[0].message.content)
```

### 8.3 流式（OpenAI SDK）

```python
stream = client.chat.completions.create(
    model="glm-4",
    messages=[{"role": "user", "content": "写一首七言绝句"}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta
    if getattr(delta, "content", None):
        print(delta.content, end="", flush=True)
```

### 8.4 Responses API

```python
resp = client.responses.create(
    model="glm-4",
    input=[{"role": "user", "content": "你好，介绍一下你自己"}],
)
print(resp.output_text)
```

### 8.5 Anthropic Messages API（Claude Code 类客户端）

```bash
curl http://127.0.0.1:8001/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-glm-xxx" \
  -d '{"model":"glm-4","max_tokens":1024,
       "messages":[{"role":"user","content":"你好"}]}'
```

### 8.6 图片生成

```bash
curl http://127.0.0.1:8001/v1/images/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-glm-xxx" \
  -d '{"model":"glm-image-1","prompt":"画个枫叶","size":"1024x1024"}'
```

```python
image = client.images.generate(model="glm-image-1", prompt="画个枫叶", size="1024x1024")
print(image.data[0].url)
```

支持参数：`prompt`、`model`、`n`、`size`、`response_format`、`style`、`scene`。
默认返回 URL；`response_format=b64_json` 时返回 base64。`size` 会自动映射到上游宽高比。

### 8.7 模型列表

```bash
curl http://127.0.0.1:8001/v1/models
```

---

## 9. 鉴权

- `SERVER_API_KEYS` 留空 → 本地接口不校验（仅建议本机自用）
- `SERVER_API_KEYS=sk-local-1,sk-local-2` → 请求需带 `Authorization: Bearer sk-local-1`
- Anthropic 客户端支持 `x-api-key` 头

**对外暴露时务必设置 `SERVER_API_KEYS`，并配合防火墙白名单**（例如只放行内网网段）。

---

## 10. 日志与排错

默认输出彩色日志，包含：请求入队、并发槽位申请/释放、上游转发、会话删除、错误原因。

排查流式问题时，把 `.env` 打开调试：

```env
LOG_LEVEL=DEBUG
DEBUG_DUMP_ALL=true
```

`DEBUG_DUMP_ALL=true` 会自动切到 `DEBUG`，并额外打印**上游原始 SSE 分片**——这正是定位本 Fork 那个流式 Bug 的手段（可以看到 `init` 碎片与 `finish` 快照的交替）。

此时日志会同时写入 `log/glm2api_debug.log`。

---

## 11. 回归测试

测试是纯 Python、无第三方依赖，直接跑：

```bash
PYTHONPATH=src python3 -m pytest tests/ -q
```

```text
65 passed
```

其中 `tests/test_translator.py` 里有 **5 个针对流式折叠的回归用例**，全部来自真实抓包形态：

| 用例 | 覆盖点 |
|---|---|
| `test_accumulator_streaming_folds_fragments_then_snapshot_without_duplicating` | 碎片 + 快照不重复 |
| `test_accumulator_streaming_counts_increments_then_snapshot` | 数字递增型碎片 |
| `test_accumulator_streaming_fragment_that_extends_previous_is_not_swallowed` | 超集重写（`"#"` → `"# 快"`）不被吞 |
| `test_accumulator_streaming_divergent_finish_snapshot_is_not_appended` | 冲突快照不追加 |
| `test_accumulator_streaming_preserves_whitespace_only_fragment` | 纯空白碎片保留段落 |

> 想更彻底地验证，可以用 `DEBUG_DUMP_ALL=true` 抓一段真实 SSE，再用 `GLMEventAccumulator.consume_event()` 逐条重放，断言「流式拼接结果 == 最后那条 `finish` 快照」。

---

## 12. 常见问题

### 12.1 启动报 `GLM_REFRESH_TOKEN` 缺失

新版本会自动退回游客模式。若想固定用账号，检查 `.env` 的 `GLM_REFRESH_TOKEN` 或 `data/token.txt`。

### 12.2 容器启动报 `Permission denied`

`data/` 属主不对。容器内以 uid `10001` 运行：

```bash
sudo chown -R 10001:10001 data
```

### 12.3 改了源码但行为没变

源码是 `COPY` 进镜像的。必须重建：

```bash
docker-compose up -d --build
```

### 12.4 返回「请等待其他对话生成完毕」

同一账号在上游有并发限制。程序内置串行队列与自动等待重试，可调大 `GLM_BUSY_MAX_RETRIES` / `GLM_BUSY_RETRY_INTERVAL_SECONDS`。

### 12.5 返回「请登录后继续使用」

账号状态无效或 token 失效，重新登录并更新 `refresh_token`。

### 12.6 流式仍然乱序？

确认容器里跑的是本 Fork 的代码：

```bash
docker exec glm2api grep -c fold_incremental /app/src/glm2api/services/translator.py
```

返回 `> 0` 才算打了补丁。

### 12.7 `git pull` 后补丁不见了

本 Fork 的 `fold_incremental` 改动在上游是不存在的。如果你从**上游**拉更新覆盖了 `translator.py`，补丁会丢失——重新应用本仓库的 `translator.py` 并 `up -d --build` 即可。

---

## 13. 风险与免责声明

1. **违反上游服务条款**。本项目把 `chatglm.cn` 的 Web 端额度转给第三方客户端使用，这可能违反智谱清言的服务条款，存在账号被限流或封禁的风险。
2. **不要用于批量注册 / 滥用**。本项目设计用途是「用你自己已登录的账号做本地代理」，请勿接入批量注册、验证码绕过、账号池买卖等场景。
3. **`build_random_x_forwarded_for()` 存在高风险**。`glm_auth.py` 会为请求伪造随机 `X-Forwarded-For` 头。这是规避上游 IP 维度风控的手段，会显著提高风控命中与封号概率。如无必要建议不要依赖。
4. **默认无鉴权**。`SERVER_API_KEYS` 为空时任何人都能调用，公网暴露前务必配置鉴权 + 防火墙白名单。
5. **登录凭据安全**。`refresh_token` 等同于账号凭据，不要提交进 git、不要贴在公开渠道。`.gitignore` 已忽略 `.env`、`log`、`docs`，请勿手动强制添加。

本项目仅供**个人学习与技术研究**使用，请自行承担使用风险。

---

## 14. 许可证

[GPL-3.0](./LICENSE)，与上游保持一致。

- 上游项目：[XxxXTeam/glm2api](https://github.com/XxxXTeam/glm2api)
- 本 Fork：[ken861222/glm2api](https://github.com/ken861222/glm2api)
