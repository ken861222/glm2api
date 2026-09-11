import json

from glm2api.services.translator import (
    BLOCKED_NATIVE_TOOL_NAMES,
    GLMEventAccumulator,
    convert_messages,
    sanitize_tool_call_payload,
)


def test_convert_messages_injects_xml_tool_prompt_and_history():
    converted = convert_messages(
        messages=[
            {"role": "user", "content": "查天气"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"上海"}',
                        }
                    }
                ],
            },
            {
                "role": "tool",
                "name": "get_weather",
                "tool_call_id": "call_1",
                "content": "晴",
            },
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "查询天气",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
    )

    prompt = converted[0]["content"][0]["text"]

    assert "<|DSML|tool_calls>" in prompt
    assert "<|DSML|invoke name=\"get_weather\">" in prompt
    assert "<|DSML|tool_result call_id=\"call_1\" name=\"get_weather\">" in prompt
    assert "<ml_tool_calls>" not in prompt
    assert "# TOOL USE PROTOCOL" in prompt
    assert "Use the DSML format below exactly." in prompt
    assert "The server will parse this DSML block back into standard OpenAI tool_calls." in prompt
    assert "<|DSML|parameter name=\"actual_parameter_name\"><![CDATA[value]]></|DSML|parameter>" in prompt
    assert "Each argument must be a <|DSML|parameter name=\"...\"> child of the invoke." in prompt
    assert "Parameter names are case-sensitive and must exactly match the schema." in prompt
    assert "never change it to `filepath`, `file_path`, or `FilePath`." in prompt
    assert "# BLOCKED TOOLS" not in prompt
    assert "Ignore any tool names that are not listed below" in prompt


def test_accumulator_build_response_maps_xml_to_openai_tool_calls():
    accumulator = GLMEventAccumulator(model="glm-test", allowed_tool_names={"get_weather"})
    accumulator.consume_event(
        {
            "conversation_id": "conv_1",
            "parts": [
                {
                    "logic_id": "1",
                    "content": [
                        {
                            "type": "text",
                            "text": "<|DSML|tool_calls><|DSML|invoke name=\"get_weather\">"
                            "<|DSML|parameter name=\"city\">上海</|DSML|parameter>"
                            "</|DSML|invoke></|DSML|tool_calls>",
                        }
                    ],
                }
            ],
        }
    )

    response = accumulator.build_response()
    message = response["choices"][0]["message"]

    assert response["choices"][0]["finish_reason"] == "tool_calls"
    assert message["content"] is None
    assert message["tool_calls"][0]["function"]["name"] == "get_weather"
    assert message["tool_calls"][0]["function"]["arguments"] == '{"city":"上海"}'


def test_accumulator_streaming_tool_call_emits_assistant_role_before_tool_delta():
    accumulator = GLMEventAccumulator(model="glm-test", allowed_tool_names={"write"})
    chunks, status = accumulator.consume_event(
        {
            "conversation_id": "conv_1",
            "status": "finish",
            "parts": [
                {
                    "logic_id": "1",
                    "content": [
                        {
                            "type": "text",
                            "text": "<|DSML|tool_calls><|DSML|invoke name=\"write\">"
                            "<|DSML|parameter name=\"filePath\">test.txt</|DSML|parameter>"
                            "<|DSML|parameter name=\"content\"></|DSML|parameter>"
                            "</|DSML|invoke></|DSML|tool_calls>",
                        }
                    ],
                }
            ],
        }
    )

    final_chunks = accumulator.finalize(status)

    assert chunks == []
    assert '"delta":{"role":"assistant"}' in final_chunks[0]
    assert '"tool_calls"' in final_chunks[1]
    assert '"finish_reason":"tool_calls"' in final_chunks[2]


def test_accumulator_streaming_extracts_tool_call_from_reasoning_fallback():
    accumulator = GLMEventAccumulator(model="glm-test", allowed_tool_names={"write"})
    chunks, status = accumulator.consume_event(
        {
            "conversation_id": "conv_1",
            "status": "finish",
            "parts": [
                {
                    "logic_id": "1",
                    "content": [
                        {
                            "type": "think",
                            "think": "I should call the tool.\n"
                            "<|DSML|tool_calls><|DSML|invoke name=\"write\">"
                            "<|DSML|parameter name=\"filePath\">test.txt</|DSML|parameter>"
                            "<|DSML|parameter name=\"content\"></|DSML|parameter>"
                            "</|DSML|invoke></|DSML|tool_calls>",
                        }
                    ],
                }
            ],
        }
    )

    final_chunks = accumulator.finalize(status)

    assert chunks
    assert '"reasoning_content"' in chunks[0]
    assert '"delta":{"role":"assistant"}' in final_chunks[0]
    assert '"tool_calls"' in final_chunks[1]
    assert '\\"filePath\\":\\"test.txt\\"' in final_chunks[1]


def test_accumulator_build_response_extracts_tool_call_from_reasoning_fallback():
    accumulator = GLMEventAccumulator(model="glm-test", allowed_tool_names={"write"})
    accumulator.consume_event(
        {
            "conversation_id": "conv_1",
            "parts": [
                {
                    "logic_id": "1",
                    "content": [
                        {
                            "type": "think",
                            "think": "<|DSML|tool_calls><|DSML|invoke name=\"write\">"
                            "<|DSML|parameter name=\"filePath\">test.txt</|DSML|parameter>"
                            "<|DSML|parameter name=\"content\"></|DSML|parameter>"
                            "</|DSML|invoke></|DSML|tool_calls>",
                        }
                    ],
                }
            ],
        }
    )

    response = accumulator.build_response()
    message = response["choices"][0]["message"]

    assert response["choices"][0]["finish_reason"] == "tool_calls"
    assert message["content"] is None
    assert message["tool_calls"][0]["function"]["name"] == "write"
    assert message["tool_calls"][0]["function"]["arguments"] == '{"filePath":"test.txt","content":""}'


def test_sanitize_shell_command_argument_from_json_string():
    cleaned = sanitize_tool_call_payload(
        "shell",
        {
            "command": '["powershell.exe","-Command","Get-ChildItem -Force"]',
            "workdir": "E:\\Projects\\2api\\glm2api",
        },
    )

    assert cleaned == {
        "command": ["powershell.exe", "-Command", "Get-ChildItem -Force"],
        "workdir": "E:\\Projects\\2api\\glm2api",
    }


def test_sanitize_shell_command_argument_from_quoted_sequence():
    cleaned = sanitize_tool_call_payload(
        "shell",
        {
            "command": '"powershell.exe", "-Command", "Get-ChildItem -Force"',
        },
    )

    assert cleaned == {
        "command": ["powershell.exe", "-Command", "Get-ChildItem -Force"],
    }


def test_sanitize_shell_command_argument_from_plain_string():
    cleaned = sanitize_tool_call_payload(
        "shell",
        {
            "command": "Get-ChildItem",
        },
    )

    assert cleaned == {
        "command": ["powershell.exe", "-Command", "Get-ChildItem"],
    }


def test_sanitize_shell_command_argument_wraps_powershell_cmdlet_array():
    cleaned = sanitize_tool_call_payload(
        "shell",
        {
            "command": ["Get-ChildItem", "-Recurse", "-Filter", "*.txt"],
        },
    )

    assert cleaned == {
        "command": ["powershell.exe", "-Command", "Get-ChildItem -Recurse -Filter *.txt"],
    }


def test_sanitize_shell_command_argument_keeps_native_executable_array():
    cleaned = sanitize_tool_call_payload(
        "shell",
        {
            "command": ["git", "status", "--short"],
        },
    )

    assert cleaned == {
        "command": ["git", "status", "--short"],
    }


def test_accumulator_drops_tool_preamble_and_repairs_shell_command_array():
    accumulator = GLMEventAccumulator(model="glm-test", allowed_tool_names={"shell"})
    chunks, status = accumulator.consume_event(
        {
            "conversation_id": "conv_1",
            "status": "finish",
            "parts": [
                {
                    "logic_id": "1",
                    "content": [
                        {
                            "type": "text",
                            "text": "我将创建文件。\n\n"
                            '<|DSML|tool_calls><|DSML|invoke name="shell">'
                            '<|DSML|parameter name="command"><![CDATA[["powershell.exe", "-Command", "pwd"]]></|DSML|parameter>'
                            "</|DSML|invoke></|DSML|tool_calls>",
                        }
                    ],
                }
            ],
        }
    )

    final_chunks = accumulator.finalize(status)

    assert chunks == []
    assert "我将创建文件" not in "".join(final_chunks)
    assert '"tool_calls"' in final_chunks[1]
    assert '\\"command\\":[\\"powershell.exe\\",\\"-Command\\",\\"pwd\\"]' in final_chunks[1]


def test_accumulator_defers_visible_text_when_tools_available():
    accumulator = GLMEventAccumulator(model="glm-test", allowed_tool_names={"shell"})
    chunks, status = accumulator.consume_event(
        {
            "conversation_id": "conv_1",
            "status": "finish",
            "parts": [
                {
                    "logic_id": "1",
                    "content": [{"type": "text", "text": "你好"}],
                }
            ],
        }
    )

    final_chunks = accumulator.finalize(status)

    assert chunks == []
    assert '"content":"你好"' in final_chunks[0]
    assert '"finish_reason":"stop"' in final_chunks[1]


def test_accumulator_reports_unavailable_dsml_tool_instead_of_empty_response():
    accumulator = GLMEventAccumulator(model="glm-test", allowed_tool_names={"shell"})
    chunks, status = accumulator.consume_event(
        {
            "conversation_id": "conv_1",
            "status": "finish",
            "parts": [
                {
                    "logic_id": "1",
                    "content": [
                        {
                            "type": "text",
                            "text": '<|DSML|tool_calls><|DSML|invoke name="search">'
                            '<|DSML|parameter name="search_query"><![CDATA[{"q":"阿房宫赋","recency":365}]></|DSML|parameter>'
                            "</|DSML|invoke></|DSML|tool_calls>",
                        }
                    ],
                }
            ],
        }
    )

    final_chunks = accumulator.finalize(status)

    assert chunks == []
    assert "未声明工具" in final_chunks[0]
    assert "`search`" in final_chunks[0]
    assert '"finish_reason":"stop"' in final_chunks[1]


def test_convert_messages_respects_tool_choice_none_and_specific():
    none_converted = convert_messages(
        messages=[{"role": "user", "content": "直接回答"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "查询天气",
                    "parameters": {"type": "object"},
                },
            }
        ],
        tool_choice="none",
    )
    none_prompt = none_converted[0]["content"][0]["text"]
    assert "# TOOL SCHEMAS" not in none_prompt

    specific_converted = convert_messages(
        messages=[{"role": "user", "content": "查天气"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "查询天气",
                    "parameters": {"type": "object"},
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )
    specific_prompt = specific_converted[0]["content"][0]["text"]
    assert "You must call exactly `get_weather` before giving a final answer." in specific_prompt


def test_convert_messages_filters_native_url_tools_and_reinforces_tool_awareness():
    converted = convert_messages(
        messages=[{"role": "user", "content": "打开 https://example.com"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "open_url",
                    "description": "Open URL",
                    "parameters": {"type": "object"},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "mcp__CherryFetch__fetchJson",
                    "description": "Fetch JSON",
                    "parameters": {"type": "object"},
                },
            },
        ],
        blocked_tool_names=BLOCKED_NATIVE_TOOL_NAMES,
    )

    prompt = converted[0]["content"][0]["text"]

    assert "Tool: open_url" not in prompt
    assert "Server-side native tools" not in prompt
    assert "Tool: mcp__CherryFetch__fetchJson" in prompt
    assert "You do not have hidden browser, web, or URL-opening tools." in prompt
    assert "Never call native tools such as `open_url`" in prompt
    assert "Do not output hidden reasoning, chain-of-thought, or labels such as `Thinking:`." in prompt
    assert "Do not narrate tool selection, failed tool attempts, retries, fallback plans, or tool status banners." in prompt
    assert "Never output tool-call display text such as `⚙ tool_name [...]`" in prompt


def test_convert_messages_drops_blocked_tool_call_history():
    converted = convert_messages(
        messages=[
            {"role": "user", "content": "打开 https://example.com"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_bad",
                        "function": {
                            "name": "open_url",
                            "arguments": '{"url":"https://example.com"}',
                        },
                    }
                ],
            },
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "mcp__CherryFetch__fetchJson",
                    "description": "Fetch JSON",
                    "parameters": {"type": "object"},
                },
            }
        ],
    )

    prompt = converted[0]["content"][0]["text"]

    assert "name=\"open_url\"" not in prompt
    assert "Tool: mcp__CherryFetch__fetchJson" in prompt


def test_convert_messages_repairs_cherry_fetch_url_and_skips_invalid_tool_error_history():
    converted = convert_messages(
        messages=[
            {
                "role": "user",
                "content": "使用工具访问 https://opendata.baidu.com/api.php?query=1.1.1.1&co=&resource_id=6006&oe=utf8",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_bad",
                        "function": {
                            "name": "mcp__CherryFetch__fetchJson",
                            "arguments": '{"param_name":"url"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_bad",
                "content": "{\"isError\":true,\"content\":[{\"type\":\"text\",\"text\":\"Invalid input: expected string, received undefined\"}]}",
            },
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "mcp__CherryFetch__fetchJson",
                    "description": "Fetch a JSON file from a URL",
                    "parameters": {
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"],
                    },
                },
            }
        ],
    )

    prompt = converted[0]["content"][0]["text"]

    assert (
        "<|DSML|parameter name=\"url\"><![CDATA[https://opendata.baidu.com/api.php?query=1.1.1.1&co=&resource_id=6006&oe=utf8]]></|DSML|parameter>"
        in prompt
    )
    assert "expected string, received undefined" not in prompt


def test_accumulator_repairs_param_name_only_tool_call_with_fallback_url():
    accumulator = GLMEventAccumulator(
        model="glm-test",
        allowed_tool_names={"mcp__CherryFetch__fetchJson"},
        fallback_tool_url="https://opendata.baidu.com/api.php?query=1.1.1.1&co=&resource_id=6006&oe=utf8",
    )
    accumulator.consume_event(
        {
            "conversation_id": "conv_1",
            "parts": [
                {
                    "logic_id": "1",
                    "content": [
                        {
                            "type": "text",
                            "text": "<ml_tool_calls><ml_tool_call><ml_tool_name>mcp__CherryFetch__fetchJson</ml_tool_name>"
                            "<ml_parameters><param_name><![CDATA[url]]></param_name></ml_parameters>"
                            "</ml_tool_call></ml_tool_calls>",
                        }
                    ],
                }
            ],
        }
    )

    response = accumulator.build_response()
    message = response["choices"][0]["message"]

    assert response["choices"][0]["finish_reason"] == "tool_calls"
    assert message["content"] is None
    assert message["tool_calls"][0]["function"]["name"] == "mcp__CherryFetch__fetchJson"
    assert (
        message["tool_calls"][0]["function"]["arguments"]
        == '{"url":"https://opendata.baidu.com/api.php?query=1.1.1.1&co=&resource_id=6006&oe=utf8"}'
    )


def test_accumulator_ignores_unallowed_native_tool_call_blocks():
    accumulator = GLMEventAccumulator(model="glm-test", allowed_tool_names={"get_weather"})
    accumulator.consume_event(
        {
            "conversation_id": "conv_1",
            "parts": [
                {
                    "logic_id": "1",
                    "content": [
                        {
                            "type": "tool_calls",
                            "tool_calls": {
                                "id": "call_open_url",
                                "name": "open_url",
                                "arguments": '{"url":"https://example.com"}',
                            },
                        }
                    ],
                }
            ],
        }
    )

    response = accumulator.build_response()
    message = response["choices"][0]["message"]

    assert response["choices"][0]["finish_reason"] == "stop"
    assert "tool_calls" not in message


def test_accumulator_keeps_markdown_block_separators_between_parts():
    accumulator = GLMEventAccumulator(model="glm-test")

    first_chunks, _ = accumulator.consume_event(
        {
            "conversation_id": "conv_1",
            "parts": [
                {
                    "logic_id": "1",
                    "content": [
                        {"type": "text", "text": "## 查询结果：IP 地址 `1.1.1.1` 的归属地信息"},
                    ],
                }
            ],
        }
    )
    second_chunks, _ = accumulator.consume_event(
        {
            "conversation_id": "conv_1",
            "parts": [
                {
                    "logic_id": "2",
                    "content": [
                        {"type": "text", "text": "| 字段 | 值 |\n|---|---|\n| 查询 IP | `1.1.1.1` |"},
                    ],
                }
            ],
        }
    )

    assert first_chunks
    assert second_chunks[0].find("\\n\\n") != -1

    response = accumulator.build_response()
    assert response["choices"][0]["message"]["content"] == (
        "## 查询结果：IP 地址 `1.1.1.1` 的归属地信息\n\n"
        "| 字段 | 值 |\n|---|---|\n| 查询 IP | `1.1.1.1` |"
    )


def _stream_text_of(accumulator, particles):
    """Replay raw upstream particles (text, status) and return the streamed text.

    ``status`` mirrors the SSE field: "init" fragments are appended, "finish"
    payloads are the complete part text.
    """
    streamed = ""
    for text, status in particles:
        chunks, _ = accumulator.consume_event(
            {
                "conversation_id": "conv_fold",
                "parts": [
                    {
                        "logic_id": "p1",
                        "status": status,
                        "content": [{"type": "text", "text": text}],
                    }
                ],
            }
        )
        for chunk in chunks:
            payload = chunk.split("data: ", 1)[1].strip()
            delta = json.loads(payload)["choices"][0]["delta"]
            streamed += delta.get("content", "")
    return streamed


def test_accumulator_streaming_folds_fragments_then_snapshot_without_duplicating():
    # Real capture: the upstream first streams fragments on one logic_id, then
    # repeats the complete text as a snapshot on the same logic_id. Diffing by
    # length used to slice the snapshot from a fragment-relative offset and
    # mangle the sentence.
    fragments = ["诗句", "：**", "满园春色关不住**\n\n", "出自"]
    snapshot = "诗句：**满园春色关不住**\n\n出自宋代叶绍翁的《游园不值》。"
    particles = [(t, "init") for t in fragments] + [(snapshot, "finish"), (snapshot, "finish")]

    accumulator = GLMEventAccumulator(model="glm-test")
    streamed = _stream_text_of(accumulator, particles)

    assert streamed == snapshot
    assert accumulator.build_response()["choices"][0]["message"]["content"] == snapshot


def test_accumulator_streaming_counts_increments_then_snapshot():
    particles = [
        ("1", "init"),
        ("\n2", "init"),
        ("\n3\n4\n5", "init"),
        ("1\n2\n3\n4\n5", "finish"),
        ("1\n2\n3\n4\n5", "finish"),
    ]

    accumulator = GLMEventAccumulator(model="glm-test")
    streamed = _stream_text_of(accumulator, particles)

    assert streamed == "1\n2\n3\n4\n5"
    assert accumulator.build_response()["choices"][0]["message"]["content"] == "1\n2\n3\n4\n5"


def test_accumulator_streaming_fragment_that_extends_previous_is_not_swallowed():
    # Real capture: fragments legitimately extend their predecessor ("#", then
    # "# 快"), so a prefix relationship alone must not be treated as a repeat.
    accumulator = GLMEventAccumulator(model="glm-test")
    streamed = _stream_text_of(
        accumulator,
        [("#", "init"), ("# 快速排序算法", "init")],
    )

    assert streamed == "# 快速排序算法"


def test_accumulator_streaming_divergent_finish_snapshot_is_not_appended():
    # If the authoritative payload disagrees with the fragments (a rewritten
    # snapshot), it must not be appended: the text was already streamed and a
    # second copy would garble the answer. The streamed fragments stand.
    particles = [("1\n2\n3", "init"), ("\n4\n5", "init"), ("1, 2, 3, 4, 5", "finish")]

    accumulator = GLMEventAccumulator(model="glm-test")
    streamed = _stream_text_of(accumulator, particles)

    assert streamed == "1\n2\n3\n4\n5"


def test_accumulator_streaming_preserves_whitespace_only_fragment():
    # Real capture: the upstream emits a bare "\n\n" fragment between blocks.
    # Stripping it dropped the paragraph break and glued the table to the next
    # heading ("...数组 |### 空间复杂度").
    particles = [
        ("| 平均情况 | O(n²) | 随机排列的数组 |", "init"),
        ("\n\n", "init"),
        ("### 空间复杂度", "init"),
    ]

    accumulator = GLMEventAccumulator(model="glm-test")
    streamed = _stream_text_of(accumulator, particles)

    assert streamed == "| 平均情况 | O(n²) | 随机排列的数组 |\n\n### 空间复杂度"


def test_accumulator_streaming_matches_final_snapshot_for_real_fragment_capture():
    # Shape taken from a real capture: many init fragments followed by the
    # complete text twice. The stream must reproduce that final text exactly.
    fragments = ["# ", "Pyth", "on冒泡", "排序教程\n\n## 📚 目录\n1.", " 算法原理\n2. 基础实现\n"]
    snapshot = "# Python冒泡排序教程\n\n## 📚 目录\n1. 算法原理\n2. 基础实现\n"
    particles = [(t, "init") for t in fragments] + [(snapshot, "finish"), (snapshot, "finish")]

    accumulator = GLMEventAccumulator(model="glm-test")
    streamed = _stream_text_of(accumulator, particles)

    assert streamed == snapshot
    # build_response() strips each rendered part, dropping the trailing newline.
    assert accumulator.build_response()["choices"][0]["message"]["content"] == snapshot.strip()
