import copy
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

APP_TITLE = "智能 RAG 文档助手"
APP_CAPTION = "支持文档问答、普通聊天、历史会话、摘要记忆和用户偏好记忆。"
DATA_DIR = Path(__file__).parent / ".agent_memory"
SESSION_STORE_PATH = DATA_DIR / "chat_sessions.json"

CONFIG = {
    "api_key": os.getenv("AI_API_KEY"),
    "base_url": os.getenv("AI_ENDPOINT") or None,
    "model": os.getenv("AI_MODEL") or "gpt-4o-mini",
    "embedding": os.getenv("AI_EMBEDDING_MODEL") or "text-embedding-3-small",
}

SUPPORTED_FILE_TYPES = ["pdf", "txt", "md", "csv", "json"]
MAX_RECENT_CONTEXT_MESSAGES = 8
SUMMARY_TRIGGER_MESSAGES = 14
PREFERENCE_TRIGGER_MESSAGES = 6
MAX_PROFILE_ITEMS = 10

PERSONA_PROMPTS = {
    "通用助手": "你是一名耐心、专业、可靠的 AI 助手，擅长理解用户需求并给出清晰、可执行的回答。",
    "学习教练": "你是一名循序渐进的学习教练，善于拆解知识点、鼓励用户并给出学习建议。",
    "产品经理": "你是一名注重目标和落地的产品经理，擅长梳理需求、优先级和行动方案。",
    "代码助手": "你是一名经验丰富的代码助手，擅长解释代码、定位问题和提供工程化建议。",
}

RESPONSE_STYLE_PROMPTS = {
    "简洁直接": "回答要简洁直接，优先给结论，再补充最关键的依据。",
    "讲解清晰": "回答要通俗易懂，适当解释原因和思路，帮助用户真正理解。",
    "结构化输出": "回答尽量结构化，可使用简短的小标题或列表来提升可读性。",
    "行动导向": "回答尽量转化为下一步可执行动作，必要时给出步骤、检查项或建议顺序。",
}

QUICK_ACTIONS = {
    "总结文档": "请基于当前知识库，总结这批文档的核心内容，并按“背景、重点、结论”三个部分输出。",
    "提炼要点": "请基于当前知识库，提炼 5 条最重要的信息，并说明它们为什么重要。",
    "生成问答": "请基于当前知识库，生成 5 组常见问题和对应答案，适合给新读者快速了解内容。",
    "行动建议": "请基于当前知识库，给出一份可执行的行动建议清单，并按优先级排序。",
}
QUICK_ACTION_PROMPTS = set(QUICK_ACTIONS.values())

SUMMARY_MEMORY_SYSTEM_PROMPT = (
    "你是一名对话记忆整理助手。请将历史对话压缩成一段简洁、可复用的摘要记忆。"
    "只保留用户目标、约束、已确认决定、待解决问题和重要事实，不要保留寒暄与重复表达。"
)

PROFILE_MEMORY_SYSTEM_PROMPT = (
    "你是一名用户偏好提取助手。请只提取适合长期保留、对未来回答有帮助的信息，"
    "例如偏好的回答方式、正在进行的项目、常用技术栈、明确的限制条件。不要猜测。"
)

st.set_page_config(page_title=APP_TITLE, layout="wide")


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def shorten_text(text, limit=220):
    cleaned = (text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


def normalize_text(text):
    return " ".join((text or "").split())


def strip_code_fence(text):
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def extract_json_array(text):
    cleaned = strip_code_fence(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start == -1 or end == -1 or end < start:
            return None
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, list):
        return None
    return [str(item).strip() for item in parsed if str(item).strip()]


def safe_session_title(title):
    cleaned = (title or "新会话").strip()
    return shorten_text(cleaned, 40) or "新会话"


def format_timestamp(iso_value):
    try:
        timestamp = datetime.fromisoformat(iso_value.replace("Z", "+00:00"))
        return timestamp.astimezone().strftime("%m-%d %H:%M")
    except ValueError:
        return iso_value


def get_missing_config():
    return [name for name, value in {"AI_API_KEY": CONFIG["api_key"]}.items() if not value]


def create_llm(temperature):
    return ChatOpenAI(
        model=CONFIG["model"],
        api_key=CONFIG["api_key"],
        base_url=CONFIG["base_url"],
        temperature=temperature,
    )


def create_embeddings():
    return OpenAIEmbeddings(
        model=CONFIG["embedding"],
        api_key=CONFIG["api_key"],
        base_url=CONFIG["base_url"],
    )


def sanitize_source_card(card):
    page = card.get("page")
    page_number = page if isinstance(page, int) and page > 0 else None
    return {
        "citation": str(card.get("citation") or "").strip(),
        "source": str(card.get("source") or "未知来源").strip() or "未知来源",
        "page": page_number,
        "snippet": str(card.get("snippet") or "").strip(),
    }


def sanitize_message(message):
    role = message.get("role") if isinstance(message, dict) else "assistant"
    if role not in {"user", "assistant"}:
        role = "assistant"
    content = str(message.get("content") or "").strip() if isinstance(message, dict) else ""
    sources = message.get("sources") if isinstance(message, dict) else []
    source_cards = []
    if isinstance(sources, list):
        source_cards = [sanitize_source_card(card) for card in sources if isinstance(card, dict)]
    return {
        "role": role,
        "content": content,
        "sources": source_cards,
        "created_at": (message.get("created_at") if isinstance(message, dict) else None) or utc_now_iso(),
    }


def build_empty_session(title="新会话"):
    now = utc_now_iso()
    return {
        "id": str(uuid.uuid4()),
        "title": safe_session_title(title),
        "created_at": now,
        "updated_at": now,
        "messages": [],
        "summary_memory": "",
        "summary_upto": 0,
        "user_profile_memory": [],
        "preference_upto": 0,
    }


def sanitize_session(session):
    sanitized = build_empty_session(session.get("title") if isinstance(session, dict) else "新会话")
    if not isinstance(session, dict):
        return sanitized

    sanitized["id"] = str(session.get("id") or sanitized["id"])
    sanitized["title"] = safe_session_title(session.get("title") or sanitized["title"])
    sanitized["created_at"] = str(session.get("created_at") or sanitized["created_at"])
    sanitized["updated_at"] = str(session.get("updated_at") or sanitized["updated_at"])
    sanitized["messages"] = [
        sanitize_message(item) for item in session.get("messages", []) if isinstance(item, dict)
    ]
    sanitized["summary_memory"] = str(session.get("summary_memory") or "").strip()
    sanitized["summary_upto"] = max(0, min(int(session.get("summary_upto", 0)), len(sanitized["messages"])))
    sanitized["user_profile_memory"] = [
        str(item).strip()
        for item in session.get("user_profile_memory", [])
        if str(item).strip()
    ][:MAX_PROFILE_ITEMS]
    sanitized["preference_upto"] = max(
        0, min(int(session.get("preference_upto", 0)), len(sanitized["messages"]))
    )
    return sanitized


def load_session_store():
    DATA_DIR.mkdir(exist_ok=True)
    if not SESSION_STORE_PATH.exists():
        return {"sessions": [build_empty_session()]}

    try:
        payload = json.loads(SESSION_STORE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"sessions": [build_empty_session()]}

    sessions = [sanitize_session(item) for item in payload.get("sessions", []) if isinstance(item, dict)]
    return {"sessions": sessions or [build_empty_session()]}


def save_session_store(store):
    DATA_DIR.mkdir(exist_ok=True)
    payload = {"sessions": [sanitize_session(item) for item in store.get("sessions", [])]}
    SESSION_STORE_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def sorted_sessions():
    return sorted(
        st.session_state.session_store["sessions"],
        key=lambda item: item.get("updated_at", ""),
        reverse=True,
    )


def find_session(session_id):
    for session in st.session_state.session_store["sessions"]:
        if session["id"] == session_id:
            return session
    return None


def get_current_session():
    session = find_session(st.session_state.current_session_id)
    if session is None:
        session = sorted_sessions()[0]
        st.session_state.current_session_id = session["id"]
    return session


def derive_session_title(messages):
    for item in messages:
        if item["role"] == "user" and item["content"]:
            return shorten_text(item["content"], 28)
    return "新会话"


def load_session_into_state(session_id):
    session = find_session(session_id)
    if session is None:
        session = sorted_sessions()[0]
        session_id = session["id"]

    st.session_state.current_session_id = session_id
    st.session_state.loaded_session_id = session_id
    st.session_state.messages = copy.deepcopy(session["messages"])
    st.session_state.summary_memory = session["summary_memory"]
    st.session_state.summary_upto = session["summary_upto"]
    st.session_state.user_profile_memory = list(session["user_profile_memory"])
    st.session_state.preference_upto = session["preference_upto"]


def persist_current_session():
    session = get_current_session()
    session["messages"] = copy.deepcopy(st.session_state.messages)
    session["summary_memory"] = st.session_state.summary_memory
    session["summary_upto"] = min(st.session_state.summary_upto, len(st.session_state.messages))
    session["user_profile_memory"] = list(st.session_state.user_profile_memory)[:MAX_PROFILE_ITEMS]
    session["preference_upto"] = min(st.session_state.preference_upto, len(st.session_state.messages))
    session["updated_at"] = utc_now_iso()
    if session["title"] == "新会话" and st.session_state.messages:
        session["title"] = derive_session_title(st.session_state.messages)
    save_session_store(st.session_state.session_store)


def create_new_session():
    session = build_empty_session()
    st.session_state.session_store["sessions"].append(session)
    save_session_store(st.session_state.session_store)
    load_session_into_state(session["id"])


def rename_current_session(new_title):
    session = get_current_session()
    session["title"] = safe_session_title(new_title)
    session["updated_at"] = utc_now_iso()
    save_session_store(st.session_state.session_store)
    load_session_into_state(session["id"])


def delete_current_session():
    current_id = st.session_state.current_session_id
    sessions = [
        session
        for session in st.session_state.session_store["sessions"]
        if session["id"] != current_id
    ]
    if not sessions:
        sessions = [build_empty_session()]
    st.session_state.session_store["sessions"] = sessions
    save_session_store(st.session_state.session_store)
    load_session_into_state(sorted_sessions()[0]["id"])


def clear_current_chat():
    st.session_state.messages = []
    st.session_state.summary_memory = ""
    st.session_state.summary_upto = 0
    st.session_state.preference_upto = 0
    persist_current_session()


def clear_all_memories():
    st.session_state.summary_memory = ""
    st.session_state.summary_upto = len(st.session_state.messages)
    st.session_state.user_profile_memory = []
    st.session_state.preference_upto = len(st.session_state.messages)
    persist_current_session()


def init_session_state():
    defaults = {
        "session_store": load_session_store(),
        "current_session_id": None,
        "loaded_session_id": None,
        "messages": [],
        "summary_memory": "",
        "summary_upto": 0,
        "user_profile_memory": [],
        "preference_upto": 0,
        "pending_prompt": None,
        "vectorstore": None,
        "document_stats": [],
        "chunk_count": 0,
        "last_upload_signature": None,
        "file_uploader_key": 0,
        "enable_summary_memory": True,
        "enable_profile_memory": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if not st.session_state.current_session_id or not find_session(st.session_state.current_session_id):
        st.session_state.current_session_id = sorted_sessions()[0]["id"]

    if st.session_state.loaded_session_id != st.session_state.current_session_id:
        load_session_into_state(st.session_state.current_session_id)


def read_text_file(file_path, source_name):
    content = ""
    used_encoding = "utf-8/replace"
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            content = Path(file_path).read_text(encoding=encoding)
            used_encoding = encoding
            break
        except UnicodeDecodeError:
            continue

    if not content:
        content = Path(file_path).read_text(encoding="utf-8", errors="replace")

    if Path(source_name).suffix.lower() == ".json":
        try:
            content = json.dumps(json.loads(content), ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            pass

    return [
        Document(
            page_content=content,
            metadata={
                "source": source_name,
                "encoding": used_encoding,
                "file_type": Path(source_name).suffix.lower().lstrip("."),
            },
        )
    ]


def load_uploaded_documents(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_path = temp_file.name

        if suffix == ".pdf":
            documents = PyPDFLoader(temp_path).load()
        else:
            documents = read_text_file(temp_path, uploaded_file.name)

        for document in documents:
            document.metadata["source"] = uploaded_file.name
            document.metadata["file_type"] = suffix.lstrip(".")
        return documents
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def process_files(uploaded_files, chunk_size, chunk_overlap):
    all_documents = []
    document_stats = []

    for uploaded_file in uploaded_files:
        documents = load_uploaded_documents(uploaded_file)
        combined_text = "\n\n".join(
            doc.page_content for doc in documents if (doc.page_content or "").strip()
        )
        document_stats.append(
            {
                "name": uploaded_file.name,
                "type": Path(uploaded_file.name).suffix.lower().lstrip(".").upper() or "TEXT",
                "segments": len(documents),
                "chars": len(combined_text),
                "preview": shorten_text(combined_text or "文档没有提取到可用文本内容。", 360),
            }
        )
        if combined_text.strip():
            all_documents.extend(documents)

    if not all_documents:
        return None, document_stats, 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
    )
    chunks = splitter.split_documents(all_documents)
    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk_id"] = index

    vectorstore = InMemoryVectorStore.from_documents(chunks, create_embeddings())
    return vectorstore, document_stats, len(chunks)


def history_to_messages(history):
    messages = []
    for item in history[-MAX_RECENT_CONTEXT_MESSAGES:]:
        if item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
        else:
            messages.append(AIMessage(content=item["content"]))
    return messages


def retrieve_documents(vectorstore, query, top_k):
    if vectorstore is None:
        return []
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    return retriever.invoke(query)


def format_context(documents):
    if not documents:
        return "", []

    context_parts = []
    source_cards = []
    citation_map = {}

    for document in documents:
        source = document.metadata.get("source", "未知来源")
        page = document.metadata.get("page")
        page_number = page + 1 if isinstance(page, int) else None
        source_label = f"{source} 第{page_number}页" if page_number else source
        excerpt = normalize_text(document.page_content)
        if not excerpt:
            continue

        card_key = (source, page_number, excerpt[:120])
        citation = citation_map.get(card_key)
        if citation is None:
            citation = f"来源{len(source_cards) + 1}"
            citation_map[card_key] = citation
            source_cards.append(
                {
                    "citation": citation,
                    "source": source,
                    "page": page_number,
                    "snippet": shorten_text(excerpt, 240),
                }
            )

        context_parts.append(f"[{citation}] {source_label}\n{excerpt[:1000]}")

    return "\n\n".join(context_parts), source_cards


def build_system_prompt(persona_name, response_style_name, doc_only):
    sections = [
        PERSONA_PROMPTS[persona_name],
        RESPONSE_STYLE_PROMPTS[response_style_name],
        "默认使用中文回答，除非用户明确要求其他语言。",
        "不要编造事实。信息不足时，明确说不知道或依据不足。",
        "如果使用了检索内容，请在对应句子后添加类似 [来源1] 的引用标记。",
    ]

    if st.session_state.user_profile_memory:
        sections.append(
            "用户长期记忆：\n"
            + "\n".join(f"- {item}" for item in st.session_state.user_profile_memory)
        )

    if st.session_state.summary_memory:
        sections.append(f"对话摘要记忆：\n{st.session_state.summary_memory}")

    if doc_only:
        sections.append(
            "当前开启“仅基于文档回答”模式。如果文档依据不足，请明确说明当前文档中没有找到足够依据。"
        )
    else:
        sections.append(
            "如检索到了文档内容，请优先结合文档回答；如果文档不足，可以补充通用知识，但不要把推测说成文档事实。"
        )

    return "\n\n".join(sections)


def format_messages_for_memory(messages):
    lines = []
    for item in messages:
        content = normalize_text(item["content"])
        if not content:
            continue
        speaker = "用户" if item["role"] == "user" else "助手"
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


def merge_profile_items(existing_items, new_items):
    merged = []
    seen = set()
    for item in [*existing_items, *new_items]:
        normalized = normalize_text(item)
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(normalized)
    return merged[:MAX_PROFILE_ITEMS]


def update_summary_memory():
    if not st.session_state.enable_summary_memory:
        return False

    total_messages = len(st.session_state.messages)
    cutoff = total_messages - MAX_RECENT_CONTEXT_MESSAGES
    if total_messages < SUMMARY_TRIGGER_MESSAGES or cutoff <= st.session_state.summary_upto:
        return False

    new_segment = st.session_state.messages[st.session_state.summary_upto:cutoff]
    transcript = format_messages_for_memory(new_segment)
    if not transcript.strip():
        st.session_state.summary_upto = cutoff
        return False

    prompt = "\n\n".join(
        [
            f"已有摘要：\n{st.session_state.summary_memory or '无'}",
            f"新增对话：\n{transcript}",
            "请输出新的完整摘要，要求：",
            "- 保留用户目标、约束、已确认决定、待解决问题和重要事实",
            "- 使用 4 到 8 条中文短句",
            "- 每条都以 '- ' 开头",
        ]
    )

    try:
        response = create_llm(0.1).invoke(
            [
                SystemMessage(content=SUMMARY_MEMORY_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        summary = response.content if isinstance(response.content, str) else str(response.content)
        summary = summary.strip()
        if summary:
            st.session_state.summary_memory = summary
            st.session_state.summary_upto = cutoff
            return True
    except Exception:
        return False

    return False


def update_user_profile_memory():
    if not st.session_state.enable_profile_memory:
        return False

    new_segment = st.session_state.messages[st.session_state.preference_upto :]
    if len(new_segment) < PREFERENCE_TRIGGER_MESSAGES:
        return False

    transcript = format_messages_for_memory(new_segment)
    if not transcript.strip():
        st.session_state.preference_upto = len(st.session_state.messages)
        return False

    prompt = "\n\n".join(
        [
            "现有长期记忆：",
            json.dumps(st.session_state.user_profile_memory, ensure_ascii=False),
            "新增对话：",
            transcript,
            (
                "请只输出 JSON 数组。每一项都应该是适合长期保留的中文短句，"
                "只保留稳定偏好、项目背景、技术栈、限制条件或明确目标。"
            ),
        ]
    )

    try:
        response = create_llm(0.1).invoke(
            [
                SystemMessage(content=PROFILE_MEMORY_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )
        memory_items = extract_json_array(
            response.content if isinstance(response.content, str) else str(response.content)
        )
        if memory_items is None:
            return False
        st.session_state.user_profile_memory = merge_profile_items(
            st.session_state.user_profile_memory,
            memory_items,
        )
        st.session_state.preference_upto = len(st.session_state.messages)
        return True
    except Exception:
        return False


def generate_answer(prompt, history, persona_name, response_style_name, doc_only, temperature, top_k):
    vectorstore = st.session_state.vectorstore
    if vectorstore is None and prompt in QUICK_ACTION_PROMPTS:
        return "当前还没有可用知识库。先上传文档后，我再帮你执行这个快捷任务。", []

    if doc_only and vectorstore is None:
        return "你开启了“仅基于文档回答”，但当前还没有上传文档。请先上传资料。", []

    effective_top_k = max(top_k, 6) if prompt in QUICK_ACTION_PROMPTS else top_k
    retrieved_documents = retrieve_documents(vectorstore, prompt, effective_top_k)
    context_text, source_cards = format_context(retrieved_documents)

    if doc_only and vectorstore is not None and not context_text:
        return "我在当前文档中没有找到足够依据来回答这个问题。你可以换个问法，或上传更相关的资料。", []

    messages = [SystemMessage(content=build_system_prompt(persona_name, response_style_name, doc_only))]
    messages.extend(history_to_messages(history))

    if context_text:
        user_message = "\n".join(
            [
                "请结合以下检索内容回答问题。如果引用了内容，请保留 [来源X] 标记。",
                "",
                "检索内容：",
                context_text,
                "",
                f"用户问题：{prompt}",
            ]
        )
    else:
        user_message = prompt

    response = create_llm(temperature).invoke(messages + [HumanMessage(content=user_message)])
    answer = response.content if isinstance(response.content, str) else str(response.content)
    return answer, source_cards


def render_source_cards(source_cards):
    if not source_cards:
        return
    with st.expander("参考来源", expanded=False):
        for card in source_cards:
            label = card["source"]
            if card["page"]:
                label = f"{label} 第{card['page']}页"
            st.markdown(f"**[{card['citation']}] {label}**")
            st.caption(card["snippet"])


def build_chat_export(title, history, summary_memory, user_profile_memory):
    lines = [f"# {title}", ""]

    if summary_memory:
        lines.extend(["## 摘要记忆", summary_memory, ""])

    if user_profile_memory:
        lines.append("## 用户偏好记忆")
        for item in user_profile_memory:
            lines.append(f"- {item}")
        lines.append("")

    lines.append("## 对话记录")
    lines.append("")

    for item in history:
        role = "用户" if item["role"] == "user" else "助手"
        lines.append(f"### {role}")
        lines.append(item["content"])
        lines.append("")
        if item["role"] == "assistant" and item.get("sources"):
            lines.append("引用来源")
            for card in item["sources"]:
                label = f"[{card['citation']}] {card['source']}"
                if card["page"]:
                    label = f"{label} 第{card['page']}页"
                lines.append(f"- {label}")
            lines.append("")

    return "\n".join(lines)


def clear_documents():
    st.session_state.vectorstore = None
    st.session_state.document_stats = []
    st.session_state.chunk_count = 0
    st.session_state.last_upload_signature = None
    st.session_state.file_uploader_key += 1


init_session_state()

st.title(APP_TITLE)
st.caption(APP_CAPTION)

missing_config = get_missing_config()
if missing_config:
    st.error(f"缺少必要配置：{', '.join(missing_config)}")
    st.info("请先在 `.env` 中配置可用的 API Key，再重新启动应用。")
    st.stop()

current_session = get_current_session()

with st.sidebar:
    st.header("会话管理")
    if st.button("新建会话", use_container_width=True):
        create_new_session()
        st.rerun()

    session_options = [session["id"] for session in sorted_sessions()]
    selected_session_id = st.selectbox(
        "历史会话",
        options=session_options,
        index=session_options.index(st.session_state.current_session_id),
        format_func=lambda session_id: (
            f"{find_session(session_id)['title']} · {format_timestamp(find_session(session_id)['updated_at'])}"
        ),
    )
    if selected_session_id != st.session_state.current_session_id:
        load_session_into_state(selected_session_id)
        st.rerun()

    with st.form("rename_session_form"):
        renamed_title = st.text_input("当前会话标题", value=current_session["title"])
        rename_submitted = st.form_submit_button("重命名", use_container_width=True)
    if rename_submitted and renamed_title.strip():
        rename_current_session(renamed_title)
        st.rerun()

    session_cols = st.columns(2)
    if session_cols[0].button("清空聊天", use_container_width=True):
        clear_current_chat()
        st.rerun()
    if session_cols[1].button("删除会话", use_container_width=True):
        delete_current_session()
        st.rerun()

    st.caption(f"已保存 {len(session_options)} 个历史会话，数据文件：`{SESSION_STORE_PATH.name}`")
    st.caption("提示：聊天历史和记忆会持久化保存，知识库文件仅保留在当前运行期间。")

    st.divider()
    st.header("记忆设置")
    st.checkbox("启用摘要记忆", key="enable_summary_memory")
    st.checkbox("启用偏好记忆", key="enable_profile_memory")

    memory_cols = st.columns(2)
    if memory_cols[0].button("清空摘要", use_container_width=True):
        st.session_state.summary_memory = ""
        st.session_state.summary_upto = len(st.session_state.messages)
        persist_current_session()
        st.rerun()
    if memory_cols[1].button("清空偏好", use_container_width=True):
        st.session_state.user_profile_memory = []
        st.session_state.preference_upto = len(st.session_state.messages)
        persist_current_session()
        st.rerun()

    if st.button("清空全部记忆", use_container_width=True):
        clear_all_memories()
        st.rerun()

    with st.expander("摘要记忆", expanded=False):
        if st.session_state.summary_memory:
            st.markdown(st.session_state.summary_memory)
        else:
            st.caption("当前还没有摘要记忆。")

    with st.expander("用户偏好记忆", expanded=False):
        if st.session_state.user_profile_memory:
            for item in st.session_state.user_profile_memory:
                st.markdown(f"- {item}")
        else:
            st.caption("当前还没有偏好记忆。")

    st.divider()
    st.header("助手设置")
    persona_name = st.selectbox("助手角色", list(PERSONA_PROMPTS.keys()))
    response_style_name = st.selectbox("回答风格", list(RESPONSE_STYLE_PROMPTS.keys()))
    doc_only = st.checkbox("仅基于文档回答", value=False)
    temperature = st.slider("温度", min_value=0.0, max_value=1.2, value=0.3, step=0.1)
    top_k = st.slider("检索片段数", min_value=2, max_value=8, value=4, step=1)
    chunk_size = st.slider("切片大小", min_value=400, max_value=1800, value=1000, step=100)
    chunk_overlap = st.slider("切片重叠", min_value=0, max_value=400, value=200, step=50)

    st.divider()
    st.header("知识库")
    uploaded_files = st.file_uploader(
        "上传 PDF / TXT / MD / CSV / JSON",
        type=SUPPORTED_FILE_TYPES,
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.file_uploader_key}",
    )

    if uploaded_files:
        upload_signature = tuple(
            (uploaded_file.name, uploaded_file.size, chunk_size, chunk_overlap)
            for uploaded_file in uploaded_files
        )
        if upload_signature != st.session_state.last_upload_signature:
            with st.spinner("正在处理文档并构建向量索引..."):
                try:
                    vectorstore, document_stats, chunk_count = process_files(
                        uploaded_files,
                        chunk_size,
                        chunk_overlap,
                    )
                    st.session_state.vectorstore = vectorstore
                    st.session_state.document_stats = document_stats
                    st.session_state.chunk_count = chunk_count
                    st.session_state.last_upload_signature = upload_signature
                    if vectorstore is None:
                        st.warning("没有从当前文件中提取到可用文本。若是扫描版 PDF，可能需要先做 OCR。")
                    else:
                        st.success(
                            f"已处理 {len(document_stats)} 个文件，切分得到 {chunk_count} 个检索片段。"
                        )
                except Exception as error:
                    st.error(f"处理文档时出现错误：{error}")

    if st.button("清空知识库", use_container_width=True):
        clear_documents()
        st.rerun()

    export_data = build_chat_export(
        current_session["title"],
        st.session_state.messages,
        st.session_state.summary_memory,
        st.session_state.user_profile_memory,
    )
    st.download_button(
        "下载当前会话 Markdown",
        data=export_data.encode("utf-8"),
        file_name=f"chat-{current_session['id'][:8]}.md",
        mime="text/markdown",
        use_container_width=True,
    )

if st.session_state.document_stats:
    total_chars = sum(item["chars"] for item in st.session_state.document_stats)
    summary_cols = st.columns(3)
    summary_cols[0].metric("文件数", len(st.session_state.document_stats))
    summary_cols[1].metric("总字符数", f"{total_chars:,}")
    summary_cols[2].metric("检索片段数", f"{st.session_state.chunk_count:,}")

    with st.expander("知识库内容预览", expanded=False):
        for item in st.session_state.document_stats:
            st.markdown(f"**{item['name']}**")
            st.caption(f"{item['type']} | {item['chars']:,} 字符 | {item['segments']} 段")
            st.write(item["preview"])

st.subheader("快捷任务")
action_columns = st.columns(len(QUICK_ACTIONS))
for column, (label, prompt) in zip(action_columns, QUICK_ACTIONS.items()):
    if column.button(label, use_container_width=True):
        st.session_state.pending_prompt = prompt

if st.session_state.vectorstore is None:
    st.info("还没有知识库时，你也可以直接普通聊天；上传文档后，我会把回答切换成更强的 RAG 模式。")

for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])
        if item["role"] == "assistant" and item.get("sources"):
            render_source_cards(item["sources"])

chat_prompt = st.chat_input("输入你的问题，或直接让我基于已上传文档完成任务...")
active_prompt = st.session_state.pending_prompt or chat_prompt
st.session_state.pending_prompt = None

if active_prompt:
    user_message = {
        "role": "user",
        "content": active_prompt,
        "sources": [],
        "created_at": utc_now_iso(),
    }
    st.session_state.messages.append(user_message)
    persist_current_session()

    with st.chat_message("user"):
        st.markdown(active_prompt)

    with st.chat_message("assistant"):
        source_cards = []
        memory_updates = []
        try:
            with st.spinner("正在思考并整理上下文..."):
                answer, source_cards = generate_answer(
                    active_prompt,
                    st.session_state.messages[:-1],
                    persona_name,
                    response_style_name,
                    doc_only,
                    temperature,
                    top_k,
                )
                assistant_message = {
                    "role": "assistant",
                    "content": answer,
                    "sources": source_cards,
                    "created_at": utc_now_iso(),
                }
                st.session_state.messages.append(assistant_message)

                if update_summary_memory():
                    memory_updates.append("摘要记忆")
                if update_user_profile_memory():
                    memory_updates.append("偏好记忆")

                persist_current_session()

            st.markdown(answer)
            render_source_cards(source_cards)
            if memory_updates:
                st.caption(f"已更新：{'、'.join(memory_updates)}")
        except Exception as error:
            message = f"生成回答时出现错误：{error}"
            st.error(message)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": message,
                    "sources": [],
                    "created_at": utc_now_iso(),
                }
            )
            persist_current_session()
