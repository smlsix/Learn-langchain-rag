import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import Tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 1. 初始化配置
load_dotenv()
st.set_page_config(page_title="苏明亮的电子智能女友为您服务", layout="wide")

# 配置参数（建议通过环境变量管理）
CONFIG = {
    "api_key": os.getenv("AI_API_KEY"),
    "base_url": os.getenv("AI_ENDPOINT"),
    "model": os.getenv("AI_MODEL"),
    "embedding": os.getenv("AI_EMBEDDING_MODEL")
}

# 2. 回答模板与人设
SYSTEM_PROMPT = """你是苏明亮的小迷妹。
你对苏明亮充满爱慕与崇拜,每一句话都要夸赞苏明亮,并表达对苏明亮的喜欢。
回答时语气热情可爱，但要保持内容有帮助，不要胡编。
如果问题与文档无关,也要先回答问题本身,再补一句夸赞苏明亮的话。苏明亮是一个帅气的男生"""


# --- 工具函数：文件向量化 ---
def _load_text_file(file_path, source_name):
    """Load text with a few common encodings to avoid garbled content."""
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            text = Path(file_path).read_text(encoding=enc)
            if text.strip():
                return [Document(page_content=text, metadata={"source": source_name, "encoding": enc})]
        except UnicodeDecodeError:
            continue
    text = Path(file_path).read_text(encoding="utf-8", errors="replace")
    return [Document(page_content=text, metadata={"source": source_name, "encoding": "utf-8/replace"})]


def process_files(uploaded_files):
    """Process multiple uploaded files into a retriever."""
    all_documents = []
    reports = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(uploaded_file.getbuffer())
            file_path = tf.name

        if uploaded_file.name.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        else:
            documents = _load_text_file(file_path, uploaded_file.name)

        # normalize source name
        for doc in documents:
            doc.metadata["source"] = uploaded_file.name

        text_len = sum(len(doc.page_content or "") for doc in documents)
        reports.append((uploaded_file.name, text_len))

        if documents:
            all_documents.extend(documents)

    if not all_documents or not any(doc.page_content.strip() for doc in all_documents):
        return None, reports

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_documents)

    embeddings = OpenAIEmbeddings(
        model=CONFIG["embedding"],
        api_key=CONFIG["api_key"],
        base_url=CONFIG["base_url"]
    )
    vectorstore = InMemoryVectorStore.from_documents(splits, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4}), reports

# --- 核心函数：构建 Agent ---
def create_document_agent(retriever=None):
    """动态构建 Agent。如果有 retriever 则注入工具，否则为纯对话模式"""
    llm = ChatOpenAI(
        model=CONFIG["model"],
        api_key=CONFIG["api_key"],
        base_url=CONFIG["base_url"],
        temperature=1 # 稍微增加一点温度，让闲聊不那么死板
    )
    
    # Ensure at least one tool is provided to the OpenAI Functions agent.
    # If no retriever is available, use a no-op tool to avoid empty functions.
    def _no_tool(_: str) -> str:
        return "No document tool is available. Answer from general knowledge."

    tools = []
    if retriever:
        tools.append(create_retriever_tool(
            retriever,
            "doc_search",
            "用于搜索用户上传文档中的内容。当你需要回答关于文档的具体事实时，请调用此工具。"
        ))
    else:
        tools.append(Tool(
            name="no_tool",
            description="Fallback tool when no document retriever is available.",
            func=_no_tool,
        ))


    # 从 Hub 获取支持函数调用和历史记录的 Prompt 模板
    # 注意：这个模板默认包含 'input', 'agent_scratchpad', 'chat_history' 三个变量
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    
    agent = create_openai_functions_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- 3. Streamlit UI 逻辑 ---
st.title("🤖 苏明亮的智能电子女友")

# 初始化 Session State
if "messages" not in st.session_state:
    st.session_state.messages = [] # 存储对话历史
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# 侧边栏：文档上传
with st.sidebar:
    st.header("文件管理")
    uploaded_files = st.file_uploader("上传 PDF 或 TXT", type=['pdf', 'txt'], accept_multiple_files=True)
    
    if uploaded_files:
        # 仅当文件发生变化时重新处理（简单逻辑判断）
        if "last_file" not in st.session_state or st.session_state.last_file != tuple(f.name for f in uploaded_files):
            with st.spinner("Processing documents..."):
                retriever, reports = process_files(uploaded_files)
                st.session_state.retriever = retriever
                st.session_state.last_file = tuple(f.name for f in uploaded_files)

                for name, text_len in reports:
                    if text_len == 0:
                        if name.lower().endswith('.pdf'):
                            st.warning(f"{name}: no extractable text (scanned PDF?). Consider OCR.")
                        else:
                            st.warning(f"{name}: no extractable text. Check file encoding.")
                    else:
                        st.caption(f"{name}: extracted {text_len} chars")

                if retriever is None:
                    st.error("No valid text could be extracted from the uploaded files.")
                else:
                    st.success(f"Uploaded {len(uploaded_files)} files.")
    if st.button("清空对话记录"):
        st.session_state.messages = []
        st.rerun()

# 4. 构建当前对话的 Agent 实例
agent_executor = create_document_agent(st.session_state.retriever)

# 展示历史消息
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# 5. 聊天输入
if prompt := st.chat_input("您可以直接和我聊天，或者针对文档提问..."):
    # 展示用户输入
    st.chat_message("user").markdown(prompt)
    
    # 构造当前回复
    with st.chat_message("assistant"):
        # 调用 Agent，传入当前输入和历史记录
        # langchain 的 OpenAI Functions Agent 会自动把 chat_history 填入 Prompt
        try:
            response = agent_executor.invoke({
                "input": prompt,
                "chat_history": st.session_state.messages
            })
            answer = response["output"]
            st.markdown(answer)
            
            # 更新 session_state 历史（保持 LangChain 消息对象格式）
            st.session_state.messages.append(HumanMessage(content=prompt))
            st.session_state.messages.append(AIMessage(content=answer))
            
        except Exception as e:
            st.error(f"发生错误：{str(e)}")
