import os
import tempfile
from dotenv import load_dotenv
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import SystemMessage
from langchain import hub

load_dotenv()

# --- 1. 页面配置与人设设定 ---
st.set_page_config(page_title="定制化文档智能体", layout="wide")
st.title("📑 定制化文档智能体")

# 你提到的“固定提示词”和“回答模板”写在这里
SYSTEM_PROMPT = """你是一个专业的文档分析助手。
1. 必须根据提供的文档内容回答问题。如果文档中没有相关信息，请诚实回答“我没在您的文档中找到相关内容”。
2. 回答模板：
   - 核心结论：[用一句话总结回答]
   - 详细细节：[分点列出文档中的证据]
   - 补充建议：[基于文档给出的延伸建议]
3. 如果用户问到“你是谁”，请按照以下固定模板回答：“我是您的专属文档管家，旨在帮您分析上传的资料。”
"""
api_key = os.getenv("AI_API_KEY")
base_url = os.getenv("AI_ENDPOINT")
embedding_model = os.getenv("AI_EMBEDDING_MODEL")
chat_model = os.getenv("AI_MODEL")
# --- 2. 处理文件上传与向量化 ---
@st.cache_resource
def process_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(uploaded_file.getbuffer())
        file_path = tf.name

    # 根据文件类型加载
    if uploaded_file.name.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding='utf-8')
    
    documents = loader.load()
    # 检查是否成功提取了文字（防止扫描版 PDF 没字）
    if not documents or not any(doc.page_content.strip() for doc in documents):
        return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings(
    model=embedding_model,
    api_key=api_key,
    base_url=base_url
    )
    vectorstore = InMemoryVectorStore.from_documents(splits, embeddings)
    return vectorstore.as_retriever()

# --- 3. 侧边栏：文件上传 ---
with st.sidebar:
    st.header("配置区")
    uploaded_file = st.file_uploader("上传您的文档 (PDF 或 TXT)", type=['pdf', 'txt'])
    if uploaded_file:
        st.success(f"文件 {uploaded_file.name} 已选中")

# --- 4. 构建 Agent 逻辑 ---
if uploaded_file:
    # 只要有文件，就生成/更新检索器
    retriever = process_file(uploaded_file)
    if retriever is None:
        st.error("⚠️ 无法从文件中提取文字。这可能是扫描件（图片），请尝试上传文字版 PDF。")
    else:
        # 在提示词里明确告诉它：货已经收到了！
        DYNAMIC_SYSTEM_PROMPT = SYSTEM_PROMPT + f"\n目前用户已经成功上传了文件：{uploaded_file.name}。你可以直接通过 doc_search 工具查询。"
    
    tool = create_retriever_tool(
        retriever,
        "doc_search",
        "搜索用户上传的文档内容。当用户提问关于文档具体事实时使用。"
    )
    tools = [tool]
    
    llm = ChatOpenAI(
    model=chat_model,
    api_key=api_key,
    base_url=base_url,
    temperature=0
    )
    
    # 获取基础 Prompt 并注入我们的 SYSTEM_PROMPT
    base_prompt = hub.pull("hwchase17/openai-functions-agent")
    # 将系统指令插入到 Prompt 的开头
    prompt = base_prompt.partial(system_message=DYNAMIC_SYSTEM_PROMPT)
    
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # --- 5. 聊天界面 ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("请针对文档提问..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            response = agent_executor.invoke({"input": user_input})
            answer = response["output"]
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("请在左侧上传文档后开始对话。")