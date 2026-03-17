# Learn-langchain-rag

一个基于 Streamlit + LangChain + OpenAI 的多功能 RAG 文档助手示例。

当前版本已经支持：

- 上传 `PDF`、`TXT`、`MD`、`CSV`、`JSON`
- 文档问答与普通聊天双模式
- 快捷任务：总结文档、提炼要点、生成问答、行动建议
- 检索参数调节：`top_k`、切片大小、切片重叠、温度
- 回答时展示引用来源片段
- 导出聊天记录为 Markdown
- 多种助手角色和输出风格切换

## 环境变量

在 `.env` 中配置：

```env
AI_API_KEY=your_api_key
AI_ENDPOINT=your_base_url
AI_MODEL=gpt-4o-mini
AI_EMBEDDING_MODEL=text-embedding-3-small
```

`AI_ENDPOINT` 可以按你的服务商情况填写；如果使用默认 OpenAI 接口，可留空或删除。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行项目

```bash
streamlit run my_agent.py
```
