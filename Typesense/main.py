from langchain_community.document_loaders import JSONLoader


load = JSONLoader(
    'data.json',
    jq_schema=".",
    text_content=False
)
print(load.load())