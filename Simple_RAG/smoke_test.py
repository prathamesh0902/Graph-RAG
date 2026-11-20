from rag_pipeline import VectorStore

vs = VectorStore("./chroma_db_test")
collection = vs.client.get_or_create_collection("podcast_rag")
collection.add(documents=["Hello world"], ids=["doc2"])

print("Stored docs:", collection.get())

vs = VectorStore("./chroma_db_test")
collection = vs.client.get_or_create_collection("podcast_rag")
collection.add(documents=["Hello world"], ids=["doc3"])

print("Stored docs:", collection.get())