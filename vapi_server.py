from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import chromadb
from chromadb.utils import embedding_functions
import json


app = FastAPI()

# Connect to the local vector DB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = chroma_client.get_collection(
    name="knowledge_base",
    embedding_function=sentence_transformer_ef
)

@app.post("/query-kb")
async def query_kb(request: Request):
    body = await request.json()
    print("\n--- [DEBUG] Raw Payload Received from Vapi ---")
    print(json.dumps(body, indent=2))

    # Parse call_id and query_text robustly across Vapi schema variations
    call_id = None
    query_text = ""

    # Schema Style 1: Standard Vapi Tool Call Webhook
    if "message" in body and "toolCalls" in body["message"]:
        tool_call = body["message"]["toolCalls"][0]
        call_id = tool_call.get("id")
        args = tool_call.get("function", {}).get("arguments", {})
        query_text = args.get("query", "") if isinstance(args, dict) else json.loads(args).get("query", "")

    # Schema Style 2: Direct Server Tool Webhook
    elif "toolCall" in body:
        tool_call = body["toolCall"]
        call_id = tool_call.get("id")
        args = tool_call.get("function", {}).get("arguments", {})
        query_text = args.get("query", "") if isinstance(args, dict) else json.loads(args).get("query", "")

    print(f"--- [DEBUG] Extracted Query: '{query_text}' | Call ID: '{call_id}' ---")

    if not query_text:
        return JSONResponse(content={
            "results": [{"toolCallId": call_id, "result": "No query was provided by the agent."}]
        })

    # Retrieve from ChromaDB
    results = collection.query(query_texts=[query_text], n_results=10)
    retrieved_docs = results["documents"][0] if results.get("documents") else []
    
    print(f"--- [DEBUG] Found {len(retrieved_docs)} matching documents ---")

    if retrieved_docs:
        context_str = "\n---\n".join(retrieved_docs)
    else:
        context_str = "No relevant information found in knowledge base."

    # Return response formatted strictly for Vapi
    return JSONResponse(content={
        "results": [
            {
                "toolCallId": call_id,
                "result": context_str
            }
        ]
    })
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)