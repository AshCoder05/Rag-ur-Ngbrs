import os
import webbrowser
from google import genai
from langchain_community.document_loaders import PyPDFLoader
from chromadb import Client, Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter

GEMINI_API_KEY = ""
PDF_PATH = os.path.abspath("RWA Welfare Society Guidelines - Whitefield Apartment.pdf")

chromadbClient = Client()
client = genai.Client(api_key=GEMINI_API_KEY)

collection = chromadbClient.get_or_create_collection("pdf_chunks")

# Load PDF with page info preserved
document = PyPDFLoader(PDF_PATH).load()

# Split each page individually to track page numbers
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

text_chunks = []
chunk_metadata = []  # Store page number for each chunk

for page in document:
    page_num = page.metadata.get("page", 0)  # 0-indexed
    page_chunks = text_splitter.split_text(page.page_content)
    for chunk in page_chunks:
        text_chunks.append(chunk)
        chunk_metadata.append({"page": page_num + 1})  # Convert to 1-indexed

embedding = client.models.embed_content(
    model="gemini-embedding-001",
    contents=text_chunks
)

collection.add(
    documents=text_chunks,
    embeddings=[e.values for e in embedding.embeddings],
    ids=[str(i) for i in range(len(text_chunks))],
    metadatas=chunk_metadata
)

print("System ready! Ask questions about your PDF.")

while True:
    query = input("\nQuery (type 'exit' to quit): ")
    if query.lower() == "exit":
        break

    # 1. Embed the query
    query_embedding = client.models.embed_content(
        model="gemini-embedding-001",
        contents=[query]
    )

    # 2. Retrieve relevant chunks
    results = collection.query(
        query_embeddings=[e.values for e in query_embedding.embeddings],
        n_results=3
    )
    
    # Build context with source references
    docs = results['documents'][0]
    metadatas = results['metadatas'][0]
    
    context_parts = []
    sources = []
    for i, (doc, meta) in enumerate(zip(docs, metadatas), 1):
        context_parts.append(f"[{i}] {doc}")
        sources.append({"ref": i, "page": meta["page"], "text": doc[:100] + "..."})
    
    retrieved_context = "\n\n".join(context_parts)

    # 3. Generate Answer
    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the following context.
Use the source references like [1], [2], [3] when citing information from the context.
    
Context:
{retrieved_context}
    
Question: 
{query}
"""

    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=prompt
    )

    print(f"\nAnswer:\n{response.text}")
    
    # Print sources
    print("\n--- Sources ---")
    for src in sources:
        print(f"[{src['ref']}] Page {src['page']}: {src['text']}")
    
    # Ask if user wants to open a source
    open_src = input("\nOpen source in PDF? Enter number (or press Enter to skip): ").strip()
    if open_src.isdigit():
        src_idx = int(open_src) - 1
        if 0 <= src_idx < len(sources):
            page = sources[src_idx]["page"]
            # Open PDF at specific page (works with most PDF readers)
            pdf_url = f"file:///{PDF_PATH}#page={page}"
            webbrowser.open(pdf_url)
            print(f"Opening page {page}...")
