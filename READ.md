Summary

The RAG (Retrieval-Augmented Generation) implementation lives in the notebook multimodal.ipynb.
It implements a multimodal RAG pipeline for PDF documents: extract text + images, embed both text and images into a shared vector space (CLIP), store embeddings in a FAISS index, retrieve relevant items for a query, assemble a multimodal prompt (text excerpts + base64 image data URLs) and call an LLM to answer.
Key components (what the code does)

Data ingestion

Uses PyMuPDF (fitz) to open the PDF and iterate pages.
Extracts page text and images. Text is converted into Document objects with metadata {page, type}.
Images are extracted to PNG bytes, stored as base64 in image_data_store keyed by image_id and represented in the docs as page_content like "[Image:page_1_img_0]".
Text chunking

Uses RecursiveCharacterTextSplitter with chunk_size=500 and chunk_overlap=100 to split long page text into chunks before embedding. This reduces token/context overload and improves retrieval granularity.
Embeddings (multimodal)

Uses HuggingFace transformers CLIP model (openai/clip-vit-base-patch32) to produce:
embed_txt(text): text embeddings from CLIP text encoder then projected and normalized.
embed_img(image): image embeddings from CLIP vision encoder then projected and normalized.
A CLIPEmbeddings class implements the Embeddings interface expected by the vectorstore, wrapping these functions.
Vector store

Builds a FAISS vector index via langchain_community.vectorstores.FAISS.from_embeddings using pairs (doc.page_content, embedding).
Metadatas stored alongside vectors include page/type/image_id so retrieval returns both text and references to images.
Retrieval

retrieve_multimodel(query, k=5): embeds the query with CLIP text embedding and runs vector_store.similarity_search_by_vector(embedding=query_embedding, k=k).
Prompt construction (multimodal)

create_multimodal_message(query, retrieved_docs) builds a HumanMessage whose content is a list:
text entries for the question and text excerpts (prefixed with page numbers)
for image docs, image entries of type "image_url" with data: URL "data:image/png;base64,<base64-data>" pulled from image_data_store
a final instruction: "Please answer the question based on the provided text and images."
This message is then passed to the chat LLM.
LLM

The notebook initializes a chat model via init_chat_model("openai/gpt-4o-mini", model_provider="openai", api_key=os.getenv("OPENROUTER_API_KEY"), base_url="https://openrouter.ai/api/v1") and calls llm.invoke([message]) to produce the answer.
Design decisions and rationale

Single unified embedding space (CLIP) for text + images

Rationale: CLIP maps text and images into a compatible vector space, making it straightforward to retrieve multimodal content for a single query embedding. This allows retrieving images that are semantically relevant to text queries without separate cross-modal matching logic.
Trade-off: CLIP’s text embeddings are optimized for short captions (max_length ~77) and for image–caption alignment; they are not as powerful for long-form semantic text retrieval as specialist text embedding models (OpenAI, Cohere, text-embedding-3, etc.). But using CLIP simplifies the pipeline and avoids coordinating two different vector spaces.
Local FAISS index

Rationale: FAISS is fast for nearest neighbor search and works locally (no external service), good for prototyping and small/medium datasets.
Trade-off: FAISS here appears used in-memory (built from embeddings). For production, persistence, replication, and scaling require explicit index persistence to disk or migrating to a managed vector DB (Pinecone, Milvus, Chroma with persistence).
Chunk size and overlap (500 / 100)

Rationale: balances chunk granularity (smaller chunks increase retrieval precision) with context richness inside each chunk. Overlap reduces information loss at chunk boundaries.
Trade-off: chosen values are heuristic; optimal values depend on document style and the LLM prompt budget.
Prompt design: include top-k text excerpts and inline base64 images

Rationale: giving the LLM both the textual context and the actual images (as data URLs) lets a model that supports multimodal input use both modalities to answer accurately.
Trade-off: embedding images as base64 increases message size dramatically. Many LLM APIs will not accept raw base64 images or may not support image inputs in the same way — this notebook relies on an LLM API or wrapper that accepts that message content format. It also increases bandwidth and memory usage. If the deployed model does not accept multimodal content, images would need to be summarized/processed separately and provided as text descriptions instead.
Using langchain_community (FAISS wrapper)

Rationale: quick integration with language model flow using an available wrapper.
Trade-off: the notebook notes a deprecation warning for langchain_community. Plan to migrate to maintained libraries (langchain-core/vectorstores, chromadb, etc.) to avoid future breakage.
Implementation details worth noting

Embeddings normalization: embeddings are normalized (features / features.norm(...)) before storing; this makes cosine similarity equivalent to dot product similarity in FAISS and stabilizes retrieval.
Metadata: each Document stores metadata (page, type, image_id) used to build the prompt and to show provenance in retrieval logs.
Retrieval method: direct similarity_search_by_vector is used — a pure dense retrieval. No re-ranking or lexical (sparse) retrieval is combined.
Query embedding: uses CLIP’s embed_txt; for longer queries, CLIP truncation to 77 tokens may remove important nuance.
LLM invocation: uses llm.invoke([message]) with a HumanMessage; the notebook wraps prompt content as structured content entries (type:text, type:image_url, etc.) — this relies on the chat-model wrapper understanding that structure.
Limitations and risks

CLIP text embeddings are suboptimal for long, nuance-heavy text queries. They are meant for short caption-style alignment.
The notebook’s FAISS is likely in-memory and not persisted/saved; index rebuilding is needed across runs unless persistence is added.
Using base64 images embedded into messages may not be compatible with all LLM providers / wrappers, and it is expensive in bandwidth and tokens.
The pipeline is synchronous and single-threaded; embedding many documents will be slow without batching and GPU acceleration.
langchain_community is deprecated (not actively maintained) — future compatibility risk.
Suggested improvements and next steps

Use a stronger text embedding model for text (e.g., OpenAI text-embedding-3-family, Mistral/other HF models) while keeping CLIP for images; then implement cross-modal alignment or a two-stage retrieval:

Stage 1: dense text retriever (text embeddings) to get candidate text chunks.
Stage 2: cross-modal match (optional) to surface relevant images, or use a learned multimodal retriever.
Persist the FAISS index and mapping:

Save and load FAISS index and metadata mapping to avoid recomputing on each run.
Or migrate to a vector DB (Chroma, Milvus, Pinecone, Weaviate) for persistence, replication, and scale.
Reranking and provenance

After the initial similarity search, optionally re-rank candidates with a cross-encoder or LLM-based relevance scorer to reduce false positives.
Always return source references and page numbers in the final answer to reduce hallucinations and increase trust.
Reduce message size for images

Option A: send images only when needed (on-demand retrieval) or convert images to compact captions/descriptions (image captioning model) and provide captions to LLM.
Option B: store image URLs hosted on a server and pass short reference URLs instead of base64.
Batching and performace

Batch embedding calls and run on GPU where possible. Precompute embeddings during ingestion and cache them.
Use Faiss IVF/OPQ/IndexIVFFlat for larger datasets.
Migrate away from langchain_community

Replace with current langchain.vectorstores classes or direct FAISS + metadata store; or use Chroma/other managed vector store SDKs.
Practical notes and environment

The notebook loads CLIP from Hugging Face; for faster downloads and higher rate limits set HF_TOKEN.
The chat model uses OPENROUTER_API_KEY as configured; verify the provider supports the multimodal message format used here.
The notebook shows warnings about langchain_community deprecation — consider code updates.
Where to look in the repo

Primary implementation and examples are in multimodal.ipynb. The core cells to inspect:
embedding functions: "embed_img" and "embed_txt"
chunking: RecursiveCharacterTextSplitter usage
index building: the cell that runs FAISS.from_embeddings
retrieval and pipeline: retrieve_multimodel, create_multimodal_message, multimodal_pdf_rag_pipeline