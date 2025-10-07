
import json
import re
import toga
from toga.style import Pack
from toga.style.pack import COLUMN, ROW

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# Hugging Face + OpenVINO imports
from transformers import AutoTokenizer, pipeline
from optimum.intel.openvino import OVModelForCausalLM
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings

# ========= CONFIG =========
JSON_FILE = "C:/New folder (2)/project_1_publications.json"
MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"
SIMILARITY_THRESHOLD = 0.5 # adjust higher/lower depending on dataset

# ========= STEP 1: Load JSON =========
def load_json(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    docs = []
    for item in raw_data:
        text = ""
        if "title" in item:
            text += f"Title: {item['title']}\n"
        if "content" in item:
            text += item["content"]

        docs.append(
            Document(
                page_content=text,
                metadata={k: v for k, v in item.items() if k not in ["content"]}
            )
        )
    return docs

# ========= STEP 2: Split into Chunks =========
def split_docs(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

# ========= STEP 3: Create Vector Store =========
def build_vectorstore(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# ========= STEP 4: Setup LLM + OpenVINO =========
def build_llm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = OVModelForCausalLM.from_pretrained(
        MODEL_ID,
        export=True,
        compile=True,
        device="GPU"  # Intel UHD GPU
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=200
    )
    return HuggingFacePipeline(pipeline=pipe)

# ========= STEP 5: Casual Chat Filter =========
def is_casual(query: str) -> bool:
    query = query.lower().strip()
    return bool(re.match(r"^(hi|hello|hey|good (morning|afternoon|evening))$", query))

# ========= STEP 6: Clean Output =========
def clean_output(text: str) -> str:
    for key in ["Final Answer:", "Answer:"]:
        if key in text:
            text = text.split(key)[-1]
    return text.strip()

# ========= STEP 7: Strict RAG QA =========
def get_rag_answer(query, vectorstore, llm):
    if is_casual(query):
        return "Hello! How can I help you?"

    # ✅ use relevance scores instead of similarity_search_with_score
    docs_and_scores = vectorstore.similarity_search_with_relevance_scores(query, k=10)

    # ✅ filter by threshold
    filtered_docs = [doc for doc, score in docs_and_scores if score >= SIMILARITY_THRESHOLD]

    if not filtered_docs:
        return "I don't know based on the available information."

    # ✅ Build strict context
    context = "\n".join([d.page_content for d in filtered_docs])

    prompt_template = """Answer the question clearly and naturally in one block of text.
Do NOT include extra phrases like "Helpful Answer", "Answer:", A/B options, or explanations about the format.
Just return the clean answer strictly using the information from the context below. 
If the answer is not in the context, respond with: "I don't know based on the available information."

Context:
{context}

Question: {question}

Final Answer:"""

    prompt = prompt_template.format(context=context, question=query)
    response = llm.invoke(prompt)
    return clean_output(response)

# ========= STEP 8: Toga App =========
class RAGApp(toga.App):
    def startup(self):
        main_box = toga.Box(style=Pack(direction=COLUMN, padding=10))

        self.chat_log = toga.MultilineTextInput(
            readonly=True,
            style=Pack(flex=1, padding=5, height=400)
        )

        input_box = toga.Box(style=Pack(direction=ROW, padding_top=5))
        self.user_input = toga.TextInput(style=Pack(flex=1, padding=5))
        self.user_input.on_confirm = self.send_message  # press Enter works
        send_button = toga.Button(
            "Send",
            on_press=self.send_message,
            style=Pack(width=100, padding=5)
        )
        input_box.add(self.user_input)
        input_box.add(send_button)

        main_box.add(self.chat_log)
        main_box.add(input_box)

        self.main_window = toga.MainWindow(title="RAG Chatbot (OpenVINO)")
        self.main_window.content = main_box
        self.main_window.show()

        # Load dataset and vector store
        self.chat_log.value += "🔄 Loading dataset...\n"
        docs = load_json(JSON_FILE)
        chunks = split_docs(docs)
        self.vectorstore = build_vectorstore(chunks)

        self.chat_log.value += "⚙️ Setting up LLM...\n"
        self.llm = build_llm()
        self.chat_log.value += "✅ Ready! Ask me anything.\n\n"

    def send_message(self, widget):
        query = self.user_input.value.strip()
        if not query:
            return

        self.chat_log.value += f"You: {query}\n"
        response = get_rag_answer(query, self.vectorstore, self.llm)
        self.chat_log.value += f"Assistant: {response}\n\n"
        self.user_input.value = ""

def main():
    return RAGApp("RAG Assistant", "org.beeware.ragbot")

if __name__ == "__main__":
    main().main_loop()
