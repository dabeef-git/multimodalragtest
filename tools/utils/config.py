import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(BASE_DIR, ".env"))

EMBEDDING_MODEL = "gemini-embedding-2-preview"
EMBEDDING_DIMS = 1536
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "multimodal-embeddings")
PINECONE_CLOUD = os.environ.get("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.environ.get("PINECONE_REGION", "us-east-1")
TMP_DIR = os.path.join(BASE_DIR, ".tmp")

os.makedirs(TMP_DIR, exist_ok=True)


def get_gemini_client():
    from google import genai
    return genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


def get_pinecone_client():
    from pinecone import Pinecone
    return Pinecone(api_key=os.environ["PINECONE_API_KEY"])


def get_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
