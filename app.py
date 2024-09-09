import copy
import difflib
import io
import json
import random
import string
import logging
import subprocess
import urllib
from datetime import timedelta

import minio
import uvicorn
from fastapi import FastAPI, Query, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, status
from langchain.text_splitter import SentenceTransformersTokenTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain_community.document_loaders.directory import DirectoryLoader, TextLoader
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
import os
from typing import Dict, List
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional
from langchain_community.vectorstores import Milvus
from transformers import AutoTokenizer
from huggingface_hub_batched import HuggingFaceHubEmbeddingsBatched
import pybase64
from minio import Minio
from minio.error import S3Error
from trt_llm import TensorRTLLM
from pymilvus import utility
from retry import retry
from pytube import YouTube
import requests
from pymilvus import db, Collection, connections, FieldSchema, CollectionSchema, DataType, utility, MilvusClient, \
    MilvusException

import nltk
from nltk import sent_tokenize, PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import cpuinfo
import requests
import numpy as np

# import IPython.display as ipd
import riva.client
from riva.client import ASRService, Auth, RecognitionConfig, StreamingRecognitionConfig

import wave
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
import grpc
import asyncio
from scipy.io import wavfile

description = """
RAG service that can be used for creating RAG based application. It works with:
- Nvidia Triton server vis LangChain client
- Milvus vector database
- Minio datastore
- embeddings hosted on text-embeddings-inference image from HuggingFace

## add_document
Let's you chunk the document and add it to the Milvus database and to the whole document to the Minio. 
The document is provided as bytes and is processed using LangChain reader. New collection and bucket will be created and named as "collection" input parameter.
We can additionally specify the chunk_size and overlap.
"""

app = FastAPI(
    title="RAG servie",
    description=description,
    summary="RAG service for building RAG based applications",
    version="0.0.1",
    terms_of_service="All Rights Reserved",
)

default_temperature = 0.5
DEPLOYED_MODEL_NAME_MISTRAL = os.environ.get("DEPLOYED_MODEL_NAME_MISTRAL")
TRITON_URL = os.environ.get('TRITON_URL')
TRITON_MODEL_NAME = os.environ.get('TRITON_MODEL_NAME')
URI_MILVUS = os.environ.get("URI_MILVUS")
SERVER_N = os.environ.get("SERVER_N")
PATH_TLS = os.environ.get("PATH_TLS")
# EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL")
EMBEDDINGS_ADDRESS = os.environ.get("EMBEDDINGS_ADDRESS")
RIVA_URL = os.environ.get("RIVA_URL")
MILVUS_USER = os.environ.get("MILVUS_USER")
MILVUS_PASSWORD = os.environ.get("MILVUS_PASSWORD")
MILVUS_DB = os.environ.get("MILVUS_DB")

# RIVA_URL = "riva-service-riva-api.riva.svc.cluster.local:50051"
# TRITON_MODEL_NAME = "ensemble"
# TRITON_URL = "demo-ml-llm-mistral.llm.svc.cluster.local:38011"
# DEPLOYED_MODEL_NAME_MISTRAL = "ltaler01/Mistral-7B-Instruct-v0.2"

print("RIVA URL 2", RIVA_URL)

AUTH_RIVA = riva.client.Auth(uri=RIVA_URL)

print("RIVA Client", AUTH_RIVA)

lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

llm = TensorRTLLM(server_url=TRITON_URL, model_name=TRITON_MODEL_NAME, tokens=500,
                  temperature=default_temperature)

milvus_client = connections.connect(
    uri=URI_MILVUS,
    secure=True,
    server_pem_path=PATH_TLS,
    server_name=SERVER_N,
    user=MILVUS_USER,
    password=MILVUS_PASSWORD,
    db_name=MILVUS_DB,
    timeout=60.0  # Change its necessary
)

embeddings = HuggingFaceHubEmbeddingsBatched(model=EMBEDDINGS_ADDRESS,
                                             batch_size=32,
                                             model_kwargs={"truncate": True})

embeddings_tokenizer = AutoTokenizer.from_pretrained(DEPLOYED_MODEL_NAME_MISTRAL)

try:
    tokenizer = AutoTokenizer.from_pretrained(DEPLOYED_MODEL_NAME_MISTRAL)
    print("tokenizer", tokenizer)
except Exception as e:
    logging.error(f"Failed to load tokenizer for model {DEPLOYED_MODEL_NAME_MISTRAL}: {str(e)}")
    raise


class CalculateEmbeddingsInput(BaseModel):
    texts: List[str] = []


class RAGInput(BaseModel):
    messages: str
    rag_template: str | None = None
    collection: str
    temperature: float | None = None
    repetition_penalty: float | None = None
    top_p: float | None = None
    top_k: float | None = None
    add_context: bool = False
    documents: List[str] | None = None


class QueryInput(BaseModel):
    messages: str
    temperature: float | None = None
    repetition_penalty: float | None = None
    top_p: float | None = None
    top_k: float | None = None
    stream: bool = True
    language: str


class ContextMatchingInput(BaseModel):
    response: str
    context_data: str


class AddDocumentInput(BaseModel):
    document: str
    document_name: str
    collection: str
    chunk_size: int = 512
    overlap: int = 50


class DropCollectionInput(BaseModel):
    collection: str


class ListDocumentsInput(BaseModel):
    collection: str


class DocumentLinkInput(BaseModel):
    collection: str
    document_name: str


class CreateCollectionInput(BaseModel):
    collection: str


class CollectionInput(BaseModel):
    collection_name: str
    dimension: int
    metric_type: str


class QueryJ(BaseModel):
    question: str
    max_tokens: int = 20


# async def query_llm(messages, temperature, top_k, top_p, repetition_penalty, language, additional_json=None):
#     try:
#         if language == "en":
#             if messages[-1]['role'] == 'user':
#                 messages.append({"role": "assistant", "content": "How can I help you?"})
#             prompt = tokenizer.apply_chat_template(messages, tokenize=False)
#         elif language == "es":
#             if messages[-1]['role'] == 'user':
#                 messages.append({"role": "assistant", "content": "En que puedo ayudarte?"})
#             prompt = f"[Spanish] {tokenizer.apply_chat_template(messages, tokenize=False)}"
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing input: {str(e)}")
#
#     request_id = str(random.randint(1, 9999999))
#
#     invocation_params = llm._get_model_default_parameters
#     invocation_params['temperature'] = temperature or invocation_params['temperature']
#     invocation_params['top_k'] = top_k or invocation_params['top_k']
#     invocation_params['top_p'] = top_p or invocation_params['top_p']
#     invocation_params['repetition_penalty'] = repetition_penalty or invocation_params['repetition_penalty']
#     invocation_params["prompt"] = [[prompt]]
#     model_params = llm._identifying_params
#
#     try:
#         result_queue = llm.client.request_streaming(
#             model_params["model_name"], request_id, **invocation_params
#         )
#
#     except Exception as e:
#         logging.error(f"Error during request: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
#
#     if additional_json is not None:
#         yield json.dumps(additional_json)
#     for token in result_queue:
#         yield token
#

async def query_llm(messages, temperature, top_k, top_p, repetition_penalty, language, additional_json=None):
    try:
        if language == "en":
            if messages[-1]['role'] == 'user':
                messages.append({"role": "assistant", "content": "How can I help you?"})
            prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        elif language == "es":
            if messages[-1]['role'] == 'user':
                # messages.append({"role": "assistant", "content": "En que puedo ayudarte?"})

                prompt = tokenizer.apply_chat_template(messages, tokenize=False)
                request_id = str(random.randint(1, 9999999))
                invocation_params = llm._get_model_default_parameters
                invocation_params['temperature'] = temperature or invocation_params['temperature']
                invocation_params['top_k'] = top_k or invocation_params['top_k']
                invocation_params['top_p'] = top_p or invocation_params['top_p']
                invocation_params['repetition_penalty'] = repetition_penalty or invocation_params['repetition_penalty']
                invocation_params["prompt"] = [[prompt]]
                model_params = llm._identifying_params

                try:
                    result_queue = llm.client.request_streaming(
                        model_params["model_name"], request_id, **invocation_params
                    )

                except Exception as e:
                    logging.error(f"Error during request: {e}")
                    raise HTTPException(status_code=500, detail=str(e))

                if additional_json is not None:
                    yield json.dumps(additional_json)
                for token in result_queue:
                    yield token
        #  prompt = f"[Spanish] {tokenizer.apply_chat_template(messages, tokenize=False)}"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing input: {str(e)}")


rag_template_global = """Eres un asistente muy útil que responde preguntas con sinceridad. Utilice el siguiente contexto para responder la pregunta del usuario. Si no sabe la respuesta, simplemente diga que no la sabe; No intentes inventar una respuesta. Utilice sólo la información del contexto.

Context:
{context}

Question:
{question}

Sólo devuelve la respuesta útil y nada más. Prepare la respuesta según el contexto. Mantenlo lo más cerca posible del contexto."""


def add_context(vector_db, messages, documents, rag_template):
    search_kwargs = {'k': 6}
    rag_prompt = PromptTemplate.from_template(rag_template or rag_template_global)

    if documents is not None:
        search_kwargs['expr'] = f"source in {documents}"
    retriever = vector_db.as_retriever(search_type='similarity', search_kwargs=search_kwargs)
    last_question = messages[-1]['content']
    docs = retriever.get_relevant_documents(last_question)
    context_documents = {}
    for doc in docs:
        name = doc.metadata['name']
        context_documents[name] = context_documents.get(name, {})
        context_documents[name]['url'] = doc.metadata['source']
        context_documents[name]['type'] = doc.metadata['type']
        context_documents[name] = context_documents.get(name, {})
        context_documents[name]['chunks'] = context_documents[name].get('chunks', [])
        context_documents[name]['chunks'].append(doc.page_content)
    messages[-1]['content'] = rag_prompt.format(question=last_question,
                                                context="\n".join([d.page_content for d in docs]))
    return messages, context_documents


@app.post("/calculate_embeddings")
async def calculate_embeddings(calculate_embeddings_input: CalculateEmbeddingsInput):
    _embeddings = embeddings.embed_documents(calculate_embeddings_input.texts)
    return JSONResponse(content={"embeddings": _embeddings}, status_code=200)


@app.post("/rag/")
async def rag_endpoint(rag_input: RAGInput):
    """
     Endpoint that performs RAG operation. It takes the messages (the last one shoule be the message from the user), collection name and documents that will be used when looking for a context.
     It returns the response from the bot for a given query using contextual information. When the add_context parameter is set it will also return the context. The response is streamed, when context is set then the first value in the stream is the contexxt.
    """
    vector_db = Milvus(embedding_function=embeddings,
                       collection_name=rag_input.collection,
                       connection_args={"url": URI_MILVUS}
                       )
    messages = json.loads(rag_input.messages)
    messages, context_documents = add_context(vector_db, messages, rag_input.documents, rag_input.rag_template)
    if not rag_input.add_context:
        context_documents = None
    return StreamingResponse(query_llm(messages,
                                       rag_input.temperature,
                                       rag_input.top_k,
                                       rag_input.top_p,
                                       rag_input.repetition_penalty,
                                       context_documents), media_type="text/plain")


@app.post("/query-LLM-Mistral/")
async def query_endpoint(query_input: QueryInput):
    """
     Endpoint that queries LLM without using context.
    """
    if not query_input.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    try:
        messages = json.loads(query_input.messages)

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")

    try:
        if query_input.stream:
            response = StreamingResponse(query_llm(messages,
                                                   query_input.temperature,
                                                   query_input.top_k,
                                                   query_input.top_p,
                                                   query_input.repetition_penalty,
                                                   query_input.language
                                                   ),
                                         media_type="text/plain")
            return response
        else:
            text = ""
            async for t in query_llm(messages,
                                     query_input.temperature,
                                     query_input.top_k,
                                     query_input.top_p,
                                     query_input.repetition_penalty,
                                     query_input.language
                                     ):
                text += t
            return JSONResponse(content={'text': text}, status_code=200)
    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)


def split_string(input_string):
    pattern = r'(\d+|\w+|\W)'
    result = re.findall(pattern, input_string)
    return result


class MappingData:
    def __init__(self):
        self.full_array = []
        self.clean_array = []
        self.mapping = {}


def convert_text(text):
    mapping_data = MappingData()
    mapping_data.full_array = split_string(text)

    for i, token in enumerate(mapping_data.full_array):
        if token.isalnum() and token.lower() not in stop_words:
            mapping_data.mapping[len(mapping_data.clean_array)] = i
            clean_token = lemmatizer.lemmatize(token.lower())
            mapping_data.clean_array.append(clean_token)
    return mapping_data


def common_substrings(a, b, min_length=3, tolerance=1):
    seqs_a = []
    seqs_b = []
    seqmatcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for (tag, a0, a1, b0, b1) in seqmatcher.get_opcodes():
        if tag == 'equal':
            if seqs_a and a0 - seqs_a[-1][1] <= tolerance and b0 - seqs_b[-1][1] <= tolerance:
                seqs_a[-1][1] = a1
                seqs_b[-1][1] = b1
            else:
                seqs_a.append([a0, a1])
                seqs_b.append([b0, b1])

    seqs_a_filtered = []
    seqs_b_filtered = []
    for seq_a, seq_b in zip(seqs_a, seqs_b):
        if seq_a[1] - seq_a[0] >= min_length and seq_b[1] - seq_b[0] >= min_length:
            seqs_a_filtered.append(seq_a)
            seqs_b_filtered.append(seq_b)

    return seqs_a_filtered, seqs_b_filtered


def match_sentence_in_chunk(sentence_data, full_array, full_array_index, chunk, url, name):
    chunk_data = convert_text(chunk)
    seqs_a, seqs_b = common_substrings(sentence_data.clean_array, chunk_data.clean_array)

    results = []
    for response_indices, chunk_indices in zip(seqs_a, seqs_b):
        chunk_begin = chunk_data.mapping[chunk_indices[0]]
        chunk_end = chunk_data.mapping[chunk_indices[1] - 1] + 1

        response_begin = sentence_data.mapping[response_indices[0]]
        response_end = sentence_data.mapping[response_indices[1] - 1] + 1

        response_begin -= 1
        chunk_begin -= 1
        while response_begin >= 0 and chunk_begin >= 0 and \
                sentence_data.full_array[response_begin].lower() == chunk_data.full_array[chunk_begin].lower():
            response_begin -= 1
            chunk_begin -= 1
        response_begin += 1
        chunk_begin += 1

        response_end += 1
        chunk_end += 1
        while response_end + full_array_index < len(full_array) and \
                chunk_end < len(chunk_data.full_array):
            response_char = full_array[response_end + full_array_index].lower()
            if not response_char.isalnum():
                response_end += 1
                continue
            context_char = chunk_data.full_array[chunk_end].lower()
            if not context_char.isalnum():
                chunk_end += 1
                continue
            if response_char != context_char:
                break
            response_end += 1
            chunk_end += 1
        response_end -= 1
        chunk_end -= 1

        results.append({
            "response_begin": response_begin,
            "response_end": response_end,
            "chunk_begin": chunk_begin,
            "chunk_end": chunk_end,
            'url': url,
            'name': name,
            'chunk': chunk,
            "response_snipped": "".join(full_array[response_begin + full_array_index:response_end + full_array_index]),
            "chunk_snipped": "".join(chunk_data.full_array[chunk_begin:chunk_end]),
        })

    return results


def check_overlap(start1, end1, start2, end2):
    return max(start1, start2) <= min(end1, end2)


def resolve_matching_colisions(sentence_data, results):
    if not sentence_data:
        return results
    if not results:
        return sentence_data

    for data in sentence_data:
        for result in results:
            if check_overlap(data['response_begin'], data['response_end'],
                             result['response_begin'], result['response_end']):
                if len(result['response_snipped']) > len(data['response_snipped']):
                    data.update(result)

    return sentence_data


def tokenize_with_whitespace(input_text):
    current_text = ""
    for sentence in sent_tokenize(input_text):
        additional = ''
        while (input_text[len(current_text)] != sentence[0]):
            additional += input_text[len(current_text)]
            current_text += input_text[len(current_text)]

        current_text += sentence
        sentence = additional + sentence
        yield sentence


@app.post("/context_matching/")
async def context_matching_endpoint(context_matching_input: ContextMatchingInput):
    """
     Endpoint that matches response from LLM to context used
    """
    if context_matching_input.response.strip() == '' or context_matching_input.context_data.strip() == '':
        return JSONResponse(content={"output_data": []}, status_code=200)

    context_data = json.loads(context_matching_input.context_data)

    full_array = []
    sentences_data = []
    for sentence in tokenize_with_whitespace(context_matching_input.response):
        sentences_data.append(convert_text(sentence))
        full_array.extend(sentences_data[-1].full_array)

    output_data = []
    full_array_index = 0
    for sentence_data in sentences_data:
        sentence_results = []
        for name, data in context_data.items():
            for chunk in data['chunks']:
                results = match_sentence_in_chunk(sentence_data, full_array, full_array_index, chunk, data['url'], name)
                sentence_results = resolve_matching_colisions(sentence_results, results)
        if sentence_results:
            output_data.append(sentence_results)
        full_array_index += len(sentence_data.full_array)

    output_data = [item for sublist in output_data for item in sublist]
    return JSONResponse(content=output_data, status_code=200)


@app.post("/add_document/")
async def add_document_endpoint(add_document_input: AddDocumentInput):
    """
     Endpoint for adding a document to the database
    """

    loader = PyPDFLoader("Ejemplos.pdf")

    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=add_document_input.chunk_size,
                                                    chunk_overlap=add_document_input.overlap,
                                               length_function=lambda t: len(embeddings_tokenizer.tokenize(t)))
   #  docs = text_splitter.split_documents(docs)

    milvus_client_out = Milvus(uri=URI_MILVUS, timeout=60.0)

    response = milvus_client_out.from_documents(
        pages,
        embeddings,
        collection_name=add_document_input.collection,
        connection_args={"uri": URI_MILVUS},
        timeout=60.0
    )
    response.col.flush()

    return JSONResponse(content=response, status_code=200)


@app.get("/list_documents/")
async def list_documents_endpoint(list_documents: ListDocumentsInput):
    """
     Endpoint for listing documents in a given collection.
    """
    collection = list_documents.collection.strip().lower()
    documents = {}
    if milvus_client.has_collection(collection):
        collection = milvus_client.list_collections()
        result = collection.query(expr='source != ""', output_fields = ["source", "name"])
        for r in result:
            documents[r['name']] = r['source']
    return JSONResponse(content={"documents": documents}, status_code=200)


@app.delete("/drop_collection/")
async def remove_collection_endpoint(drop_collection_input: DropCollectionInput):
    """
     Endpoint for removing collection
    """
    milvus_client.delete(
        collection_name=drop_collection_input.collection)
    return JSONResponse(content={"result": 'ok'}, status_code=200)


@app.get("/list_collections/")
async def list_collections_endpoint():
    """
     Endpoint for listing collections
    """
    collections = milvus_client.list_collections()
    return JSONResponse(content={"collections": collections}, status_code=200)


class TTSInput(BaseModel):
    text: str
    voice: str = "Spanish-US.Female-1"
    sample_rate_hz: int = 44100


@app.post("/TTS-Spanish/")
async def tts_endpoint(tts_input: TTSInput):
    try:
        riva_tts = riva.client.SpeechSynthesisService(AUTH_RIVA)
        req = {
            # "language_code": "en-US",
            "language_code": "es-US",
            "encoding": riva.client.AudioEncoding.LINEAR_PCM,
            "sample_rate_hz": tts_input.sample_rate_hz,
            "voice_name": tts_input.voice,
            "text": tts_input.text
        }

        resp = riva_tts.synthesize(**req)
        audio_samples = np.frombuffer(resp.audio, dtype=np.int16)
        audio_file_path = "/tmp/output.wav"

        with wave.open(audio_file_path, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(tts_input.sample_rate_hz)
            wav_file.writeframes(audio_samples.tobytes())

        return FileResponse(audio_file_path, media_type="audio/wav", filename="output.wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def generate_audio_chunks(audio_stream, chunk_size=1024):
    return (audio_stream[i:i + chunk_size] for i in range(0, len(audio_stream), chunk_size))


@app.post("/STT/Offline/Spanish")
async def asr_endpoint(audio_file: UploadFile = File(...)):
    try:
        riva_asr = ASRService(AUTH_RIVA)
        audio_content = await audio_file.read()
        config = riva.client.RecognitionConfig(
            language_code="es-US",
            max_alternatives=2,
            enable_automatic_punctuation=True,
            audio_channel_count=1
        )

        response = riva_asr.offline_recognize(audio_content, config)
        asr_best_transcript = response.results[0].alternatives[0].transcript

        return {
            "transcript": asr_best_transcript,
            "full_response": str(response)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/STT/Offline/English")
async def asr_endpoint(audio_file: UploadFile = File(...)):
    try:
        riva_asr = ASRService(AUTH_RIVA)
        audio_content = await audio_file.read()
        config = riva.client.RecognitionConfig(
            language_code="en-US",  # Language code of the audio clip, set to Spanish
            max_alternatives=1,  # How many top-N hypotheses to return
            enable_automatic_punctuation=True,  # Add punctuation when end of VAD detected
            audio_channel_count=1  # Mono channel
        )
        response = riva_asr.offline_recognize(audio_content, config)
        asr_best_transcript = response.results[0].alternatives[0].transcript

        return {
            "transcript": asr_best_transcript,
            "full_response": str(response)  # Optionally return the full response message
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/version")
async def get():
    return "0.1"


@app.get("/test-milvus-connection/")
async def test_milvus_connection():
    print("MILVUS FILE", PATH_TLS)
    try:
        connections.connect(
            uri=URI_MILVUS,
            secure=True,
            server_pem_path=PATH_TLS,
            server_name=SERVER_N,
            user=MILVUS_USER,
            password=MILVUS_PASSWORD,
            db_name=MILVUS_DB,
            timeout=30.0  # Change its necessary
        )

        databases = db.list_database()
        connections.disconnect("default")
        return {"status": "success", "message": "Connected to Milvus successfully!", "databases": databases}
    except Exception as e:
        connections.disconnect("default")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/health')
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    print("WORKING IN THE MAIN")
    uvicorn.run(app, host="0.0.0.0", port=8109)
