#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp, HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os
import argparse
import time
import ast
import argostranslate.package
import argostranslate.translate

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))
translate_from_code = os.environ.get('TRANSLATE_FROM_CODE')
translate_to_code = os.environ.get('TRANSLATE_TO_CODE')
transformers_offline = os.environ.get('TRANSFORMERS_OFFLINE')
if transformers_offline == "1":
    embeddings_model_name = "models/" + embeddings_model_name
    model_path = "models/" + model_path

from constants import CHROMA_SETTINGS

def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case "HuggingFace":
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
            pipeline_params = {"task":"text-generation", "model":model, "tokenizer":tokenizer,
                "max_new_tokens":32, "framework":"pt",
                "pad_token_id":tokenizer.pad_token_id, "bos_token_id":tokenizer.bos_token_id, "eos_token_id":tokenizer.eos_token_id}
            if os.environ.get('PIPELINE_PARAMS'):
                pipeline_params.update(ast.literal_eval(os.environ.get('PIPELINE_PARAMS')))
            pipe = pipeline(**pipeline_params)
            llm = HuggingFacePipeline(pipeline=pipe)
        case _default:
            # raise exception if model_type is not supported
            raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        start = time.time()
        if translate_from_code and translate_to_code and translate_from_code != translate_to_code:
            res = qa(argostranslate.translate.translate(query, translate_from_code, translate_to_code))
            answer, docs = argostranslate.translate.translate(res['result'], translate_to_code, translate_from_code), [] if args.hide_source else res['source_documents']
        else:
            res = qa(query)
            answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        end = time.time()

        # Print the result
        print("\n\n> Question:")
        print(query)
        print(f"\n> Answer (took {round(end - start, 2)} s.):")
        print(answer)

        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            if translate_from_code and translate_to_code and translate_from_code != translate_to_code:
                print(argostranslate.translate.translate(document.page_content, translate_to_code, translate_from_code))
            else:
                print(document.page_content)

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    main()
