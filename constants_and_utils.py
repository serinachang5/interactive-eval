import ast
from azure.identity import AzureCliCredential, ChainedTokenCredential, DefaultAzureCredential, get_bearer_token_provider
import json
import numpy as np
from openai import OpenAI, AzureOpenAI
import os 
import sys
import io
import ssl
import urllib.request
from datasets import load_dataset
import pandas as pd

MODEL_NAMES = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo",
               "llama-3.1-70b", "llama-3.1-8b", "llama-3-70b", "llama-3-8b", 
               "o1", "phi-4", "fine-tuned-full", "pretrained-gpt-4o"]
EMB_MODELS = ["text-embedding-ada-002"]
DEFAULT_TEMP = 0.7
DEFAULT_SIM_TEMP = 1.0 
DEFAULT_MAX_TOKENS = 1000
PATH_TO_USER_STUDY = "../ai-user-sim-exp-orig/prolific-pilot/"
SEP = "========================"
BENCHMARK_ANSWER_OPTIONS = {
    "MMLU": ["A", "B", "C", "D"],
}
VALID_BENCHMARKS = ["MMLU"]

########################################################################
# Functions to call LLMs.
########################################################################
# REMOVED TO PRESERVE ANONYMITY.

########################################################################
# Load and process MMLU and MMLU-Redux datasets.
########################################################################
def load_MMLU_dataset(dataset, split):
    """
    Load MMLU dataset from Hugging Face.
    """
    ds = load_dataset("cais/mmlu", dataset, split=split)
    rows = []
    for i, d in enumerate(ds):
        row = edit_mmlu_row(d)
        row['questionID'] = f"{dataset}-{split}-{i}"
        rows.append(row)
    options = [f'option_{l}' for l in BENCHMARK_ANSWER_OPTIONS['MMLU']]
    col_order = ['question'] + options + ['answer']
    df = pd.DataFrame(rows).set_index('questionID')[col_order]
    df = df.drop_duplicates(subset=["question"] + options)
    return df 

def load_MMLU_redux_dataset(dataset):
    """
    Load MMLU-Redux dataset from Hugging Face. Make sure ground-truth answers match MMLU.
    """
    ds = load_dataset("edinburgh-dawg/mmlu-redux-2.0", dataset, split="test")  # only test for MMLU-Redux
    assert len(ds) == 100
    rows = []
    for i, d in enumerate(ds):
        row = edit_mmlu_row(d)
        row['questionID'] = f"{dataset}-redux-{i}"
        rows.append(row)
    options = [f'option_{l}' for l in BENCHMARK_ANSWER_OPTIONS['MMLU']]
    col_order = ['question'] + options + ['answer', 'error_type', 'source', 'correct_answer', 'potential_reason']
    df = pd.DataFrame(rows).set_index('questionID')[col_order]
    return df 

def edit_mmlu_row(row):
    """
    Expand out list of choices into separate columns, replace answer index with answer letter.
    """
    for i, l in enumerate(BENCHMARK_ANSWER_OPTIONS['MMLU']):
        row[f'option_{l}'] = row["choices"][i]
    row['answer'] = BENCHMARK_ANSWER_OPTIONS['MMLU'][row['answer']]
    return row 

def extract_qa_from_row(row, benchmark, include_options=True):
    """
    Extract question (as string) and answer from row in dataframe.
    """
    assert benchmark in VALID_BENCHMARKS
    if benchmark == 'MMLU':
        question = f"{row['question']}"
        if include_options:
            question += "\n"
            for opt in BENCHMARK_ANSWER_OPTIONS['MMLU']:
                question += f"{opt}. {row['option_' + opt]}\n"
            question = question[:-1]  # remove trailing \n
        answer = row['answer']
    return question, answer

def convert_mmlu_id_to_mmlu_redux_id(ds):
    """
    For HS and college math, we used IDs from MMLU csv's, instead of IDs from MMLU-Redux on Hugging Face.
    Want to 1) check that the CSV and Hugging Face match, and 2) convert MMLU IDs to MMLU Redux IDs.
    """
    id_cols = ["question", "option_A", "option_B", "option_C", "option_D", "answer"]

    # first check that CSV and Hugging Face match
    df_csv = pd.read_csv(f"./MMLU/test/{ds}_test.csv", header=None, names=["question", "option_A", "option_B", "option_C", "option_D", "answer"])
    df_csv = df_csv.drop_duplicates(id_cols)
    df_csv["ID"] = np.arange(len(df_csv))
    df_hf = load_MMLU_dataset(ds, split="test")
    df_hf["ID"] = np.arange(len(df_hf))
    assert len(df_csv) == len(df_hf)
    assert all(df_csv["answer"].values == df_hf["answer"].values)
    merged = df_hf.merge(df_csv, left_on=id_cols, right_on=id_cols, how="left")
    matched = ~merged["ID_y"].isnull()  # should be close to 100% but maybe slightly below if formatting has changed
    print(f"csv vs Hugging Face: matched {matched.sum()} out of {len(df_hf)} MMLU questions")  
    assert all(merged[matched]["ID_x"].values == merged[matched]["ID_y"].values)  # where they match, IDs should match

    # now convert MMLU ID to MMLU Redux ID
    df_redux = load_MMLU_redux_dataset(ds)
    df_redux["ID"] = np.arange(len(df_redux))
    merged = df_hf.merge(df_redux, left_on=id_cols, right_on=id_cols, how="left")
    matched = ~merged["ID_y"].isnull()
    print(f"Hugging Face MMLU vs redux: matched {matched.sum()} out of {len(df_redux)} MMLU-Redux questions")
    mmlu2redux = dict(zip(merged["ID_x"], merged["ID_y"]))
    return mmlu2redux

def convert_row_to_latex(question_num, row):
    """
    Format question and answer as latex string.
    """
    s = "\\noindent \\textbf{Question " + str(question_num) + ":} " + row['question'] + "\\\\\n\n"
    for letter in ['A', 'B', 'C', 'D']:
        if row['option_' + letter].startswith("\\frac"):
            s += "\\noindent \\textbf{" + letter + ":} $" + row['option_' + letter] + "$\\\\\n\n"
        else:
            s += "\\noindent \\textbf{" + letter + ":} " + row['option_' + letter] + "\\\\\n\n"
    s += "\\noindent \\textbf{Correct Answer:} " + row['answer'] + "\\\\\\\\\n\n"
    return s

def format_dict_as_str(d):
    """
    Helper function to format a dict as a string.
    """
    s = "{\n"
    for k, v in d.items():
        k_str = f"\"{k}\"" if type(k) == str else k
        v_str = f"\"{v}\"" if type(v) == str else v
        s += f"    {k_str} : {v_str},\n"
    s += "}"
    return s

def convert_json_question_to_mathjax(question_dict, model="o1", return_prompts=False):
    """
    Add MathJax to question text and choices.
    """
    system_prompt = "You are a helpful AI assistant."
    user_prompt = "Edit this json so that, if there are mathematical expressions in \"questionText\" or \"choices\", add inline MathJax using \\\\( and \\\\). Enclose ONLY the mathematical expressions, not other text, and don't edit any fields besides \"questionText\" and \"choices\". Return the json in the same format.\n"
    user_prompt += format_dict_as_str(question_dict) + "\n"
    if return_prompts:
        return system_prompt, user_prompt
    response = query_model(model, [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
    return response 

def convert_json_questions_to_mathjax(json_fn, model="o1"):
    """
    Add MathJax to all questions.
    """
    with open(json_fn, "r") as f:
        questions = json.load(f)
    print(f"loaded {len(questions)} questions", flush=True)
    new_fn = json_fn.replace(".json", "_mathjax.json")
    print(f"Saving in {new_fn}")
    if os.path.isfile(new_fn):
        with open(new_fn, "r") as f2:
            new_json = json.load(f2)
        already_done = []
        for q in new_json:
            already_done.append(q['questionID'])
        print(f"Already done with {len(already_done)} questions", flush=True)
    else:
        new_json = []
        already_done = []
    for q in questions:
        if q['questionID'] in already_done:
            print(f"{q['questionID']}: already done!", flush=True)
        else:
            print("\nORIGINAL", flush=True)
            print(format_dict_as_str(q), flush=True)
            response = convert_json_question_to_mathjax(q, model)
            try:
                q_new = json.loads(response)
                new_json.append(q_new)
                print("\nNEW", flush=True)
                print(format_dict_as_str(q_new), flush=True)
            except:
                print("FAILED TO CONVERT RESPONSE TO JSON", flush=True)
                print(response, flush=True)
            with open(new_fn, "w") as f:
                json.dump(new_json, f, indent=4)


if __name__ == '__main__':
    messages = [{"role": "user", "content": 'who are you'}]
    for model in ["llama-3.1-8b"]: # MODEL_NAMES:
        print('MODEL:', model)
        try:
            response = query_model(model, messages)
            print('RESPONSE:', response)
        except:
            print('Failed!')