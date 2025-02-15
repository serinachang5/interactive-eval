from constants_and_utils import *
from generate_conversations import convert_conversation_to_str, generate_simulated_conversation
import json
import pandas as pd
import re
from collections import Counter
import argparse
import sys
import io
import matplotlib.pyplot as plt

VALID_BASELINES = ["letter_only_zero_shot", "letter_only_few_shot", "copy_and_paste", "copy_and_paste_no_mc", "zero_shot_cot", "zero_shot_cot_direct", "multiagent_debate"]

########################################################################
# AI-alone methods.
########################################################################
def letter_only(row, model, benchmark='MMLU', example_rows=None, return_prompts=False):
    """
    Letter-only: restrict the model's answers to one of the letter options.
    Prompt follows HELM's format exactly.
    """
    sys_prompt = "You are a helpful AI assistant."
    user_prompt = f"Answer with only a single letter.\n\n"
    if example_rows is not None:  # provide in-context examples
        ds = row.name.split("-")[0].replace("_", " ")
        user_prompt += f"The following are multiple choice questions (with answers) about {ds}.\n\n"
        for _, ex_row in example_rows.iterrows():
            q, a = extract_qa_from_row(ex_row, benchmark)
            user_prompt += f"Question: {q}\nAnswer: {a}\n\n"
    question, _ = extract_qa_from_row(row, benchmark)
    user_prompt += f"Question: {question}\nAnswer:"
    if return_prompts:
        return sys_prompt, user_prompt
    
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = query_model(model, messages, kwargs={"max_tokens": 10}).strip()
    if response[0] is not None and response[0] in BENCHMARK_ANSWER_OPTIONS[benchmark]:  # try to match answer options
        return response[0]
    print("Warning: received invalid answer in letter_only")
    return response

def copy_and_paste(row, model, include_options=True, benchmark='MMLU'):
    """
    Copy and paste the question and answer options, no other text.
    """
    question, _ = extract_qa_from_row(row, benchmark, include_options=include_options)
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": question}
    ]
    response = query_model(model, messages)
    extracted, json_success = extract_answer_from_response(row, response, benchmark=benchmark, include_options=include_options)
    return response, extracted, json_success

def extract_answer_from_response(row, response, benchmark='MMLU', return_prompts=False, include_options=True):
    """
    Use GPT-4o to extract answer from response.
    """
    system_prompt = "You are a helpful assistant."
    question, _ = extract_qa_from_row(row, benchmark, include_options=include_options)
    user_prompt = f"Here is a question that someone was asked:\n\n{SEP}\n{question}\n{SEP}\n\n" 
    user_prompt += f"Here is a response:\n\n{SEP}\n{response}\n{SEP}\n\n"
    user_prompt += "Did the response provide a final answer to the question? Respond with a JSON object that contains one key \"attempted_answer\" with a value that is true or false. "
    user_prompt += "If \"attempted_answer\" is true, then include a second key \"answer_val\" with the final answer's value in quotations. "
    if include_options:
        answer_opts = [f"\"{a}\"" for a in BENCHMARK_ANSWER_OPTIONS[benchmark]]
        answer_opts = ", ".join(answer_opts[:-1]) + ", or " + answer_opts[-1]
        user_prompt += f"If the final answer value matches one of the answer options, include a third key \"answer_letter\" with a value that is one of the letters {answer_opts}."
    if return_prompts:
        return system_prompt, user_prompt
    
    extracted = query_model("gpt-4o", [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
    if "{" in extracted and "}" in extracted:
        # isolate content within json, drop surrounding text
        extracted = extracted.split("{", 1)[1]
        extracted = extracted.rsplit("}", 1)[0]
    try:
        extracted = json.loads("{" + extracted + "}")
    except Exception as e:
        print(f"Failed to convert to json: {str(e)}\nExtracted: {extracted}")
        return extracted, False  
    if not include_options and "answer_val" in extracted:
        correct = check_correctness(row, extracted["answer_val"], benchmark=benchmark)
        extracted["correct"] = correct
    return extracted, True 

def check_correctness(row, model_answer, benchmark='MMLU', return_prompts=False):
    """
    Check if model answer is correct.
    """
    correct_answer = row['option_' + row['answer']]
    # try numerical comparison first
    if correct_answer.isnumeric() and str(model_answer).isnumeric():
        return bool(np.isclose(float(correct_answer), float(model_answer)))
    system_prompt = "You are a helpful assistant."
    question, _ = extract_qa_from_row(row, benchmark, include_options=False)
    user_prompt = f"Here is a question that someone was asked:\n\n{SEP}\n{question}\n{SEP}\n\n"
    user_prompt += f"The correct answer is: {correct_answer}\n\n"
    user_prompt += f"The person's answer is: {model_answer}\n\n"
    user_prompt += "Was their answer correct? Respond with either \"yes\" or \"no\"."
    if return_prompts:
        return system_prompt, user_prompt
    response = query_model("gpt-4o", [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
    return response.strip(".").lower() == "yes"


def zero_shot_cot(row, model, benchmark='MMLU', answer_extract="orig"):
    """
    Zero-shot chain-of-thought (Kojima et al., 2022): extract reasoning with prompt 1, extract answer with prompt 2.
    """
    assert answer_extract in ["orig", "direct"]
    reasoning = zero_shot_cot_first_turn(row, model, benchmark=benchmark)
    response = zero_shot_cot_second_turn(row, model, reasoning, benchmark=benchmark, answer_extract=answer_extract)
    if response[0] in BENCHMARK_ANSWER_OPTIONS[benchmark]:  # try to match answer options
        return reasoning, response[0]
    print("Warning: received invalid answer in zero_shot_cot")
    return reasoning, response  # worst case, return response, parse later

def zero_shot_cot_first_turn(row, model, benchmark="MMLU", return_prompts=False):
    """
    Do first turn of zero-shot CoT, extract reasoning.
    """
    sys_prompt = "You are a helpful assistant."
    question, _ = extract_qa_from_row(row, benchmark)
    user_prompt = f"Question: {question}\n\nAnswer: Let's think step by step. "
    if return_prompts:
        return sys_prompt, user_prompt
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = query_model(model, messages)
    return response

def zero_shot_cot_second_turn(row, model, reasoning, benchmark="MMLU", answer_extract="orig", 
                              return_prompts=False):
    """
    Do second turn of zero-shot CoT, extract answer from reasoning.
    """
    sys_prompt = "You are a helpful assistant."
    question, _ = extract_qa_from_row(row, benchmark)
    user_prompt = f"Question: {question}\n\nAnswer: Let's think step by step. " + reasoning
    if answer_extract == "orig":
        first_opt = BENCHMARK_ANSWER_OPTIONS[benchmark][0]
        last_opt = BENCHMARK_ANSWER_OPTIONS[benchmark][-1]
        user_prompt += f"\n\nTherefore, among {first_opt} through {last_opt}, the answer is"
    elif answer_extract == "direct":
        answer_opts = [f"\"{a}\"" for a in BENCHMARK_ANSWER_OPTIONS[benchmark]]
        answer_opts = ", ".join(answer_opts[:-1]) + ", or " + answer_opts[-1]
        user_prompt += f"\n\nBased on this reasoning, provide your answer using ONLY the letter of the answer option: {answer_opts}."
    else:
        raise Exception("Unknown answer_extract: " + answer_extract)
    if return_prompts:
        return sys_prompt, user_prompt
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = query_model(model, messages, kwargs={"max_tokens": 10})
    return response


def multiagent_debate(row, model, benchmark, max_rounds=3, verbose=False):
    """
    Conduct a multi-agent debate, following Du et al. (2024).
    """    
    user_prompt, response1 = multiagent_debate_starting(row, model, benchmark=benchmark)
    conv1 = [{'role': 'user', 'content': user_prompt},
             {'role': 'assistant', 'content': response1}]
    _, response2 = multiagent_debate_starting(row, model, benchmark=benchmark)
    conv2 = [{'role': 'user', 'content': user_prompt},
             {'role': 'assistant', 'content': response2}]
    answer1 = extract_debate_answer(response1, benchmark)
    answer2 = extract_debate_answer(response2, benchmark)
    if verbose:
        print('====== ROUND 0 =====')
        print('RESPONSE 1:', response1)
        print('ANSWER 1:', answer1)
        print('\nRESPONSE 2:', response2)
        print('ANSWER 2:', answer2)
    if answer1 == answer2:
        return answer1, conv1, conv2
        
    for r in range(1, max_rounds):
        new_user_prompt1, new_response1 = multiagent_debate_round(model, conv1, response2)
        conv1.extend([{'role': 'user', 'content': new_user_prompt1},
                      {'role': 'assistant', 'content': new_response1}])
        new_user_prompt2, new_response2 = multiagent_debate_round(model, conv2, response1)
        conv2.extend([{'role': 'user', 'content': new_user_prompt2},
                      {'role': 'assistant', 'content': new_response2}])
        response1 = new_response1
        response2 = new_response2
        answer1 = extract_debate_answer(response1, benchmark)
        answer2 = extract_debate_answer(response2, benchmark)
        if verbose:
            print(f'====== ROUND {r} =====')
            print('RESPONSE 1:', response1)
            print('ANSWER 1:', answer1)
            print('\nRESPONSE 2:', response2)
            print('ANSWER 2:', answer2)
        if answer1 == answer2:
            return answer1, conv1, conv2
    
    if answer1 == answer2:
        return answer1, conv1, conv2
    print(f"Warning: did not reach consensus, answer 1={answer1}, answer 2={answer2}")
    return answer1, conv1, conv2  # arbitrarily return first answer

def multiagent_debate_starting(row, model, benchmark="MMLU", return_prompts=False):
    """
    Issue starting prompt in multiagent debate, following Du et al. (2024) MMLU experiments - see Figure A4.
    """
    sys_prompt = "You are a helpful assistant."
    question, _ = extract_qa_from_row(row, benchmark)
    answer_opts = [f"\"{a}\"" for a in BENCHMARK_ANSWER_OPTIONS[benchmark]]
    answer_opts = ", ".join(answer_opts[:-1]) + ", or " + answer_opts[-1]
    user_prompt = f"Can you answer the following question as accurately as possible?\n\n{question}"
    user_prompt += f"\n\nExplain your answer, putting the answer in the form (X) at the end of your response, where X is the answer letter: {answer_opts}."
    if return_prompts:
        return sys_prompt, user_prompt 
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = query_model(model, messages)
    return user_prompt, response

def multiagent_debate_round(model, conversation, other_response, return_prompts=False):
    """
    Issue debate round in multiagent debate, following Du et al. (2024) MMLU experiments - see Figure A4.
    """
    assert len(conversation) % 2 == 0
    assert conversation[-1]['role'] == 'assistant'
    sys_prompt = "You are a helpful assistant."
    user_prompt = f"This is the solution to the problem from other agent: {other_response}"
    user_prompt += "Using the reasoning from other agent as additional advice, can you give an updated answer? Examine your solution and that other agents. Put your answer in the form (X) at the end of your response."
    if return_prompts:
        return sys_prompt, user_prompt 
    
    messages = [{'role': 'system', 'content': sys_prompt}] + conversation
    messages.append({'role': 'user', 'content': user_prompt})
    response = query_model(model, messages)
    return user_prompt, response

def extract_debate_answer(response, benchmark):
    """
    Extract model's answer for multiagent debate, using expected format, (X).
    """
    resp = response[-100:].strip()  # should only check characters near the end of answer
    pattern = r"\(([A-Z])\)"
    matches = re.findall(pattern, resp)
    if len(matches) > 0:
        found = []
        for match in matches:
            if match in BENCHMARK_ANSWER_OPTIONS[benchmark]:
                found.append(match)
        if len(set(found)) == 1:  # only one unique match found
            return list(set(found))[0]
    return None 


def test_baseline_method(method, row, model, benchmark="MMLU", example_rows=None):
    """
    Helper function to call the right baseline method.
    """
    assert method in VALID_BASELINES
    if method == "letter_only_zero_shot":
        return letter_only(row, model, benchmark=benchmark, example_rows=None)
    elif method == "letter_only_few_shot":
        assert example_rows is not None
        return letter_only(row, model, benchmark=benchmark, example_rows=example_rows)
    elif method == "copy_and_paste":
        return copy_and_paste(row, model, benchmark=benchmark, include_options=True)
    elif method == "copy_and_paste_no_mc":
        return copy_and_paste(row, model, benchmark=benchmark, include_options=False)
    elif method == "zero_shot_cot":
        return zero_shot_cot(row, model, benchmark=benchmark, answer_extract="orig")
    elif method.startswith("zero_shot_cot"):
        answer_extract = method.rsplit("_", 1)[1]
        return zero_shot_cot(row, model, benchmark=benchmark, answer_extract=answer_extract)
    elif method == "multiagent_debate":
        return multiagent_debate(row, model, benchmark=benchmark)
    raise Exception("Method not implemented: " + method)


########################################################################
# Experiments
########################################################################
def test_methods_on_benchmark(df, sim_model, sys_model, benchmark, save_fn, num_convs=5, max_turns=10, 
                              sim_variants=None, baselines=None, example_rows=None):
    """
    Run experiment on a series of benchmark questions.
    """
    assert benchmark in VALID_BENCHMARKS
    if sim_variants is None:
        sim_variants = [""]
    if baselines is None:
        baselines = []
    if any(["few_shot" in b for b in baselines]):
        assert example_rows is not None

    params = {'sim_model': sim_model, 'sys_model': sys_model, 'sim_temp': DEFAULT_SIM_TEMP, 'sys_temp': DEFAULT_TEMP, 
              'max_tokens': DEFAULT_MAX_TOKENS, 'benchmark': benchmark, 'num_convs': num_convs, 'max_turns': max_turns, 
              'sim_variants': sim_variants, 'baselines': baselines}
    if os.path.isfile(save_fn):
        print(f"Found {save_fn}, appending new results", flush=True)
        with open(save_fn, "r") as f:
            all_results = json.load(f)
        for k in params:  # check that all parameters match
            if k in ['sim_variants', 'baselines']:
                new_methods = set(params[k]) - set(all_results[k])
                if len(new_methods) > 0:
                    print(f"Note: adding new {k} {new_methods}")
                    all_results[k] = list(set(params[k]).union(all_results[k]))
            elif k == "num_convs":
                assert params[k] >= all_results[k], f"New num_convs {params[k]} should be greater than {all_results[k]}"
                if params[k] > all_results[k]:
                    print(f"Warning: increasing num_convs from {all_results[k]} to {params[k]}", flush=True)
                    all_results[k] = params[k]
            else:  # values should match for all other parameters
                assert params[k] == all_results[k], f"Mismatch on {k}: orig = {all_results[k]}, current = {params[k]}"                
    else:
        all_results = params
    
    for c, row in df.iterrows():
        print(f"============ QUESTION {c} ============", flush=True)
        question, answer = extract_qa_from_row(row, benchmark)
        print("QUESTION:", question, flush=True)
        print("\nCORRECT ANSWER:", answer, flush=True)

        if str(c) in all_results:  
            # we already have partial or complete results for this question
            results = all_results[str(c)]
        else:
            results = row.to_dict()

        added = []
        for k in sim_variants:
            if k not in results or len(results[k]) < num_convs:  # missing this variant
                added.append(k)
                conversations = [] if k not in results else results[k]
                for i in range(len(conversations), num_convs):
                    try:
                        conv, ans = generate_simulated_conversation(k, sim_model, sys_model, row, benchmark=benchmark, 
                                                                    max_turns=max_turns)
                        conv_dict = {'length': len(conv), 'answer': ans, 'correct': ans == answer, 'conversation': conv}
                        print(f"{k} {i}. Conv length: {len(conv)}, Answer: {ans}, Correct: {ans == answer}", flush=True)
                        conversations.append(conv_dict)
                    except Exception as e:
                        if str(e).startswith("Error code: 400"):  # if bad request, try another conversation
                            print(f"{k} {i}. Bad request (error code: 400)", flush=True)
                        else:
                            raise Exception(e)
                results[k] = conversations
            all_results[c] = results 
            with open(save_fn, "w") as f:
                json.dump(all_results, f)   # save along the way

        for b in baselines:
            if b not in results or len(results[b]) < num_convs:  # missing this baseline
                added.append(b)
                b_answers = [] if b not in results else results[b]
                consecutive_error = 0
                for i in range(len(b_answers), num_convs):
                    try:
                        ans = test_baseline_method(b, row, sys_model, benchmark=benchmark, example_rows=example_rows)
                        consecutive_error = 0  # reset 
                        if b == "copy_and_paste":
                            resp, extracted, json_success = ans
                            if json_success:
                                if "answer_letter" in extracted:
                                    answer_dict = {'answer': extracted["answer_letter"], 'answer_val': extracted["answer_val"], 'correct': extracted["answer_letter"] == answer, 'response': resp}
                                elif "answer_val" in extracted:  # has answer but doesn't match options - incorrect
                                    answer_dict = {'answer': None, 'answer_val': extracted["answer_val"], 'correct': False, 'response': resp}
                                else:  # didn't attempt answer - incorrect
                                    answer_dict = {'answer': None, 'correct': False, 'response': resp}
                            else:
                                answer_dict = {'answer': None, 'correct': None, 'response': resp}  # extract_answer failed to return json, correctness unknown
                            print(f"{b} {i}. Extracted: {extracted}", flush=True)
                        elif b == "copy_and_paste_no_mc":
                            resp, extracted, json_success = ans
                            if json_success:
                                if "answer_val" in extracted:
                                    answer_dict = {'answer_val': extracted["answer_val"], 'correct': extracted["correct"], 'response': resp}
                                else:  # didn't attempt answer - incorrect
                                    answer_dict = {'answer_val': None, 'correct': False, 'response': resp}
                            else:
                                answer_dict = {'answer_val': None, 'correct': None, 'response': resp}  # extract_answer failed to return json, correctness unknown  
                            print(f"{b} {i}. Extracted: {extracted}", flush=True)
                        elif 'cot' in b:
                            reasoning, ans = ans
                            answer_dict = {'answer': ans, 'correct': ans == answer, 'reasoning': reasoning}
                            print(f"{b} {i}. Answer: {ans}, Correct: {ans == answer}", flush=True)
                        elif b == 'multiagent_debate':
                            ans, conv1, conv2 = ans 
                            answer_dict = {'answer': ans, 'correct': ans == answer, 'conversation_1': conv1, 'conversation_2': conv2}
                            print(f"{b} {i}. Rounds: {int(len(conv1)/2)}, Answer: {ans}, Correct: {ans == answer}", flush=True)
                        else:
                            assert b.startswith("letter_only")
                            answer_dict = {'answer': ans, 'correct': ans == answer}
                            print(f"{b} {i}. Answer: {ans}, Correct: {ans == answer}", flush=True)
                        b_answers.append(answer_dict)
                    except Exception as e:
                        if str(e).startswith("Error code: 400"):
                            print(f"{b} {i}. Bad request (error code: 400)", flush=True)
                            consecutive_error += 1
                            if consecutive_error > 10:
                                print(f"Too many consecutive errors for {b}, stopping", flush=True)
                                break  # stop trying after 10 consecutive errors
                        else:
                            raise Exception(e)
                results[b] = b_answers
                all_results[c] = results 
                with open(save_fn, "w") as f:
                    json.dump(all_results, f)   # save along the way
        
        if len(added) == 0:
            print(f"Skipping question {c}, already done!", flush=True)
        print(flush=True) 
    print("Finished!!", flush=True)    


def parse_invalid_answer(ans, answer_letters, answer_vals):
    """
    Post-processing to process invalid answers, checking for common patterns.
    """
    assert len(answer_letters) == len(answer_vals)
    val2letter = {}
    full_answers = []
    for l, v in zip(answer_letters, answer_vals):
        val2letter[v] = l 
        full_answers.append(f"{l}. {v}")

    ans = ans[-100:].strip()  # should only check characters at the end of answer
    # see if answer is on its own on the last line
    lines = ans.split("\n")
    if lines[-1] in answer_letters:
        return lines[-1]
    if lines[-1] in answer_vals:
        return val2letter[lines[-1]]
    if lines[-1] in full_answers:
        return lines[-1][0]
    
    patterns = [
        r"\*\*(.+)\*\*",  # **A**
        r"\\boxed{(.+)}",  # \boxed{A}
        r"correct answer is (.+)"  # correct answer is A
    ]
    for p in patterns:
        matches = re.findall(p, ans)
        if len(matches) == 1:
            match = matches[0]
            if match in answer_letters:
                return match 
            elif match in answer_vals:
                return val2letter[match]
            elif match in full_answers:
                return match[0]
    return None


def summarize_benchmark_results(save_fn="", results=None, parse_invalid=False, annotations=None,
                                methods=None, verbose=False):
    """
    Print results from benchmark experiment.
    """
    if len(save_fn) > 0:
        assert os.path.isfile(save_fn), save_fn
        with open(save_fn, "r") as f:
            results = json.load(f)
    else:
        assert results is not None 
    if verbose:
        print(f"Sim: {results['sim_model']}, System: {results['sys_model']}")
    if methods is None:
        methods = results['sim_variants'] + results['baselines']
    if annotations is not None:
        if verbose:
            print("Received annotations about questions")
        to_skip = annotations["answer_wrong"]
        if "copy_and_paste_no_mc" in methods:
            to_skip += annotations["option_dependent"] + annotations["false_negative"]
    else:
        to_skip = []

    answer_options = BENCHMARK_ANSWER_OPTIONS[results['benchmark']]
    question_table = []
    question_keys = [k for k in results.keys() if type(results[k]) == dict and k not in to_skip]
    for k in question_keys:
        question_results = {"questionID": int(k) if k.isnumeric() else k}
        for m in methods:
            if m not in results[k]:
                question_results[m + "_count"] = 0
                question_results[m + "_invalid"] = 0
                question_results[m + "_none"] = 0
            else:
                km_correct = []  # all binary "correct" labels for this question and method
                km_none = 0  # num times where answer was None 
                km_invalid = 0  # num times where answer was not None but didn't match letter options
                for i in range(len(results[k][m])):
                    ans = results[k][m][i].get('answer', None)
                    correct = results[k][m][i].get('correct', None)
                    if ans in answer_options:  # add if ans is one of the letter options
                        assert correct == (ans == results[k]['answer'])
                        km_correct.append(correct)
                    elif m.startswith('copy_and_paste') and correct is not None:
                        # special logic for copy_and_paste: defer to already computed "correct"
                        km_correct.append(correct)
                    elif ans is None:  # missing answer
                        km_none += 1
                    else:  # has answer, but it does not match answer options
                        if parse_invalid:  # try parsing invalid answer 
                            answer_vals = [str(results[k][f"option_{a}"]) for a in answer_options]
                            new_a = parse_invalid_answer(ans, answer_options, answer_vals)
                            if new_a is not None:  # was able to get valid answer
                                km_correct.append(new_a == results[k]['answer'])
                            else:  # still invalid
                                km_correct.append(False)
                                km_invalid += 1
                        else:
                            km_correct.append(False)
                            km_invalid += 1

                assert len(results[k][m]) == (len(km_correct) + km_none)
                question_results[m + "_count"] = len(km_correct)  # total count of non-None answers
                question_results[m + "_mean"] = np.mean(km_correct) if len(km_correct) > 0 else None  # accuracy over non-None answers, treating invalid as False
                question_results[m + "_invalid"] = km_invalid
                n_valid = len(km_correct) - km_invalid
                n_valid_and_correct = sum(km_correct)
                question_results[m + "_mean_valid"] = n_valid_and_correct / n_valid if n_valid > 0 else None  # accuracy over valid answers, dropping invalid, should be >= _mean
                question_results[m + "_none"] = km_none

        question_table.append(question_results)
    question_table = pd.DataFrame(question_table)
    if verbose:
        for m in methods:
            m_df = question_table.dropna(subset=[m + "_mean"])
            n_questions = len(m_df)
            n_answers = m_df[m + "_count"].sum()
            n_none = m_df[m + "_none"].sum()
            print(f"{m}: {n_questions} questions, {n_answers} answers, {n_none} none -> acc = {m_df[m + '_mean'].mean():0.3f}")
        print()
    return question_table


def print_experiment_stats(methods, method_correct, method_none, method_invalid, method_mode):
    """
    Print summary stats from experimental results.
    """
    for m in methods:
        vec = method_correct[m]  # num_questions x num_convs
        mean = np.mean(vec)
        moe = 1.96 * np.sqrt(mean * (1-mean) / len(vec))
        print(f"{m}: {mean:0.3f} ± {moe:0.3f}, with N={len(vec)}; None={method_none[m]}, Invalid={method_invalid[m]}")

        vec = method_mode[m]  # num_questions (taking mode per question)
        if len(vec) > 0:
            mean = np.mean(vec)
            moe = 1.96 * np.sqrt(mean * (1-mean) / len(vec))
            print(f"{m} (mode): {mean:0.3f} ± {moe:0.3f}, with N={len(vec)}")
        print()


def report_results_over_multiple_experiments(fns, sim_model, sys_model, benchmark, num_convs, sim_variants, baselines,
                                             annotations=None):
    """
    Combine results from multiple experiments and print summary.
    """
    all_results = {'sim_model': sim_model, 'sys_model': sys_model, 'benchmark': benchmark, 'num_convs': num_convs,
                   'sim_variants': sim_variants, 'baselines': baselines}
    methods = ["conversations" if len(v) == 0 else f"conversations_{v}" for v in sim_variants] + baselines
    method_correct = {m:[] for m in methods}
    method_mode = {m:[] for m in methods}
    method_none = {m:0 for m in methods}
    method_invalid = {m:0 for m in methods}
    for fn in fns:
        ds = fn.split('/')[-1].split('.')[0]
        ds = '_'.join(ds.split('_')[:-2])
        with open(fn, "r") as f:
            results = json.load(f)
        for k in ['sim_model', 'sys_model', 'benchmark', 'num_convs']:
            assert results[k] == all_results[k]
        for s in sim_variants:
            assert s in results['sim_variants'], "missing sim variant " + s
        for b in baselines: 
            assert b in results['baselines'], "missing baseline " + b
        if annotations is None:
            f_method_correct, f_method_none, f_method_invalid, f_method_mode, _ = summarize_benchmark_results(
                results=results, verbose=False)
        else:
            assert ds in annotations, ds + " missing from annotations"
            f_method_correct, f_method_none, f_method_invalid, f_method_mode, _ = summarize_benchmark_results(
                results=results, annotations=annotations[ds], verbose=False)
        total = len(f_method_correct[methods[0]]) + f_method_none[methods[0]] + f_method_invalid[methods[0]]
        print(f"{ds}: {total} labels, {int(total/num_convs)} questions")
        for m in methods:
            method_correct[m] += f_method_correct[m]
            method_none[m] += f_method_none[m]
            method_invalid[m] += f_method_invalid[m]
            method_mode[m] += f_method_mode[m]

    print()
    print_experiment_stats(methods, method_correct, method_none, method_invalid, method_mode)
    return method_correct, method_none, method_invalid, method_mode


def plot_conversation_lengths(save_fn):
    """
    Plot conversation lengths over different simulator variants.
    """
    assert os.path.isfile(save_fn), save_fn
    with open(save_fn, "r") as f:
        results = json.load(f)
    sim_variants = ["conversations" if len(v) == 0 else f"conversations_{v}" for v in results['sim_variants']]
    method2lens = {s:[] for s in sim_variants}
    question_keys = [k for k in results.keys() if k.isnumeric()]
    for k in question_keys:
        for s in sim_variants:   
            if s in results[k]:     
                for i in range(results['num_convs']):
                    method2lens[s].append(results[k][s][i]['length'] / 2)
            
    plt.figure(figsize=(5,4))
    lengths = np.arange(1, 6, 1)
    for s, s_lens in method2lens.items():
        counts = dict(Counter(s_lens).most_common())
        plt.scatter(lengths, [counts.get(l, 0)/len(s_lens) for l in lengths], label=s)
    plt.legend(fontsize=12)
    plt.xlabel('Num turns', fontsize=12)
    plt.ylabel('Prop conversations', fontsize=12)
    plt.grid(alpha=0.2)
    plt.xticks(lengths)
    plt.show()

def write_df_to_text_file(dataset, split, save_fn, methods, method_names=None, only_redux=False):
    """
    Write dataset to text file so that we can review for user study. 
    Include QA accuracy from GPT-4o.
    """
    assert split in ['test', 'val_dev']
    if only_redux:
        assert split == "test"
        df = load_MMLU_redux_dataset(dataset)
        o1_results_fn  = f"MMLU_results/redux/{dataset}_o1_o1.json"
        with open(o1_results_fn) as o1_f:
            o1_results = json.load(o1_f)
    else:
        if split == "test":
            df = load_MMLU_dataset(f'./MMLU/{split}/{dataset}_{split}.csv')
        else:
            val_df = load_MMLU_dataset(f'./MMLU/val/{dataset}_val.csv')
            dev_df = load_MMLU_dataset(f'./MMLU/dev/{dataset}_dev.csv')
            df = pd.concat([val_df, dev_df]).reset_index(drop=True)
    print(f"Loaded {len(df)} questions from {dataset} {split} (only redux = {only_redux})")

    results_fn = f'MMLU_results/{split}/{dataset}_gpt-4o_gpt-4o.json'
    if os.path.isfile(results_fn):
        with open(results_fn, "r") as results_f:
            results = json.load(results_f)  
        methods_to_keep = [m for m in methods if m in results['0']]
        print('Loaded GPT-4o results, keeping methods', methods_to_keep)
    else:
        print("GPT-4o do not exist")
        results = None
    if method_names is not None:
        method2name = dict(zip(methods, method_names))
    else:
        method2name = {m: m for m in methods}

    method_sums = {m: 0 for m in methods_to_keep}
    method_ns = {m: 0 for m in methods_to_keep}
    with open(save_fn, 'w', encoding="utf-8") as f:
        for idx, row in df.iterrows():
            f.write("==================================================\n")
            if results is not None and str(idx) in results:
                question_results = results[str(idx)]
                assert question_results['question'] == row['question']
                printout = f"Question {idx}:"
                for m in methods_to_keep:
                    correct = [r['correct']==True for r in question_results[m] if r['correct'] is not None]
                    method_sums[m] += int(np.sum(correct))
                    method_ns[m] += len(correct)
                    printout += f" {method2name[m]}={int(np.sum(correct))}/{len(correct)},"
                printout = printout[:-1] + "\n\n"  # remove trailing comma
                f.write(printout)
            else:
                f.write(f"Question {idx}:\n\n")
            question, answer = extract_qa_from_row(row, "MMLU")
            f.write(question + "\n")
            f.write(f"Answer: {answer}\n\n")
            if only_redux:
                f.write(f"MMLU-Redux Label: {row['error_type']}\n")
                o1_question_results = o1_results[str(idx)]["copy_and_paste"][0]
                o1_correct = o1_question_results['correct']
                f.write(f"o1 correct: {o1_correct}")
                if not o1_correct:
                    f.write(f" -> o1 answer: {o1_question_results['answer_val']} (reasoning below)\n")
                    f.write(f"{o1_question_results['response']}\n\n")
                else:
                    f.write("\n\n")

        if results is not None:
            f.write("==================================================\n")
            printout = "SUMMARY:"
            for m in methods_to_keep:
                printout += f" {method2name[m]}={method_sums[m]}/{method_ns[m]} ({method_sums[m]/method_ns[m]:0.3f}),"
            printout = printout[:-1] + "\n"
            f.write(printout)
    f.close()

def write_user_sim_conversations_to_text_file(results, question, save_fn, sim_method="conversations"):
    """
    Print conversations between AI system and user simulator for this question.
    """
    assert str(question) in results and sim_method in results[str(question)]
    with open(save_fn, 'w', encoding="utf-8") as f:
        q_data = results[str(question)]
        q_key = f"{q_data["dataset"]}-{q_data["split"]}-{q_data["question_number"]}"
        convs = q_data[sim_method]
        f.write(f"Sim model = {results["sim_model"]}, System model = {results["sys_model"]}\n")
        f.write(f"Found {len(convs)} conversations for question {q_key}\n")
        corr = np.array([c["correct"] == True for c in convs])
        f.write(f"ACCURACY: {corr.sum()}/{len(convs)} ({corr.mean():0.3f})\n\n")

        f.write(q_data["question"] + "\n")
        for letter in ['A', 'B', 'C', 'D']:
            f.write(f"{letter}. {q_data["option_" + letter]}\n")
        correct_letter = q_data["answer"]
        f.write(f"CORRECT ANSWER: {correct_letter}. {q_data["option_" + correct_letter]}\n\n")

        for i, conv in enumerate(convs):
            f.write(SEP + "\n")
            assert conv["length"] % 2 == 0
            f.write(f"Conversation {i+1} ({int(conv["length"]/2)} turns)\n")
            f.write(convert_conversation_to_str(conv["conversation"], cutoff=None, user_term="USER", assistant_term="SYSTEM") + "\n")
            selected_letter = conv["answer"]
            if selected_letter is None:
                f.write(f"SELECTED ANSWER: None (correct={conv["correct"]})\n\n")
            else:
                f.write(f"SELECTED ANSWER: {selected_letter}. {q_data["option_" + selected_letter]} (correct={conv["correct"]})\n\n")
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments on MMLU dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset.")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "dev", "val_dev"], help="Dataset split to use.")
    parser.add_argument("--redux", action="store_true", help="Only keep questions from MMLU-Redux.")
    parser.add_argument("--sim_model", type=str, default="gpt-4o", help="Simulator model to use.")
    parser.add_argument("--sys_model", type=str, default="gpt-4o", help="System model to use.")
    parser.add_argument("--num_convs", type=int, default=1, help="Number of conversations to simulate.")
    args = parser.parse_args()

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')  # so we don't get encoding issues when printing
    examples = None
    if args.redux:
        assert args.dataset != "pilot" and args.dataset != "fine_tune"
        assert args.split == "test"
        df = load_MMLU_redux_dataset(args.dataset)
        print(f"Loaded MMLU-Redux dataset for {args.dataset}: {len(df)} rows")
        save_fn = f"./MMLU_results/redux/{args.dataset}_{args.sim_model}_{args.sys_model}.json"
        examples = load_MMLU_dataset(args.dataset, split="dev")
        assert len(examples) == 5   
    elif args.dataset == "pilot":
        df = pd.read_csv("MMLU/pilot_questions.csv")
        print(f"Loaded pilot dataset: {len(df)} rows")
        save_fn = f"./MMLU_results/pilot/{args.dataset}_{args.sim_model}_{args.sys_model}.json"
    elif args.dataset == "fine_tune":
        fn = "./MMLU_results/redux/user_study_questions.csv"
        user_study_questions = pd.read_csv(fn).rename(columns={"Unnamed: 0": "questionID"}).set_index("questionID")
        with open("./MMLU_results/redux/fine_tuning_split.json", "r") as f:
            fine_tuning_split = json.load(f)
        question_ids = []
        for ds in fine_tuning_split:
            question_ids += fine_tuning_split[ds][args.split]
        df = user_study_questions.loc[question_ids]
        print(f"Loaded {args.split} from fine-tuning dataset: {len(df)} rows")
        save_fn = f"./MMLU_results/fine_tuning/{args.split}_{args.sim_model}_{args.sys_model}.json"
    else:
        if args.split == "val_dev":
            val_df = load_MMLU_dataset(args.dataset, split="val")
            dev_df = load_MMLU_dataset(args.dataset, split="dev")
            df = pd.concat([val_df, dev_df]).reset_index(drop=True)
        else:
            df = load_MMLU_dataset(args.dataset, split=args.split)
        print(f"Loaded {args.split} for {args.dataset}: {len(df)} rows")
        save_fn = f"./MMLU_results/{args.split}/{args.dataset}_{args.sim_model}_{args.sys_model}.json"
    
    test_methods_on_benchmark(df, args.sim_model, args.sys_model, "MMLU", save_fn, num_convs=args.num_convs, max_turns=10, sim_variants=["two_step", "iqa_eval"]) 
                            #   sim_variants=[], baselines=["letter_only_zero_shot", "letter_only_few_shot", "copy_and_paste"], example_rows=examples)
