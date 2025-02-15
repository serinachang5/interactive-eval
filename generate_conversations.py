from constants_and_utils import *
import json
import pandas as pd
import evaluate

VALID_SIM = ["two_step", "two_step_problem", "two_step_help", "iqa_eval"]

def convert_conversation_to_str(conversation, cutoff=100, user_term="YOU", assistant_term="SYSTEM"):
    """
    Convert a conversation to a string, to provide as input to LLM.
    """
    s = ""
    for u in conversation:
        if u['role'] == 'user' or u['role'] == 'You':  # keep entire utterance
            s += f"{user_term}: {u['content']}\n\n"
        elif u['role'] == 'assistant' or u['role'] == 'Bot':
            if cutoff is None:
                content = u['content']
            else:
                toks = u['content'].split(' ')  # keep first <cutoff> tokens
                if len(toks) > cutoff:
                    toks = toks[:cutoff] + ['...']
                content = ' '.join(toks)
            s += f"{assistant_term}: {content}\n\n"
        else:
            raise Exception(f"Unknown role: {u['role']}")
    return s[:-2]  # drop final \n

def two_step_simulator(row, sim_model, conversation, benchmark='MMLU', sim_temp=DEFAULT_SIM_TEMP, variant="", 
                       require_answer=False, return_prompts=False):
    """
    Our simulator: (1) given question, generate first prompt, (2) given question and conversation, generate next prompt or answer question.
    """
    question, _ = extract_qa_from_row(row, benchmark)
    sys_prompt = f"You are a human user interacting with an AI system, and you are trying to answer the following question:\n\n{question}"
    if len(conversation) == 0:
        user_prompt = "Generate the first prompt you would say to the system to get started with answering your question. "
        if variant == "":  # original
            user_prompt += "Remember to write exactly as a real user would."
        if variant == "problem":
            user_prompt += "Include the problem statement in some form, and remember to write exactly as a real user would."
        if variant == "help":
            return "I'm trying to solve a math problem. Can you help me get started on the first step?", False
    else:
        assert len(conversation) >= 2
        conv_str = convert_conversation_to_str(conversation, cutoff=None, user_term="YOU", assistant_term="SYSTEM")
        user_prompt = f"Here is your conversation so far with the AI system:\n{SEP}\n{conv_str}\n{SEP}\n"
        ans_options = ", ".join([f"or {l}" if l == BENCHMARK_ANSWER_OPTIONS[benchmark][-1] else l for l in BENCHMARK_ANSWER_OPTIONS[benchmark]])
        if require_answer:
            user_prompt += f"Answer the question based on this conversation. Return ONLY the answer in the format \"Answer: {ans_options}\"."
        else:
            user_prompt += (
                f"If your question is answered by this conversation, return ONLY the answer in the format \"Answer: {ans_options}\". "
                "If not, generate the next prompt you would say to the system to answer your question. Remember to keep your writing style consistent."
            )
    if return_prompts:
        return sys_prompt, user_prompt 
    
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = query_model(sim_model, messages, kwargs={'temperature': sim_temp}).strip()
    response = response.strip('\"')  # strip quotations
    if response.startswith("YOU: "):
        response = response[len("YOU: "):].strip()
    if len(conversation) > 0 and response.startswith("Answer: "):
        answer = response[len("Answer: "):]
        if answer[0] in BENCHMARK_ANSWER_OPTIONS[benchmark]:  # first character should match an answer option
            return answer[0], True 
        print("Warning: received invalid answer from two-step simulator")
        return answer, True 
    return response, False

def iqa_eval_simulator(row, sim_model, conversation, benchmark='MMLU', sim_temp=DEFAULT_SIM_TEMP, 
                       require_answer=False, return_prompts=False):
    """
    User simulator from IQA-EVAL (Li et al., 2024).
    """
    question, _ = extract_qa_from_row(row, benchmark)
    ans_options = ", ".join([f"or {l}" if l == BENCHMARK_ANSWER_OPTIONS[benchmark][-1] else l for l in BENCHMARK_ANSWER_OPTIONS[benchmark]])
    sys_prompt = "You are mimicking a human. You are trying to choose the correct answer to the given question. "
    if require_answer:
        sys_prompt += (
            "Please choose the correct answer to the given question. "
            f"Please output \"So, the answer is: {ans_options}.\"\n\n{question}"
        )
    else:
        sys_prompt += (
            "Please ask an assistant sub-questions for help approaching answers. "
            "In each turn, please only ask one sub-question to interact with an assistant. "
            "In the sub-questions, please include all necessary information, such as the question and options, in the original question. "
            f"If you know the answer, please output \"So, the answer is: {ans_options}.\"\n\n{question}"
        )
    if len(conversation) > 0:
        conv_str = convert_conversation_to_str(conversation, cutoff=None, user_term="YOU", assistant_term="ASSISTANT")
        sys_prompt += "\n\n" + conv_str 
    if return_prompts:
        return sys_prompt, None
    
    messages = [
        {"role": "system", "content": sys_prompt},
    ]
    response = query_model(sim_model, messages, kwargs={'temperature': sim_temp}).strip()
    response = response.lstrip('\"').strip('\"')  # strip quotations
    if response.startswith("YOU: "):
        response = response[len("YOU: "):].strip()  # strip leading "USER:"
    if "So, the answer is: " in response:
        index = response.index("So, the answer is: ")
        answer = response[index + len("So, the answer is: "):]
        if answer[0] in BENCHMARK_ANSWER_OPTIONS[benchmark]:  # first character should match an answer option
            return answer[0], True 
        print("Warning: received invalid answer from IQA-EVAL simulator")
        return answer, True 
    return response, False


def generate_simulated_conversation(simulator_method, sim_model, sys_model, row, benchmark='MMLU', max_turns=10, verbose=False):
    """
    Generate conversation between simulator model and system model, where the simulator is trying to
    answer a benchmark question.
    """
    assert simulator_method in VALID_SIM
    conversation = []
    for t in range(max_turns+1):
        require_answer = t == max_turns
        if simulator_method == "two_step":
            prompt, is_answer = two_step_simulator(row, sim_model, conversation, benchmark=benchmark, variant="", require_answer=require_answer)
        elif simulator_method == "two_step_problem":
            prompt, is_answer = two_step_simulator(row, sim_model, conversation, benchmark=benchmark, variant="problem", require_answer=require_answer)
        elif simulator_method == "two_step_help":
            prompt, is_answer = two_step_simulator(row, sim_model, conversation, benchmark=benchmark, variant="help", require_answer=require_answer)
        else:
            prompt, is_answer = iqa_eval_simulator(row, sim_model, conversation, benchmark=benchmark, require_answer=require_answer)
        if is_answer:  # user got the answer to their question, prompt is their answer
            return conversation, prompt 
        conversation.append({'role': 'user', 'content': prompt})
        if verbose:
            print("\nUSER:", prompt)
        resp = get_next_system_response(sys_model, conversation)
        conversation.append({'role': 'assistant', 'content': resp}) 
        if verbose:
            print("\nSYSTEM:", resp)
    return conversation, None 

def get_next_system_response(sys_model, conversation, sys_temp=DEFAULT_TEMP):
    """
    Get next system response from the evaluated system, given the conversation so far.
    """
    assert len(conversation) >= 1
    assert conversation[-1]['role'] == "user"
    messages=[{"role": "system", "content": "You are a helpful AI assistant."}] + conversation
    response = query_model(sys_model, messages, kwargs={'temperature': sys_temp})
    return response 

def get_first_prompts_for_question(qid, sim_results, conversations, sim_method):
    """
    Get first prompts from simulator and human users for this question.
    """
    q_results = sim_results[qid]
    sim_prompts = []
    for conv in q_results[sim_method]:
        if len(conv["conversation"]) > 0:
            first_prompt = conv["conversation"][0]
            assert first_prompt["role"] == "user"
            sim_prompts.append(first_prompt["content"])
    if len(sim_prompts) != 10:
        print(f"Warning: {qid} has {len(sim_prompts)} prompts for {sim_method}")

    user_prompts = []
    for conv in conversations[qid]:
        first_prompt = conv["chatHistory"][0]
        if first_prompt["role"] == "You":
            user_prompts.append(first_prompt["content"])
    
    return sim_prompts, user_prompts

def get_bleu_and_rouge_for_question(qid, sim_results, conversations, bleu_metric, rouge_metric,
                                    sim_method="two_step"):
    """
    Get all pairs of prompts (simulator and human), compute BLEU and ROUGE scores.
    """
    sim_prompts, user_prompts = get_first_prompts_for_question(qid, sim_results, conversations, sim_method)
    # get all pairs of prompts
    all_sim_prompts = []
    all_user_prompts = []
    for p1 in sim_prompts:
        for p2 in user_prompts:
            all_sim_prompts.append(p1)
            all_user_prompts.append(p2)
    bleu_scores = bleu_metric.compute(predictions=all_sim_prompts, references=all_user_prompts)
    rouge_scores = rouge_metric.compute(predictions=all_sim_prompts, references=all_user_prompts)
    return bleu_scores, rouge_scores

def get_bleu_and_rouge_for_all_questions(sim_model, sys_model, split, sim_method="two_step"):
    """
    Get BLEU and ROUGE scores for all questions in the split.
    """
    fn = f"./MMLU_results/fine_tuning/{split}_{sim_model}_{sys_model}.json"
    with open(fn, "r") as f:
        sim_results = json.load(f)
    with open("../ai-user-sim-exp-orig/prolific_results/full_20250207/conversations.json", "r") as f:
        conversations = json.load(f)
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")

    qids = [k for k in sim_results if "-redux-" in k]
    print(f"Number of questions: {len(qids)}")
    all_bleu = []
    all_rouge = []
    for i, qid in enumerate(qids):
        bleu, rouge = get_bleu_and_rouge_for_question(qid, sim_results, conversations, bleu_metric, rouge_metric,
                                                      sim_method=sim_method)
        print(f"{i}. {qid}, BLEU={bleu["bleu"]:.3f}, ROUGE={rouge["rouge1"]:.3f}", flush=True)
        all_bleu.append(bleu["bleu"])
        all_rouge.append(rouge["rouge1"])
    print(f"Finished! BLEU: mean={np.mean(all_bleu)}, std={np.std(all_bleu)}; ROUGE: mean={np.mean(all_rouge)}, std={np.std(all_rouge)}", flush=True)
    with open(f"./MMLU_results/fine_tuning/{split}_{sim_model}_{sys_model}_{sim_method}_bleu_rouge.json", "w") as f:
        json.dump({"bleu": all_bleu, "rouge": all_rouge}, f, indent=2)


if __name__ == "__main__":
    sim_model = "pretrained-gpt-4o"
    sys_model = "llama-3.1-8b"
    split = "test"
    get_bleu_and_rouge_for_all_questions(sim_model, sys_model, split, sim_method="iqa_eval")
    