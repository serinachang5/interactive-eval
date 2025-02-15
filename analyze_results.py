import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import json 
import os
import sys
import pickle
from datetime import datetime

from constants_and_utils import query_model
from generate_conversations import two_step_simulator

DATASETS = ["elementary_mathematics", "high_school_mathematics", "college_mathematics", "conceptual_physics", "moral_scenarios"]
MODELS = ["gpt-4o", "llama-3.1-8b"]

########################################################################
# Functions to load and process data from Prolific study.
########################################################################
def get_question_id_to_batch_mapping():
    """
    Get dictionary mapping question ID to batch number.
    """
    id2batch = {}
    for subject in ["math", "physics", "moral"]:
        with open(f"../prolific-pilot/src/assets/{subject}_batches.json", "r") as f:
            batches = json.load(f)
        for b, batch in enumerate(batches):
            if subject == "math":
                for ds in batch:
                    for q in batch[ds]:
                        id2batch[q["questionID"]] = b
            else:
                for q in batch:
                    id2batch[q["questionID"]] = b
    return id2batch

def get_checkpoints(df, cutoff=5):
    """
    Take raw dataframe from get_mturk_azure_results_by_hit and expand out value column
    for checkpoint data.
    """
    # expand out "value" into separate columns
    checkpoints = []
    for _, row in df[df.variable == "checkpoint_data"].iterrows():
        value = json.loads(row["value"])
        d = {"worker_id": row["worker_id"], "timestamp": value["timestamp"]}
        d.update(json.loads(value["checkpoint"]))
        checkpoints.append(d)
    checkpoints = pd.DataFrame(checkpoints)
    
    # get backup logs from final storage and replace original whenever missing
    backup = get_data_from_final_storage(checkpoints)
    checkpoints = checkpoints[checkpoints.event != "all_data"]
    merged = checkpoints.merge(backup, on=["worker_id", "timestamp", "event", "checkpoint"], 
                               how="outer", suffixes=('', '_backup'))
    print(f"Merged checkpoints with backup: {len(checkpoints)} checkpoints, {len(backup)}, {len(merged)} merged")    
    for col in checkpoints.columns:
        if col + "_backup" in merged.columns:
            # whenever both are present, they should agree
            both_present = (~merged[col].isna()) & (~merged[col + "_backup"].isna())
            assert (merged[both_present][col] != merged[both_present][col + "_backup"]).sum() == 0
            # where backup is present and selectedAnswer is not, use backup
            backup_needed = (merged[col].isna()) & (~merged[col + "_backup"].isna())
            if col == "selectedAnswer":
                print(f"Filling in {backup_needed.sum()} selectedAnswer from backup")
            merged.loc[backup_needed, col] = merged.loc[backup_needed, col + "_backup"]
            backup_needed = (merged[col].isna()) & (~merged[col + "_backup"].isna())
            assert backup_needed.sum() == 0
    checkpoints = merged[checkpoints.columns]

    qids_per_worker = checkpoints.groupby("worker_id")["questionID"].nunique()
    kept_workers = qids_per_worker[qids_per_worker >= cutoff].index
    checkpoints = checkpoints[checkpoints.worker_id.isin(kept_workers)]
    print(f"Kept rows from {len(kept_workers)} workers who answered at least {cutoff} questions")
    return checkpoints 

def get_data_from_final_storage(checkpoints):
    """
    Use final storage as a backup for missing logs.
    """
    all_data = checkpoints[checkpoints.event == "all_data"]
    print(f"Found final storage for {len(all_data)} workers")
    parsed_data = []
    for _, row in all_data.iterrows():
        for d in row.data:
            if "event" in d["checkpoint"]:  # freeform data
                parsed = json.loads(d["checkpoint"])
                parsed["worker_id"] = row["worker_id"]
                parsed["timestamp"] = d["timestamp"]
            else:
                parsed = {"checkpoint": json.loads(d["checkpoint"]),
                          "event": "checkpoint",
                          "worker_id": row["worker_id"],
                          "timestamp": d["timestamp"]}
            parsed_data.append(parsed)
    parsed_data = pd.DataFrame(parsed_data)
    return parsed_data 

def get_answers_from_checkpoints(checkpoints, drop_att_check=True):
    """
    Keep only checkpoints that are user-alone or user-AI answers, add a few columns.
    """
    answers = checkpoints[checkpoints.event.str.endswith("Answer_answer")].copy()
    answers["acc"] = (answers["selectedAnswer"] == answers["correctAnswer"]).astype(int)
    answers["ds"] = answers.questionID.apply(lambda x: x.split("-")[0])
    answers["answer_type"] = answers.event.apply(lambda x: x.split("_")[2])
    answers["question_position"] = answers.event.apply(lambda x: int(x.split("_")[1]))

    print('num answers:', len(answers))
    att_check = answers[answers.questionID == "example-1"]
    print('attention check acc:', att_check.acc.mean())
    if drop_att_check:
        answers = answers[answers.questionID != "example-1"]
        print('num answers without attention check:', len(answers))  # expecting to drop number of workers
    return answers 

def get_confidence_from_checkpoints(checkpoints):
    """
    Keep only checkpoints that are confidence answers, add a few columns.
    """
    confidence = checkpoints[checkpoints.event.str.endswith("_confidence_answer")].copy()
    confidence["ds"] = confidence.questionID.apply(lambda x: x.split("-")[0])
    confidence["question_position"] = confidence.event.apply(lambda x: int(x.split("_")[1]))
    
    found_ds = confidence.ds.unique()
    datasets = [d for d in DATASETS if d in found_ds]
    for ds in datasets:
        subdf = confidence[confidence.ds == ds]
        print(ds)
        counts = subdf.selectedAnswer.value_counts()
        for val in ["not-confident", "somewhat-confident", "very-confident"]:
            print(val, round(counts.loc[val] / len(subdf), 3))
        print()
    return confidence

def get_conversations_from_checkpoints(checkpoints):
    """
    Keep only checkpoints that are user-AI chats, add a few columns.
    """
    conversations = checkpoints[checkpoints["event"].str.endswith("userAIAnswer_chat")].copy()
    conversations["acc"] = (conversations.correctAnswer == conversations.selectedAnswer).astype(int)
    conversations["ds"] = conversations.questionID.apply(lambda x: x.split("-")[0])

    # get some stats
    found_ds = conversations.ds.unique()
    datasets = [d for d in DATASETS if d in found_ds]
    for ds in datasets:
        subdf = conversations[conversations.ds == ds]
        print(ds, len(subdf), "conversations")
        conv_lengths = []
        user_lengths = []
        ai_lengths = []
        for _, row in subdf.iterrows():
            conv_lengths.append(len(row["chatHistory"]))
            for c in row["chatHistory"]:
                if c["role"] == "You":
                    user_lengths.append(len(c["content"]))
                else:
                    ai_lengths.append(len(c["content"]))
        strings = ["chat length", "user utterance length", "AI utterance length"]
        for l, s in zip([conv_lengths, user_lengths, ai_lengths], strings):
            print(f'\t{s}: mean={np.mean(l):.2f}, SE={np.std(l)/np.sqrt(len(l)):.2f}')
        acc = subdf.acc.mean()
        se = np.sqrt(acc * (1-acc) / len(subdf))
        print(f"\tacc={acc:.3f}, SE={se:.3f}")
    return conversations

########################################################################
# Functions to analyze processed data.
########################################################################
def group_by_question_and_summarize(subdf, label):
    """
    Helper function to group by question and summarize mean accuracy and standard error.
    """  
    if len(subdf) == 0:
        print(f"\t{label}: no answers")
        return np.nan, np.nan
    subdf_q = subdf.groupby("questionID")["acc"].mean()  # mean acc per question
    n_questions = len(subdf_q)
    n_answers = len(subdf)
    acc = subdf_q.mean()  # mean over questions
    se = np.std(subdf_q) / np.sqrt(n_questions)  # SE on the mean, ignores per-question uncertainty
    print(f"\t{label}: {n_questions} questions, {n_answers} answers -> acc={acc:.3f} (SE={se:.3f})")
    return acc, se

def compare_study_results_and_ai_alone(answers, ai_alone_results=None, plot_title=None, color="tab:blue"):
    """
    Summarize results from user study and compare to AI-alone.
    """        
    datasets = [ds for ds in DATASETS if ds in answers.ds.unique()]
    models = [m for m in MODELS if m in answers.model.unique()]
    _, axes = plt.subplots(len(datasets), len(models), figsize=(len(models)*2.5, len(datasets)*2.5), sharey=True)
    all_plot_results = {}
    for ds, axes_row in zip(datasets, axes):
        for model, ax in zip(models, axes_row):
            plot_results = {}
            subdf = answers[(answers.ds == ds) & (answers.model == model)]
            n_workers = len(subdf.worker_id.unique())
            print(f"{ds}, {model}: {len(subdf)} answers from {n_workers} workers")
            if ai_alone_results is not None:
                ai_alone = ai_alone_results[(ai_alone_results.dataset == ds) & (ai_alone_results.model == model)]
                for m in ["letter_only_few_shot", "copy_and_paste"]:
                    m_df = ai_alone.dropna(subset=[m + "_mean"])
                    n_questions = len(m_df)
                    n_answers = m_df[m + "_count"].sum()
                    accs = m_df[m + "_mean"]  # mean acc per question
                    acc = accs.mean()
                    se = np.std(accs) / np.sqrt(len(accs))  # SE on the mean, ignores per-question uncertainty
                    print(f"\tAI-alone, {m}: {n_questions} questions, {n_answers} answers -> acc = {acc:0.3f} (SE = {se:0.3f})")
                    plot_results["AI-alone, letter-only" if m == "letter_only_few_shot" else "AI-alone, free-text"] = (acc, se)
                
            user_alone = subdf[subdf.answer_type == "userAnswer"]
            acc, se = group_by_question_and_summarize(user_alone, "User-alone")
            plot_results["User-alone"] = (acc, se)
            user_ai_direct = subdf[(subdf.answer_type == "userAIAnswer") & (subdf.condition.str.startswith("onlyAI"))]
            acc, se = group_by_question_and_summarize(user_ai_direct, "User-AI, direct-to-AI")
            plot_results["User-AI, direct-to-AI"] = (acc, se)
            user_ai_answerFirst = subdf[(subdf.answer_type == "userAIAnswer") & (subdf.condition.str.startswith("answerFirst"))]
            acc, se = group_by_question_and_summarize(user_ai_answerFirst, "User-AI, answer-first")
            plot_results["User-AI, answer-first"] = (acc, se)
            all_plot_results[(ds, model)] = plot_results

            ax.bar(plot_results.keys(), [v[0] for v in plot_results.values()], 
                   yerr=[v[1] for v in plot_results.values()], capsize=3, color=color)
            if ds == datasets[0]:
                ax.set_title(model)
            if model == models[0]:
                ax.set_ylabel(ds.replace("_", " ").replace("mathematics", "math"), fontsize=12)
            if ds == datasets[-1]:
                ax.set_xticklabels(plot_results.keys(), rotation=90)
            else:
                ax.set_xticks([])
            ax.set_ylim(0.1, 1.05)   
            ax.grid(alpha=0.3)
    if plot_title is not None:
        plt.suptitle(plot_title, fontsize=14)
    plt.tight_layout()
    plt.show()
    return all_plot_results

def get_results_per_worker(checkpoints, onlyAI_events, answerFirst_events, id2batch, wids_to_check=None):
    """
    Iterate over workers, get their model/condition/subject, check if they have all the expected events,
    and check if they have feedback.
    """
    worker_info = []
    missing_workers = []
    expected_num_logs = 0
    missing_logs = []
    if wids_to_check is None:
        wids_to_check = checkpoints.worker_id.unique()

    for i, wid in enumerate(wids_to_check):
        if i % 100 == 0:
            print(f"Processing worker {i}/{len(wids_to_check)}")
        subdf = checkpoints[checkpoints.worker_id == wid].copy().sort_values("timestamp")
        subdf = subdf.sort_values("timestamp").reset_index(drop=True)
        if len(subdf) == 0:
            print(f"WARNING: {wid} has no logs")
            missing_workers.append(wid)
            continue
        wid_info = {"worker_id": wid}

        # get basic info: model, condition, subject, batch, num questions, num events
        subdf["batch"] = subdf.questionID.apply(lambda x: id2batch.get(x, np.nan))
        assign_cols = ["model", "condition", "subject", "batch"]
        assignments = subdf.dropna(subset=assign_cols)[assign_cols].drop_duplicates()
        if len(assignments) > 1:
            print(f"Warning: {wid} has {len(assignments)} assignments, keeping first")
        assignment = assignments.iloc[0]
        for k in assign_cols:
            wid_info[k] = assignment[k]
        qids = subdf[~subdf.questionID.isna()].questionID.unique()
        wid_info["num_questions"] = len(qids)
        events = subdf[~subdf.event.isna()].event.unique()
        wid_info["num_events"] = len(events)

        # check if worker passed attention check
        att_check = subdf[(subdf.questionID == "example-1") & (subdf.event.str.endswith("userAnswer_answer"))]
        if len(att_check) == 0:
            print(f"Warning: {wid} missing attention check")
        else:
            if len(att_check) > 1:
                print(f"Warning: {wid} has {len(att_check)} attention checks, keeping first")
            att_check = att_check.iloc[0]
            wid_info["passed_att_check"] = att_check.selectedAnswer == att_check.correctAnswer
        
        # check accuracy
        answer_subdf = subdf[subdf.event.str.endswith("Answer_answer")].copy()
        answer_subdf["acc"] = (answer_subdf["selectedAnswer"] == answer_subdf["correctAnswer"]).astype(int)
        question_correct = answer_subdf.groupby("questionID")["acc"].max()  # could be user-alone or user-AI correct
        n_correct = question_correct.sum()
        wid_info["num_correct"] = n_correct  # used to compute bonuses
        user_subdf = answer_subdf[(answer_subdf.event.str.endswith("userAnswer_answer")) & (answer_subdf.questionID != "example-1")]
        user_ai_subdf = answer_subdf[answer_subdf.event.str.endswith("userAIAnswer_answer")]
        wid_info["user_acc"] = user_subdf.acc.mean()
        wid_info["user_n"] = len(user_subdf)
        wid_info["user_ai_acc"] = user_ai_subdf.acc.mean()
        wid_info["user_ai_n"] = len(user_ai_subdf)
        
        # check chat and utterance lengths
        chats = subdf[subdf.event.str.endswith("userAIAnswer_chat")]
        chat_lengths = []
        utt_lengths = []
        first_prompt_too_short = []
        for _, row in chats.iterrows():
            chat_lengths.append(len(row["chatHistory"]))
            for i, c in enumerate(row["chatHistory"]):
                if c["role"] == "You":
                    utt_lengths.append(len(c["content"]))
                    if i == 0:
                        first_prompt_too_short.append(len(c["content"]) < 5)  # check if first prompt is short
        wid_info["avg_chat_length"] = np.mean(chat_lengths)
        wid_info["avg_utt_length"] = np.mean(utt_lengths)
        wid_info["prop_first_prompt_too_short"] = np.mean(first_prompt_too_short)

        # check for missing logs
        wid_info["missing_logs"] = []
        expected_events = onlyAI_events if wid_info["condition"].startswith("onlyAI") else answerFirst_events
        for e in expected_events:
            expected_num_logs += 1
            if e not in events:
                wid_info["missing_logs"].append(e)
                missing_logs.append((wid, e))  # all missing logs
        
        # get feedback
        feedback = subdf[subdf.event == "feedback_questions"]
        expected_num_logs += 1
        if len(feedback) == 0:
            print(f"Warning: {wid} missing feedback")
            wid_info["missing_logs"].append("feedback_questions")
            missing_logs.append((wid, "feedback_questions"))
        else:
            if len(feedback) > 1:
                print(f"Warning: {wid} has {len(feedback)} feedback logs, keeping first")
            answers = json.loads(feedback.iloc[0]["answers"])
            for k, v in answers.items():
                wid_info[k] = v

        worker_info.append(wid_info)
    
    worker_info = pd.DataFrame(worker_info).set_index("worker_id")
    print("\nFinished processing!")
    print(f"Expected {len(wids_to_check)} workers, found {len(worker_info)} ({100. * len(worker_info)/len(wids_to_check):.2f}%)")
    found_logs = expected_num_logs - len(missing_logs)
    print(f"Out of found workers: expected {expected_num_logs} logs, found {found_logs} ({100. * found_logs/expected_num_logs:.2f}%)")
    return worker_info, missing_workers, missing_logs

def get_timestamps_per_worker(checkpoints, worker_info, wids_to_keep=None):
    """
    Get timestamps and duration of tasks per worker.
    """
    kept_checkpoints = checkpoints[checkpoints.worker_id.isin(wids_to_keep)]
    worker2timestamps = {}
    for wid, subdf in kept_checkpoints.groupby("worker_id"):
        wid_data = {}
        for i in range(13):
            if i < 4:
                phase = "pretest"
                tasks = [f"{phase}_{i}_confidence", f"{phase}_{i}_userAnswer"]
            else:
                phase = "test"
                if worker_info.loc[wid]["condition"].startswith("answerFirst"):
                    tasks = [f"{phase}_{i}_confidence", f"{phase}_{i}_userAnswer", f"{phase}_{i}_userAIAnswer"]
                else:
                    tasks = [f"{phase}_{i}_confidence", f"{phase}_{i}_userAIAnswer"]

            for t, task in enumerate(tasks):
                if t == 0:  # first task for this question
                    start = f"{phase}_{i}_start"
                else:
                    start = f"{tasks[t-1]}_finish"
                end = f"{task}_finish"
                start_time = subdf[subdf["checkpoint"] == start]["timestamp"].values
                end_time = subdf[subdf["checkpoint"] == end]["timestamp"].values
                if len(start_time) == 1 and len(end_time) == 1:
                    start_time = datetime.strptime(start_time[0], "%Y-%m-%d %H:%M:%S")
                    end_time = datetime.strptime(end_time[0], "%Y-%m-%d %H:%M:%S")
                    wid_data[task] = (start_time, end_time, end_time - start_time)
                else:
                    print("missing", wid, task, len(start_time), len(end_time))
        worker2timestamps[wid] = wid_data
    print(f"Found timestamps for {len(worker2timestamps)} workers")
    return worker2timestamps


def plot_durations_of_tasks(worker2timestamps, worker_info, subject, outlier_minutes=10, 
                            color="tab:blue", title=None):
    """
    Plot durations of tasks.
    """
    labels = []
    counts = []
    means = []
    std = []
    math_details = ["elem", "HS", "col", "attention", "elem", "elem", "elem", "elem", "HS", "HS", "HS", "HS", "col"]
    assert len(math_details) == 13
    for i in range(13):
        if i < 4:
            phase = "pretest"
            tasks = [f"{phase}_{i}_confidence", f"{phase}_{i}_userAnswer"]
            if subject == "math":
                task_labels = [f"P1.{i+1} ({math_details[i]}), conf", f"P1.{i+1} ({math_details[i]}), user"]
            elif i == 3:
                task_labels = [f"P1.{i+1} (attention), conf", f"P1.{i+1} (attention), user"]
            else:
                task_labels = [f"P1.{i+1}, conf", f"P1.{i+1}, user"]
        else:
            phase = "test"
            tasks = [f"{phase}_{i}_confidence", f"{phase}_{i}_userAnswer", f"{phase}_{i}_userAIAnswer"]
            if subject == "math":
                task_labels = [f"P2.{i+1-4} ({math_details[i]}), conf", f"P2.{i+1-4} ({math_details[i]}), user", f"P2.{i+1-4} ({math_details[i]}), user-AI"]
            else:
                task_labels = [f"P2.{i+1-4}, conf", f"P2.{i+1-4}, user", f"P2.{i+1-4}, user-AI"]
        
        for t, l in zip(tasks, task_labels):
            labels.append(l)
            durations = []
            for wid, timestamps in worker2timestamps.items():
                if worker_info.loc[wid]["subject"] == subject and t in timestamps:
                    sec = timestamps[t][2].seconds
                    if sec < (60*outlier_minutes):
                        durations.append(sec)
                    else:
                        print(wid, t, f"outlier: took {sec/60:.3f} minutes")
            counts.append(len(durations))
            means.append(np.mean(durations))
            std.append(np.std(durations))
            print(t, np.mean(durations), np.std(durations))
    
    _, ax = plt.subplots(figsize=(10, 5))
    ses = [s/np.sqrt(n) for s, n in zip(std, counts)]
    ax.bar(labels, means, yerr=ses, capsize=5, color=color)
    ax.set_xticklabels(labels, rotation=90)
    ax.grid(alpha=0.3)
    ax.set_ylabel("Time (s)", fontsize=12)
    if title is not None:
        ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_accuracy_of_tasks(answers, subject, title=None):
    """
    Plot accuracy per position.
    """
    # get accuracy per position
    x = []
    errs = []
    labels = []
    for i in range(13):
        if i != 3:  # skip attention check
            if i < 4:
                phase = "pretest"
                tasks = [f"{phase}_{i}_userAnswer_answer"]
                task_labels = [f"P1.{i+1}, user"]
            else:
                phase = "test"
                tasks = [f"{phase}_{i}_userAnswer_answer", f"{phase}_{i}_userAIAnswer_answer"]
                task_labels = [f"P2.{i+1-4}, user", f"P2.{i+1-4}, user-AI"]
            for t, l in zip(tasks, task_labels):
                task_answers = answers[(answers["event"] == t) & (answers["subject"] == subject)]
                task_acc = task_answers.acc.mean()
                task_se = task_answers.acc.std() / np.sqrt(len(task_answers))
                print(f"{subject} {l} n={len(task_answers)} {task_acc:.2f} ± {task_se:.2f}")
                labels.append(l)
                x.append(task_acc)
                errs.append(task_se)

    _, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, x, yerr=errs, color="tab:blue", capsize=3)
    ax.set_ylabel("Accuracy", fontsize=12)
    if title is not None:
        ax.set_title(title, fontsize=14)
    ax.grid(alpha=0.3)
    ax.set_xticklabels(labels, rotation=90, fontsize=11)
    plt.show()

def make_plot_for_question(q_acc, ax=None):
    """
    Make plot per question.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(3.5,3))
    user_alone = q_acc[q_acc.answer_type == "userAnswer_answer"].iloc[0]
    ax.errorbar(0, user_alone["mean"], yerr=user_alone["se"], fmt="o", color="black")
    model_let = q_acc.iloc[0]["vanilla_zero_shot_mean"]
    se = (model_let * (1 - model_let) / q_acc.iloc[0]["vanilla_zero_shot_count"])**0.5
    ax.errorbar(1, model_let, yerr=se, fmt="o", color="black")
    model_cp = q_acc.iloc[0]["copy_and_paste_mean"]
    se = (model_cp * (1 - model_cp) / q_acc.iloc[0]["copy_and_paste_count"])**0.5
    ax.errorbar(2, model_cp, yerr=se, fmt="o", color="black")
    model_sim = q_acc.iloc[0]["conversations_mean"]
    se = (model_sim * (1 - model_sim) / q_acc.iloc[0]["conversations_count"])**0.5
    ax.errorbar(3, model_sim, yerr=se, fmt="o", color="black")
    user_ai = q_acc[q_acc.answer_type == "userAIAnswer_answer"].iloc[0]
    ax.errorbar(4, user_ai["mean"], yerr=user_ai["se"], fmt="o", color="black")

    ax.set_xticks(range(5), ["user-alone", "AI-alone, letter", "AI-alone, free", "AI-sim", "user-AI"], rotation=45)
    ax.tick_params(labelsize=12)
    # ax.set_ylabel("Accuracy", fontsize=12)
    # ds = "EM" if q.startswith("elementary_mathematics") else "HS"
    ax.set_title(q.split("-", 1)[-1], fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.2)


########################################################################
# Functions to analyze user-AI conversations.
########################################################################
def print_conversations_for_dataset(conversations, ds, ai_alone_results=None, dir="./"):
    """
    Print user-AI conversations for a dataset, sorted by questionID then model.
    For each question x model, compare user-AI acc to AI-alone acc.
    """
    ds_conversations = conversations[conversations.ds == ds]
    print(ds, len(ds_conversations), "conversations")
    with open(os.path.join(dir, f"{ds}_conversations.txt"), "w", encoding="utf-8") as f:
        qid_sorted = sorted(ds_conversations.questionID.unique(), key=lambda x: int(x.split("-")[-1]))
        for qid in qid_sorted:
            df_q = ds_conversations[ds_conversations.questionID == qid]
            assert (df_q[["questionText", "correctAnswer"]].nunique() == 1).all()
            row = df_q.iloc[0]
            f.write(f"============================ {qid} ============================\n")
            f.write(f"{row['questionText']}\n")
            for choice, label in zip(row["choices"], row["choiceLabels"]):
                f.write(f"{label}. {choice}\n")
            ans = row.choices[row.choiceLabels.index(row.correctAnswer)]
            f.write(f"CORRECT ANSWER: {row.correctAnswer}. {ans}\n\n")
            for m, df_qm in df_q.groupby("model"):
                user_ai_acc = df_qm.acc.mean()
                if ai_alone_results is not None:
                    ai_alone_row = ai_alone_results[(ai_alone_results.questionID == qid) & (ai_alone_results.model == m)]
                    assert len(ai_alone_row) <= 1
                    if len(ai_alone_row) == 1:
                        ai_alone_row = ai_alone_row.iloc[0]
                        if user_ai_acc > ai_alone_row.copy_and_paste_mean:
                            status = "user-AI > free-text"
                        elif user_ai_acc < ai_alone_row.copy_and_paste_mean:
                            status = "user-AI < free-text"
                        else:
                            status = "user-AI = free-text"
                        f.write(f"{m}: AI-alone letter-only={ai_alone_row.letter_only_few_shot_mean:.3f}, free-text={ai_alone_row.copy_and_paste_mean:.3f}; user-AI={df_qm.acc.mean():.3f} (n={len(df_qm)}) -> {status}\n")
                    else:
                        f.write(f"{m}: AI-alone in progress, user-AI={df_qm.acc.mean():.3f} (n={len(df_qm)})\n")
                else:
                    f.write(f"{m}: user-AI={df_qm.acc.mean():.3f} (n={len(df_qm)})\n")
                df_qm = df_qm.sort_values("condition")
                for _, row in df_qm.iterrows():
                    f.write(f"{row.worker_id}, condition={row.condition}, correct={row.acc == 1}\n")
                    chat_history = row.chatHistory
                    for chat in chat_history:
                        role = "User" if chat["role"] == "You" else "System"
                        f.write(f"{role.upper()}: {chat['content']}\n")
                    ans = row.choices[row.choiceLabels.index(row.selectedAnswer)]
                    f.write(f"SELECTED ANSWER: {row.selectedAnswer}. {ans}\n")               
                    f.write("\n")

def convert_conversation_to_simulator_data(row, question_df):
    """
    Convert a user-AI conversation into data for the two-step simulator.
    """
    data = []
    conv = row["chatHistory"]
    question_row = question_df.loc[row["questionID"]]
    
    # first prompt data 
    assert conv[0]["role"] == "You", "first message from Bot"    
    sys_prompt, user_prompt = two_step_simulator(question_row, None, [], return_prompts=True)
    real_prompt = conv[0]["content"]
    data.append({"messages": [{"role": "system", "content": sys_prompt}, 
                 {"role": "user", "content": user_prompt},
                 {"role": "assistant", "content": real_prompt}]})
    
    # later prompt data
    for i in range(1, len(conv)):
        if conv[i]["role"] == "Bot":
            sys_prompt, user_prompt = two_step_simulator(question_row, None, conv[:i+1], return_prompts=True)
            if i == (len(conv)-1):  # this is the last message, user selected answer after this message
                real_prompt = "Answer: " + row["selectedAnswer"]
            else:
                assert conv[i+1]["role"] == "You", "consecutive messages from Bot"
                real_prompt = conv[i+1]["content"]  # next human message 
            data.append({"messages": [{"role": "system", "content": sys_prompt}, 
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": real_prompt}]})
    return data

def extract_answer_from_conversation(row, return_prompts=False):
    """
    Extract an answer to the question from the conversation. Logic is similar to 
    extract_answer_from_response for automated copy_and_paste method.
    """
    system_prompt = "You are a helpful assistant."
    user_prompt = f"Here is a question that someone was asked:\n\n{row['questionText']}\n"
    for lbl, choice in zip(row["choiceLabels"], row["choices"]):
        user_prompt += f"{lbl}. {choice}\n"
    user_prompt += "\nHere is a conversation that the person had with an AI system about the question:\n"
    for chat in row["conversation"]:
        if chat["role"] == "You":
            user_prompt += f"Person: {chat['content']}\n"
        else:
            user_prompt += f"System: {chat['content']}\n"
    user_prompt += f"\nDoes this conversation provide a final answer to the question? Respond with a JSON object that contains one key \"attempted_answer\" with a value that is true or false. "
    user_prompt += "If \"attempted_answer\" is true, then include a second key \"answer_val\" with the final answer's value in quotations. "
    user_prompt += f"If the final answer value matches one of the answer options, include a third key \"answer_letter\" which is the letter corresponding to the matching answer option."
    if return_prompts:
        return system_prompt, user_prompt
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    extracted = query_model("gpt-4o", messages, kwargs={"temperature": 0.})
    if "{" in extracted and "}" in extracted:
        # isolate content within json, drop surrounding text
        extracted = extracted.split("{", 1)[1]
        extracted = extracted.rsplit("}", 1)[0]
    try:
        extracted = json.loads("{" + extracted + "}")
        return extracted, True 
    except Exception as e:
        print(f"Failed to convert to json: {str(e)}\nExtracted: {extracted}")
        return extracted, False  

def check_consistency_in_conversation_and_selection(conversations):
    """ 
    Go through conversations and check if user's selected answer is consistent with
    user-AI conversation.
    """
    convs_to_check = []
    for conv in conversations:
        consistent = False 
        resp, success = extract_answer_from_conversation(conv)
        if success:
            if "answer_letter" in resp:
                if resp["answer_letter"] == conv["selectedAnswer"]:
                    print(conv["worker_id"], conv["questionID"], "consistent", flush=True)
                    consistent = True 
                else:
                    print(conv["worker_id"], conv["questionID"], f"inconsistent", flush=True)
            else:
                print(conv["worker_id"], conv["questionID"], "no answer letter in response", flush=True)
        else:
            print(conv["worker_id"], conv["questionID"], "unsuccessful json parsing", flush=True)
        if not consistent:
            convs_to_check.append(conv)
    print("Number of conversations to check:", len(convs_to_check), flush=True)
    return convs_to_check

def classify_chatbot_mistakes_in_feedback():
    results_dir = "../prolific_results/10_percent_pilot_20250204"
    with open(os.path.join(results_dir, "feedback.json"), "r") as f:
        feedback = json.load(f)
    updated_feedback = []
    for d in feedback:
        s = d["chatbot_mistakes"]
        user_prompt = f"A user answered the question, \"Did you notice any mistakes in the chatbot's responses?\" with the following answer:\n\"{s}\"\nClassify the answer as \"Yes\" or \"No\" for whether the user noticed a mistake. Output ONLY \"Yes\" or \"No\"."
        messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": user_prompt}]    
        resp = query_model("gpt-4o", messages, kwargs={"temperature": 0.})
        print(s, "->", resp, flush=True)
        d["chatbot_mistakes_binary"] = resp
        updated_feedback.append(d)
    with open(os.path.join(results_dir, "feedback_updated.json"), "w") as f:
        json.dump(updated_feedback, f)

def classify_if_user_corrected_model(row, return_prompts):
    """
    Classify if user corrected the model from conversation.
    """
    system_prompt = "You are a helpful assistant."
    user_prompt = f"Here is a question that someone was asked:\n\n{row['questionText']}\n"
    for lbl, choice in zip(row["choiceLabels"], row["choices"]):
        user_prompt += f"{lbl}. {choice}\n"
    user_prompt += "\nHere is a conversation that the person had with an AI system about the question:\n"
    for chat in row["chatHistory"]:
        if chat["role"] == "You":
            user_prompt += f"Person: {chat['content']}\n"
        else:
            user_prompt += f"System: {chat['content']}\n"
    user_prompt += f"\nDoes the user correct the AI system at any point in this conversation? Respond with a JSON object that contains one key \"corrected\" with a value that is true or false. "
    user_prompt += "If \"corrected\" is true, then include a second key \"corrections\" which is a list of the user's corrections, with exact quotes of the user."
    if return_prompts:
        return system_prompt, user_prompt
    
    if len(row["chatHistory"]) <= 2:
        return {"corrected": False}, True  # no corrections possible
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    extracted = query_model("gpt-4o", messages, kwargs={"temperature": 0.})
    if "{" in extracted and "}" in extracted:
        # isolate content within json, drop surrounding text
        extracted = extracted.split("{", 1)[1]
        extracted = extracted.rsplit("}", 1)[0]
    try:
        extracted = json.loads("{" + extracted + "}")
        return extracted, True 
    except Exception as e:
        print(f"Failed to convert to json: {str(e)}\nExtracted: {extracted}")
        return extracted, False  

def classify_corrections_in_conversations(results_dir, hit, ds):
    """
    Classify all conversations from this HIT for the dataset ds.
    """
    df = pd.read_csv(os.path.join(results_dir, f"from_azure_{hit}.csv"))
    print(len(df), df.worker_id.nunique(), flush=True)
    checkpoints = get_checkpoints(df)
    conversations = get_conversations_from_checkpoints(checkpoints)

    conversations = conversations[conversations.ds == ds]
    print(ds, len(conversations), "conversations", flush=True)
    corrections_data = []
    for _, row in conversations.iterrows():
        corrections, success = classify_if_user_corrected_model(row, return_prompts=False)
        if success:
            print(row["worker_id"], row["questionID"], corrections["corrected"], flush=True)
            corrections["worker_id"] = row["worker_id"]
            corrections["questionID"] = row["questionID"]
            corrections_data.append(corrections)
        else:
            print("Failed to classify corrections for", row["worker_id"], row["questionID"], flush=True)
            corrections_data.append({"worker_id": row["worker_id"], "questionID": row["questionID"], "corrected": None})
    with open(os.path.join(results_dir, f"{ds}_corrections_data.json"), "w") as f:
        json.dump(corrections_data, f)

def write_conversations_with_corrections(results_dir, hit):
    """
    Write all conversations from a HIT that had corrections.
    """
    df = pd.read_csv(os.path.join(results_dir, f"from_azure_{hit}.csv"))
    print(len(df), df.worker_id.nunique(), flush=True)
    checkpoints = get_checkpoints(df)
    conversations = get_conversations_from_checkpoints(checkpoints)
    with open(os.path.join(results_dir, "conversations_with_corrections.txt"), "w", encoding="utf-8") as f:
        for ds in DATASETS:
            fn = os.path.join(results_dir, f"{ds}_corrections_data.json")
            if os.path.isfile(fn):
                with open(fn, "r") as corrections_f:
                    corrections = json.load(corrections_f)
                num_corrected = np.sum([int(c["corrected"]) for c in corrections])
                f.write(f"============================= {ds} =============================\n")
                f.write(f"{num_corrected}/{len(corrections)} of conversations have corrections\n\n")
                for c in corrections:
                    if c["corrected"]:
                        f.write("-----------------------------\n")
                        row = conversations[(conversations.worker_id == c["worker_id"]) & (conversations.questionID == c["questionID"])]
                        assert len(row) == 1
                        row = row.iloc[0]
                        f.write(f"{row['worker_id']}, {row['questionID']}, {row['model']}, {row['condition']}\n")
                        f.write(f"{row['questionText']}\n")
                        for choice, label in zip(row["choices"], row["choiceLabels"]):
                            f.write(f"{label}. {choice}\n")
                        ans = row.choices[row.choiceLabels.index(row.correctAnswer)]
                        f.write(f"CORRECT ANSWER: {row.correctAnswer}. {ans}\n")
                        ans = row.choices[row.choiceLabels.index(row.selectedAnswer)]
                        f.write(f"SELECTED ANSWER: {row.selectedAnswer}. {ans} (correct = {row.correctAnswer == row.selectedAnswer})\n\n")
                        for c in row.chatHistory:
                            role = "User" if c["role"] == "You" else "System"
                            f.write(f"{role.upper()}: {c['content']}\n")
                        f.write("\n")
