import pandas as pd
import numpy as np
import json
import os
import argparse
from analyze_results import *

def get_clean_answers(checkpoints, hit, exp_dir, match_orig=False, multi_assign_filter="strong"):
    """
    Function to get clean user answers for main statistical results.
    """
    # get answers
    answers = get_answers_from_checkpoints(checkpoints, drop_att_check=False).sort_values("timestamp")
    print(answers.groupby(["ds", "model", "answer_type"]).size())

    if match_orig:
        orig_len = len(answers)
        # drop workers with fewer than 5 questions
        counts = answers.groupby("worker_id")["questionID"].nunique()
        workers_to_keep = counts[counts >= 5].index
        print(len(counts), len(workers_to_keep))
        answers = answers[answers.worker_id.isin(workers_to_keep)]
        print("Num answers after filtering out workers with fewer than 5 questions:", len(answers))
    
    else:
        # drop attention check from answers
        att_check = answers[answers.questionID == "example-1"]
        answers = answers[answers.questionID != "example-1"]
        print("Num answers after dropping attention check:", len(answers))
        orig_len = len(answers)

        # filter answers by workers who completed study
        workers = pd.read_csv(os.path.join(exp_dir, f"prolific_export_{hit}.csv"))
        completed_wids = workers[workers.Status.isin(["AWAITING REVIEW", "APPROVED"])]["Participant id"].unique()        
        answers = answers[answers.worker_id.isin(completed_wids)]
        print(f"Found {len(answers.worker_id.unique())} out of {len(completed_wids)} workers in database")
        print("Num answers after only keeping workers who completed study:", len(answers))

        # only keep workers who passed attention check
        att_check_acc = att_check.groupby("worker_id")["acc"].mean()
        passed_att_check = att_check_acc[att_check_acc == 1.0].index
        prev_num_workers = len(answers.worker_id.unique())
        answers = answers[answers.worker_id.isin(passed_att_check)]
        print(f"Num workers who passed attention check: {len(answers.worker_id.unique())} ({100. * len(answers.worker_id.unique()) / prev_num_workers:.1f}%)")
        print("Num answers after only keeping workers who passed attention check:", len(answers))
        
        # add "batch" column to answers
        id2batch = get_question_id_to_batch_mapping()
        answers["batch"] = answers["questionID"].map(id2batch)
        assignment_cols = ["subject", "batch", "model", "condition"]
        for col in assignment_cols:
            num_na = answers[col].isna().sum()
            if num_na > 0:
                print("Warning: found", num_na, "NAs in", col)

        # check for multiple assignments per worker
        num_multiple = 0
        num_multiple_keep = 0 
        grouped_by_worker = []
        for worker_id, w_df in answers.groupby("worker_id"):
            unique_assignments = w_df[w_df.questionID != "example-1"].drop_duplicates(assignment_cols, keep="first")
            if len(unique_assignments) > 1:
                num_multiple += 1
                # get answers from first assignment
                first_assignment = unique_assignments.iloc[0]
                first_answers = w_df[(w_df.subject == first_assignment["subject"]) & (w_df.batch == first_assignment["batch"]) & (w_df.model == first_assignment["model"]) & (w_df.condition == first_assignment["condition"])]
                if multi_assign_filter == "weak" and (first_answers.answer_type == "userAnswer").all():
                    # keep answers from second assignment if first didn't reach user-AI
                    second_assignment = unique_assignments.iloc[1]
                    second_answers = w_df[(w_df.subject == second_assignment["subject"]) & (w_df.batch == second_assignment["batch"]) & (w_df.model == second_assignment["model"]) & (w_df.condition == second_assignment["condition"])]
                    w_df = pd.concat([first_answers, second_answers])
                    num_multiple_keep += 1
                else:
                    w_df = first_answers
            grouped_by_worker.append(w_df)
        answers = pd.concat(grouped_by_worker)
        print(f"Num workers with multiple assignments: {num_multiple} ({100. * num_multiple / len(answers.worker_id.unique()):.1f}%)")
        if multi_assign_filter == "weak":
            print(f"Num workers where we kept answers from first two assignments since worker didn't reach user-AI in first: {num_multiple_keep} ({100. * num_multiple_keep / len(answers.worker_id.unique()):.1f}%)")
        print(f"Num answers after filtering on multiple assignments: {len(answers)}")

        # deduplicate answers for same worker, question, and answer_type
        dupes = answers.groupby(["worker_id", "questionID", "answer_type"]).size()
        dupes = dupes[dupes > 1]
        print("Num (worker, question, answer_type) tuples with multiple answers:", len(dupes))
        for (wid, qid, at), _ in dupes.items():
            rows = answers[(answers.worker_id == wid) & (answers.questionID == qid) & (answers.answer_type == at)]
            assert rows.selectedAnswer.nunique() == 1, f"Found differing answers for worker {wid}, question {qid}, answer type {at}"
        answers = answers.drop_duplicates(["worker_id", "questionID", "answer_type"], keep="first")
        print(f"Num answers after deduplicating: {len(answers)}")
    
    print(f"Final num answers: {len(answers)} ({(len(answers) / orig_len) * 100:.1f}% of original)")
    print(answers.groupby(["ds", "model", "answer_type"]).size())
    return answers
    

if __name__ == "__main__":
    # get hit id from first (unnamed) command line argument using argparse
    # and get pretty hit name from second (unnamed) command line argument
    parser = argparse.ArgumentParser()
    parser.add_argument("hit_id", type=str, help="hit id")
    parser.add_argument("readable_name", type=str, help="readable version of the hit name")
    parser.add_argument("--match_orig", action="store_true", help="match earlier version of make_clean_data")
    parser.add_argument("--filter", type=str, default="strong", choices=["strong", "weak"], help="whether to apply a strong filter (only keep answers from first assignment) or weak filter (keep answers from second assignment if first didn't reach user-AI)")
    args = parser.parse_args()

    # extract hit id and readable name from arguments
    hit = args.hit_id
    readable_name = args.readable_name
    
    # set up directories and filenames
    exp_dir = os.path.join("..", "prolific_results", readable_name)
    out_file = os.path.join(exp_dir, f"clean_user_answers.csv")

    # load data from azure
    df = pd.read_csv(os.path.join(f"{exp_dir}/from_azure_{hit}.csv"))
    print(len(df))
    print(df.worker_id.nunique())
    print(df.head())

    # get checkpoints
    cutoff = 5 if args.match_orig else 1
    checkpoints = get_checkpoints(df, cutoff=cutoff)
    print(checkpoints.head())
    
    # get clean answers
    answers = get_clean_answers(checkpoints, hit, exp_dir, match_orig=args.match_orig, 
                                multi_assign_filter=args.filter)

    # get confidence
    confidence = get_confidence_from_checkpoints(checkpoints)
    key_cols = ["worker_id", "questionID", "question_position", "subject", "model", "condition"]  # cols to match on
    # deduplicate confidence data
    dupes = confidence.groupby(key_cols).size()
    dupes = dupes[dupes > 1]
    print(f"Num {key_cols} tuples with multiple confidences:", len(dupes))
    for key, _ in dupes.items():
        rows = confidence[confidence[key_cols].eq(key).all(axis=1)]
        assert rows.selectedAnswer.nunique() == 1, f"Found differing confidences for {key}"
    confidence = confidence.drop_duplicates(key_cols, keep="first")
    print(f"Num confidences after deduplicating: {len(confidence)}")
    confidence = confidence[key_cols + ["selectedAnswer"]].rename(columns={"selectedAnswer": "confidence"})

    # merge answers with confidence
    merged = answers.merge(confidence, how="left", on=key_cols)
    assert len(merged) == len(answers)
    print(f"Found confidence for {100 * merged.confidence.notna().sum()/len(merged):.1f}% of answers")

    # save user data
    data = merged.copy()
    data["correct"] = data.acc.astype(int)
    data["userAI"] = (data["answer_type"] == "userAIAnswer").astype(int) 
    data["userAI_answerFirst"] = (data["answer_type"] == "userAIAnswer").astype(int) * data["condition"].str.startswith("answerFirst").astype(int)
    data["userAI_allowCopy"] = (data["answer_type"] == "userAIAnswer").astype(int) * data["condition"].str.endswith("allowCopy").astype(int)
    data["phase"] = data.event.apply(lambda x: 1 if x.startswith("pretest") else 2)
    print(data[["userAI", "userAI_answerFirst", "userAI_allowCopy", "correct"]].sum())
    print(data.groupby("phase").size())
    print(data.groupby("question_position").size()) 
    # data used in R scripts
    data[["correct", "userAI", "userAI_answerFirst", "userAI_allowCopy", "phase", "question_position", "worker_id", "questionID", "model", "confidence"]].to_csv(f"{exp_dir}/clean_user_answers.csv", index=False)