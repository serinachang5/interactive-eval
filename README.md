Code for the paper "ChatBench: From Static Benchmarks to Human-AI Evaluation" (under review). Contents include:
- ``get_mturk_azure_results_by_hit.py``: function to pull raw data from our Azure database, where data from our user studies are logged.
- ``analyze_results.py``: code to process raw user study data and analyze results.
- ``make_clean_data.py``: code to make a clean version of answers for statistical analyses, following the filtering criteria defined in our [pre-registration](https://aspredicted.org/n84n-sn3f.pdf).
- ``qa_reasoning.py``: implementation of AI-alone methods; code to run experiments over all questions.
- ``generate_conversations.py``: implementation of user simulators.
- ``constants_and_utils.py``: constants and functions to query models (removed for anonymity) and load MMLU / MMLU-Redux datasets.
