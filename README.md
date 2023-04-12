# Truthful QA Experiments

## Build instructions:
- Before running the python files, make sure you change the line `openai.api_key = "add-your-api-key-here"` to your OpenAI key.
- To run either of the python files, use `python file.py input_csv_arg output_csv_arg` where arg is the csv file of the TruthfulQA dataset

## API Settings
- For a list of available models, refer to `openai_models.json`
- A temperature of 0 is used and max tokens of 50 is set. All other parameters are default values.
- For model retried answer for chat models, the max token is increased to 60 as the model deterministically replies "I apologize for the confusion".

## Non-chat models:
- Use `completions_nonchat.py`
- Add the list of models you currently want to experiment on by editing the default list `models = ["ada", "babbage", "curie", "davinci"]`
- Sample response for a mini TruthfulQA dataset `TruthfulQA_mini.csv` (has 10 questions) can be found in `nonchat_completion_outputs.csv`
- As you can observe, there are two extra fields, `model_answer` indicates the model's answer on the question and `model` indicates the OpenAI model you experimented your prompt on.


## Chat models:
- Use `completions_chat.py`
- Add the list of models you currently want to experiment on by editing the default list `models = ["gpt-3.5-turbo"]`
- Sample response for a mini TruthfulQA dataset `TruthfulQA_mini.csv` (has 10 questions) can be found in `chat_completion_outputs.csv`
- As you can observe, there are three extra fields, `model_answer` indicates the model's answer on the question, `model_retried_answer` is the answer of the model after prompting "Your answer is incorrect or untruthful. Can you retry answering?" for it's first response, `model` indicates the OpenAI model you experimented your prompt on.

## Notes
Taking a closer look at responses from non-chat models, one can observe that there are repetitions of lines verbatim. This can be controlled by adjusting frequency and presence penalties.

