import json
import sys
import openai
import csv
openai.api_key = "add-your-api-key-here"
model_used = 'davinci'
def preprocess_nq_data(nq_data_path, output_path):
    fieldnames = ['question', 'choice_a', 'choice_b', 'choice_c', 'choice_d', 'answer', 'model', 'model_answer']
    with open(nq_data_path, 'r') as f:
        with open(output_path, 'w', encoding='utf-8', newline='') as output_csv_file:
            writer = csv.DictWriter(output_csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for line in f:
                example = json.loads(line)
                question = example['question']['stem']
                qprompt = "Q: What doesn't eliminate waste? A: robots. Q: When food is reduced in the stomach A: nutrients are being deconstructed. Q: " + question

                choice_a, choice_b, choice_c, choice_d = "", "", "", ""
                for choice in example["question"]["choices"]:
                    if choice["label"] == "A":
                        choice_a = choice["text"]
                    elif choice["label"] == "B":
                        choice_b = choice["text"]
                    elif choice["label"] == "C":
                        choice_c = choice["text"]                    
                    elif choice["label"] == "D":
                        choice_d = choice["text"]
                answer = example["answerKey"]
                parsed_response = openai.Completion.create(model = model_used, prompt = qprompt, temperature = 0, max_tokens = 50, stop = ".")                
                print("Q: " + question)
                print(parsed_response["choices"][0]["text"])
                row = {'question': question, "choice_a": choice_a, "choice_b": choice_b, "choice_c": choice_c, "choice_d": choice_d,
                        "answer": answer, "model": parsed_response["model"], "model_answer":  parsed_response["choices"][0]["text"]}
                writer.writerow(row)
preprocess_nq_data(sys.argv[1], sys.argv[2])