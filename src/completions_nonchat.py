import sys
import csv
import os
import openai
import json
models = ["davinci"]
fieldnames = ['type', 'category', 'question', 'prompt', 'model', 'model_answer', 'logprobs',
            'best_answer', 'correct_answers', 'incorrect_answers', 'source']
openai.api_key = "add-your-api-key-here"
csv_file_path = sys.argv[1]

with open(csv_file_path) as csv_file:
    with open(sys.argv[2], 'w', encoding='utf-8', newline='') as output_csv_file:
        writer = csv.DictWriter(output_csv_file, fieldnames=fieldnames)
        writer.writeheader()
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for i, row in enumerate(csv_reader):
            question = row[2]
            for model in models:
                question = "Q: Where were the 1992 Olympics held? A: The 1992 Olympics were held in Barcelona, Spain.\n\nQ: " + question
                parsed_response = openai.Completion.create(model = model, prompt = question, temperature = 0, max_tokens = 50, logprobs = 1, stop = "\n\n")
                print("Question " + str(i))
                response_dict = {}
                response_dict["type"] = row[0]
                response_dict["category"] = row[1]
                response_dict["question"] = row[2]            
                response_dict["prompt"] = question            
                response_dict["model"] = parsed_response["model"]
                print(model + ": " + parsed_response["choices"][0]["text"])
                response_dict["model_answer"] = parsed_response["choices"][0]["text"]
                response_dict["logprobs"] = json.dumps(parsed_response["choices"][0]["logprobs"]) # dict itself, has tokens, token_logprobs, top_logprobs and text_offset
                response_dict["best_answer"] = row[3]
                response_dict["correct_answers"] = row[4]
                response_dict["incorrect_answers"] = row[5]
                response_dict["source"] = row[6]
                writer.writerow(response_dict)
