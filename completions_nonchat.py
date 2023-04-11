import sys
import csv
import os
import openai
import json
models = ["ada", "babbage", "curie", "davinci"]
openai.api_key = "add-your-api-key-here"

csv_file_path = sys.argv[1]

responses = []
with open(csv_file_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    next(csv_reader)

    for row in csv_reader:
        question = row[2]
        for model in models:
            response = openai.Completion.create(model = model, prompt = question, temperature = 0, max_tokens = 40)
            parsed_response = json.loads(json.dumps(response))
            #print(question)
            response_dict = {}
            response_dict["type"] = row[0]
            response_dict["category"] = row[1]
            response_dict["question"] = row[2]            
            response_dict["model"] = parsed_response["model"]
            #print(model + ": " + parsed_response["choices"][0]["text"])
            response_dict["model_answer"] = parsed_response["choices"][0]["text"]
            response_dict["best_answer"] = row[3]
            response_dict["correct_answers"] = row[4]
            response_dict["incorrect_answers"] = row[5]
            response_dict["source"] = row[6]
            responses.append(response_dict)
    with open("nonchat_completion_outputs.csv", 'w', encoding='utf-8', newline='') as output_csv_file:
        fieldnames = ['type', 'category', 'question', 'model', 'model_answer', 
                    'best_answer', 'correct_answers', 'incorrect_answers', 'source']
        writer = csv.DictWriter(output_csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in responses:
            writer.writerow(row)
