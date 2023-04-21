import sys
import csv
import openai
import time
from tqdm import tqdm
model_name = "davinci"
fieldnames = ['question', 'document_url', 'answer1', 'answer2', 'answer3', 'answer4', 'answer5', 'model', 'model_answer']
openai.api_key = "add-your-api-key-here"
csv_file_path = sys.argv[1]
count = 0
with open(csv_file_path) as csv_file:
    with open(sys.argv[2], 'w', encoding='utf-8', newline='') as output_csv_file:
        writer = csv.DictWriter(output_csv_file, fieldnames=fieldnames)
        writer.writeheader()
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)
        for row in tqdm(csv_reader):
            count += 1
            if count < 7048:
                continue
            question = row[0]
            while True:
                try:
                    question = "Q: Where were the 1992 Olympics held? A: The 1992 Olympics were held in Barcelona, Spain.\n\nQ: " + question
                    if not question.endswith('?'):
                        question += '?'
                    parsed_response = openai.Completion.create(model = model_name, prompt = question, temperature = 0, max_tokens = 50, logprobs = 1, stop = "\n\n")
                    # print("Question " + str(i))
                    response_dict = {}
                    response_dict["question"] = row[0]
                    response_dict["document_url"] = row[6]
                    response_dict["answer1"] = row[1]
                    response_dict["answer2"] = row[2]    
                    response_dict["answer3"] = row[3]    
                    response_dict["answer4"] = row[4]    
                    response_dict["answer5"] = row[5]                            
                    response_dict["model"] = parsed_response["model"]
                    # print(model_name + ": " + parsed_response["choices"][0]["text"])
                    response_dict["model_answer"] = parsed_response["choices"][0]["text"]
                    writer.writerow(response_dict)
                    break
                except Exception as e:
                    print(e)
                    time.sleep(5)
