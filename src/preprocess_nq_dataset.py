import json
import pandas as pd
import sys
import re
import csv

def remove_html_tags(sentence):
    sentence = re.sub('<[^<]+?>', '', sentence)
    sentence = re.sub(r'[^\w\s]', '', sentence)
    sentence = re.sub(' +', ' ', sentence)
    return sentence[:25000]

def preprocess_nq_data(nq_data_path, output_path):
    count = 0
    fieldnames = ['question', 'answer1', 'answer2', 'answer3', 'answer4', 'answer5', 'url']
    with open(nq_data_path, 'r') as f:
        with open(output_path, 'w', encoding='utf-8', newline='') as output_csv_file:
            writer = csv.DictWriter(output_csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for line in f:
                count +=1
                example = json.loads(line)
                question = example['question_text']
                answers = []
                urls = example['document_url']
                for annotation in example['annotations']:
                    if 'long_answer' in annotation:
                        long_answer = annotation['long_answer']
                        if long_answer['start_byte'] != -1:
                            answer = ' '.join([tok['token'] for tok in example['document_tokens'][long_answer['start_token']:long_answer['end_token']]])
                            answers.append(remove_html_tags(answer))
                            continue
                    if 'short_answers' in annotation and len(annotation['short_answers']) > 0:
                        short_answer = annotation['short_answers'][0]
                        answer = ' '.join([tok['token'] for tok in example['document_tokens'][short_answer['start_token']:short_answer['end_token']]])
                        answers.append(remove_html_tags(answer))
                        continue
                    if 'yes_no_answer' in annotation:
                        answers.append(remove_html_tags(annotation['yes_no_answer']))
                
                if len(answers) > 0:
                    row = {'question': question, 'answer1': answers[0] if len(answers) > 0 else 'NA', 'answer2': answers[1] if len(answers) > 1 else 'NA', 'answer3': answers[2] if len(answers) > 2 else 'NA', 'answer4': answers[3] if len(answers) > 3 else 'NA', 'answer5': answers[4] if len(answers) > 4 else 'NA', 'url': urls}
                    writer.writerow(row)


preprocess_nq_data(sys.argv[1], sys.argv[2])