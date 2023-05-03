import pandas as pd
import spacy
import lftk
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")
searched_features = lftk.search_features(return_format = "list_key")

output_paths = ["../outputs/openbookqa_davinci_labelled.csv", "../outputs/ada_output_labeled.csv", "../outputs/babbage_output_labeled.csv", "../outputs/curie_output_labeled.csv", "../outputs/davinci_output_labeled.csv"]

for output_path in output_paths:
    df = pd.read_csv(output_path)
    rows = df.to_dict('records')

    features_label = []
    for row in tqdm(rows):
        try:
            doc = nlp(row['model_answer'])
            LFTK = lftk.Extractor(docs = doc)
            extracted_features = LFTK.extract(features = searched_features)
            
            new_row = {**extracted_features}
            if output_path != "../outputs/openbookqa_davinci_labelled.csv":
                new_row['truthfulness'] = row['pred']
            else:
                new_row['truthfulness'] = row['label']
            features_label.append(new_row)
        except:
            print(row['model_answer'])
    
    df = pd.DataFrame(features_label)
    df.to_csv(f"{output_path[:-4]}_features.csv", index=False)
    df_corr = df.corr(method='pearson')
    df_corr_pred = df_corr.truthfulness.sort_values(ascending=False)
    df_corr_pred.to_csv(f"{output_path[:-4]}_corr.csv")