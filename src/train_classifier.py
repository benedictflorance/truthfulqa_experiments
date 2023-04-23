from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import svm
from sklearn import metrics
from sklearn.inspection import permutation_importance
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

output_paths = ["../outputs/ada_output_labeled_features.csv", "../outputs/babbage_output_labeled_features.csv", "../outputs/curie_output_labeled_features.csv", "../outputs/davinci_output_labeled_features.csv"]





df_ada = pd.read_csv("../outputs/ada_output_labeled_features.csv")
y_ada = df_ada['truthfulness']
X_ada = df_ada.drop(labels="truthfulness", axis=1)
X_train_ada, X_test_ada, y_train_ada, y_test_ada = train_test_split(X_ada, y_ada, test_size=0.2,random_state=2023)

clf_ada = svm.SVC(kernel='rbf')
clf_ada.fit(X_train_ada, y_train_ada)

y_test_pred_ada = clf_ada.predict(X_test_ada)
print("Accuracy-train-ada-test-ada:",metrics.accuracy_score(y_test_ada, y_test_pred_ada))


df_babbage = pd.read_csv("../outputs/babbage_output_labeled_features.csv")
y_babbage = df_babbage['truthfulness']
X_babbage = df_babbage.drop(labels="truthfulness", axis=1)
X_train_babbage, X_test_babbage, y_train_babbage, y_test_babbage = train_test_split(X_babbage, y_babbage, test_size=0.2,random_state=2023)

clf_babbage = svm.SVC(kernel='rbf')
clf_babbage.fit(X_train_babbage, y_train_babbage)

y_test_pred_babbage = clf_babbage.predict(X_test_babbage)
print("Accuracy-train-babbage-test-babbage:",metrics.accuracy_score(y_test_babbage, y_test_pred_babbage))


df_curie = pd.read_csv("../outputs/curie_output_labeled_features.csv")
y_curie = df_curie['truthfulness']
X_curie = df_curie.drop(labels="truthfulness", axis=1)
X_train_curie, X_test_curie, y_train_curie, y_test_curie = train_test_split(X_curie, y_curie, test_size=0.2,random_state=2023)

clf_curie = svm.SVC(kernel='rbf')
clf_curie.fit(X_train_curie, y_train_curie)

y_test_pred_curie = clf_curie.predict(X_test_curie)
print("Accuracy-train-curie-test-curie:",metrics.accuracy_score(y_test_curie, y_test_pred_curie))


df_davinci = pd.read_csv("../outputs/davinci_output_labeled_features.csv")
y_davinci = df_davinci['truthfulness']
X_davinci = df_davinci.drop(labels="truthfulness", axis=1)
X_train_davinci, X_test_davinci, y_train_davinci, y_test_davinci = train_test_split(X_davinci, y_davinci, test_size=0.2,random_state=2023)

clf_davinci = svm.SVC(kernel='rbf')
clf_davinci.fit(X_train_davinci, y_train_davinci)

y_cross = clf_davinci.predict(X_test_davinci)
print("Accuracy-train-davinci-test-davinci:",metrics.accuracy_score(y_test_davinci, y_cross))


print("\n")

y_cross = clf_davinci.predict(X_curie)
print("Accuracy-train-davinci-test-curie:",metrics.accuracy_score(y_curie, y_cross))

y_cross = clf_davinci.predict(X_babbage)
print("Accuracy-train-davinci-test-babbage:",metrics.accuracy_score(y_babbage, y_cross))

y_cross = clf_davinci.predict(X_ada)
print("Accuracy-train-davinci-test-ada:",metrics.accuracy_score(y_ada, y_cross))


y_cross = clf_curie.predict(X_davinci)
print("Accuracy-train-curie-test-davinci:",metrics.accuracy_score(y_davinci, y_cross))

y_cross = clf_curie.predict(X_babbage)
print("Accuracy-train-curie-test-babbage:",metrics.accuracy_score(y_babbage, y_cross))

y_cross = clf_curie.predict(X_ada)
print("Accuracy-train-curie-test-ada:",metrics.accuracy_score(y_ada, y_cross))


y_cross = clf_babbage.predict(X_davinci)
print("Accuracy-train-babbage-test-davinci:",metrics.accuracy_score(y_davinci, y_cross))

y_cross = clf_babbage.predict(X_curie)
print("Accuracy-train-babbage-test-curie:",metrics.accuracy_score(y_curie, y_cross))

y_cross = clf_babbage.predict(X_ada)
print("Accuracy-train-babbage-test-ada:",metrics.accuracy_score(y_ada, y_cross))


y_cross = clf_ada.predict(X_davinci)
print("Accuracy-train-ada-test-davinci:",metrics.accuracy_score(y_davinci, y_cross))

y_cross = clf_ada.predict(X_curie)
print("Accuracy-train-ada-test-curie:",metrics.accuracy_score(y_curie, y_cross))

y_cross = clf_ada.predict(X_babbage)
print("Accuracy-train-ada-test-babbage:",metrics.accuracy_score(y_babbage, y_cross))



print("\n")


"""
EXPERIMENT: Generalization across sizes
"""

df_babbage_curie_davinci = pd.concat([df_babbage, df_curie, df_davinci], axis=0)
y_babbage_curie_davinci = df_babbage_curie_davinci['truthfulness']
X_babbage_curie_davinci = df_babbage_curie_davinci.drop(labels="truthfulness", axis=1)

clf_babbage_curie_davinci = svm.SVC(kernel='rbf')
clf_babbage_curie_davinci.fit(X_babbage_curie_davinci, y_babbage_curie_davinci)

y_pred_ada = clf_babbage_curie_davinci.predict(X_ada)
print("Accuracy-train-babbage-curie-davinci-test-ada:",metrics.accuracy_score(y_ada, y_pred_ada))

df_babbage_curie_davinci = pd.concat([df_babbage, df_curie, df_davinci], axis=0)
y_babbage_curie_davinci = df_babbage_curie_davinci['truthfulness']
X_babbage_curie_davinci = df_babbage_curie_davinci.drop(labels="truthfulness", axis=1)

clf_babbage_curie_davinci = svm.SVC(kernel='rbf')
clf_babbage_curie_davinci.fit(X_babbage_curie_davinci, y_babbage_curie_davinci)

y_pred_babbage = clf_babbage_curie_davinci.predict(X_babbage)
print("Accuracy-train-babbage-curie-davinci-test-babbage:",metrics.accuracy_score(y_babbage, y_pred_babbage))

df_babbage_curie_davinci = pd.concat([df_babbage, df_curie, df_davinci], axis=0)
y_babbage_curie_davinci = df_babbage_curie_davinci['truthfulness']
X_babbage_curie_davinci = df_babbage_curie_davinci.drop(labels="truthfulness", axis=1)

clf_babbage_curie_davinci = svm.SVC(kernel='rbf')
clf_babbage_curie_davinci.fit(X_babbage_curie_davinci, y_babbage_curie_davinci)

y_pred_curie = clf_babbage_curie_davinci.predict(X_curie)
print("Accuracy-train-babbage-curie-davinci-test-curie:",metrics.accuracy_score(y_curie, y_pred_curie))

df_babbage_curie_davinci = pd.concat([df_babbage, df_curie, df_davinci], axis=0)
y_babbage_curie_davinci = df_babbage_curie_davinci['truthfulness']
X_babbage_curie_davinci = df_babbage_curie_davinci.drop(labels="truthfulness", axis=1)

clf_babbage_curie_davinci = svm.SVC(kernel='rbf')
clf_babbage_curie_davinci.fit(X_babbage_curie_davinci, y_babbage_curie_davinci)

y_pred_davinci = clf_babbage_curie_davinci.predict(X_davinci)
print("Accuracy-train-babbage-curie-davinci-test-davinci:",metrics.accuracy_score(y_davinci, y_pred_davinci))


print("\n")



df_ada_curie_davinci = pd.concat([df_ada, df_curie, df_davinci], axis=0)
y_ada_curie_davinci = df_ada_curie_davinci['truthfulness']
X_ada_curie_davinci = df_ada_curie_davinci.drop(labels="truthfulness", axis=1)

clf_ada_curie_davinci = svm.SVC(kernel='rbf')
clf_ada_curie_davinci.fit(X_ada_curie_davinci, y_ada_curie_davinci)

y_pred_ada = clf_ada_curie_davinci.predict(X_ada)
print("Accuracy-train-ada-curie-davinci-test-ada:",metrics.accuracy_score(y_ada, y_pred_ada))

df_ada_curie_davinci = pd.concat([df_ada, df_curie, df_davinci], axis=0)
y_ada_curie_davinci = df_ada_curie_davinci['truthfulness']
X_ada_curie_davinci = df_ada_curie_davinci.drop(labels="truthfulness", axis=1)

clf_ada_curie_davinci = svm.SVC(kernel='rbf')
clf_ada_curie_davinci.fit(X_ada_curie_davinci, y_ada_curie_davinci)

y_pred_babbage = clf_ada_curie_davinci.predict(X_babbage)
print("Accuracy-train-ada-curie-davinci-test-babbage:",metrics.accuracy_score(y_babbage, y_pred_babbage))

df_ada_curie_davinci = pd.concat([df_ada, df_curie, df_davinci], axis=0)
y_ada_curie_davinci = df_ada_curie_davinci['truthfulness']
X_ada_curie_davinci = df_ada_curie_davinci.drop(labels="truthfulness", axis=1)

clf_ada_curie_davinci = svm.SVC(kernel='rbf')
clf_ada_curie_davinci.fit(X_ada_curie_davinci, y_ada_curie_davinci)

y_pred_curie = clf_ada_curie_davinci.predict(X_curie)
print("Accuracy-train-ada-curie-davinci-test-curie:",metrics.accuracy_score(y_curie, y_pred_curie))

df_ada_curie_davinci = pd.concat([df_ada, df_curie, df_davinci], axis=0)
y_ada_curie_davinci = df_ada_curie_davinci['truthfulness']
X_ada_curie_davinci = df_ada_curie_davinci.drop(labels="truthfulness", axis=1)

clf_ada_curie_davinci = svm.SVC(kernel='rbf')
clf_ada_curie_davinci.fit(X_ada_curie_davinci, y_ada_curie_davinci)

y_pred_davinci = clf_ada_curie_davinci.predict(X_davinci)
print("Accuracy-train-ada-curie-davinci-test-davinci:",metrics.accuracy_score(y_davinci, y_pred_davinci))


print("\n")


df_ada_babbage_davinci = pd.concat([df_ada, df_babbage, df_davinci], axis=0)
y_ada_babbage_davinci = df_ada_babbage_davinci['truthfulness']
X_ada_babbage_davinci = df_ada_babbage_davinci.drop(labels="truthfulness", axis=1)

clf_ada_babbage_davinci = svm.SVC(kernel='rbf')
clf_ada_babbage_davinci.fit(X_ada_babbage_davinci, y_ada_babbage_davinci)

y_pred_ada = clf_ada_babbage_davinci.predict(X_ada)
print("Accuracy-train-ada-babbage-davinci-test-ada:",metrics.accuracy_score(y_ada, y_pred_ada))

df_ada_babbage_davinci = pd.concat([df_ada, df_babbage, df_davinci], axis=0)
y_ada_babbage_davinci = df_ada_babbage_davinci['truthfulness']
X_ada_babbage_davinci = df_ada_babbage_davinci.drop(labels="truthfulness", axis=1)

clf_ada_babbage_davinci = svm.SVC(kernel='rbf')
clf_ada_babbage_davinci.fit(X_ada_babbage_davinci, y_ada_babbage_davinci)

y_pred_babbage = clf_ada_babbage_davinci.predict(X_babbage)
print("Accuracy-train-ada-babbage-davinci-test-babbage:",metrics.accuracy_score(y_babbage, y_pred_babbage))

df_ada_babbage_davinci = pd.concat([df_ada, df_babbage, df_davinci], axis=0)
y_ada_babbage_davinci = df_ada_babbage_davinci['truthfulness']
X_ada_babbage_davinci = df_ada_babbage_davinci.drop(labels="truthfulness", axis=1)

clf_ada_babbage_davinci = svm.SVC(kernel='rbf')
clf_ada_babbage_davinci.fit(X_ada_babbage_davinci, y_ada_babbage_davinci)

y_pred_curie = clf_ada_babbage_davinci.predict(X_curie)
print("Accuracy-train-ada-babbage-davinci-test-curie:",metrics.accuracy_score(y_curie, y_pred_curie))

df_ada_babbage_davinci = pd.concat([df_ada, df_babbage, df_davinci], axis=0)
y_ada_babbage_davinci = df_ada_babbage_davinci['truthfulness']
X_ada_babbage_davinci = df_ada_babbage_davinci.drop(labels="truthfulness", axis=1)

clf_ada_babbage_davinci = svm.SVC(kernel='rbf')
clf_ada_babbage_davinci.fit(X_ada_babbage_davinci, y_ada_babbage_davinci)

y_pred_davinci = clf_ada_babbage_davinci.predict(X_davinci)
print("Accuracy-train-ada-babbage-davinci-test-davinci:",metrics.accuracy_score(y_davinci, y_pred_davinci))



print("\n")


df_ada_babbage_curie = pd.concat([df_ada, df_babbage, df_curie], axis=0)
y_ada_babbage_curie = df_ada_babbage_curie['truthfulness']
X_ada_babbage_curie = df_ada_babbage_curie.drop(labels="truthfulness", axis=1)

clf_ada_babbage_curie = svm.SVC(kernel='rbf')
clf_ada_babbage_curie.fit(X_ada_babbage_curie, y_ada_babbage_curie)

y_pred_ada = clf_ada_babbage_curie.predict(X_ada)
print("Accuracy-train-ada-babbage-curie-test-ada:",metrics.accuracy_score(y_ada, y_pred_ada))

df_ada_babbage_curie = pd.concat([df_ada, df_babbage, df_curie], axis=0)
y_ada_babbage_curie = df_ada_babbage_curie['truthfulness']
X_ada_babbage_curie = df_ada_babbage_curie.drop(labels="truthfulness", axis=1)

clf_ada_babbage_curie = svm.SVC(kernel='rbf')
clf_ada_babbage_curie.fit(X_ada_babbage_curie, y_ada_babbage_curie)

y_pred_babbage = clf_ada_babbage_curie.predict(X_babbage)
print("Accuracy-train-ada-babbage-curie-test-babbage:",metrics.accuracy_score(y_babbage, y_pred_babbage))

df_ada_babbage_curie = pd.concat([df_ada, df_babbage, df_curie], axis=0)
y_ada_babbage_curie = df_ada_babbage_curie['truthfulness']
X_ada_babbage_curie = df_ada_babbage_curie.drop(labels="truthfulness", axis=1)

clf_ada_babbage_curie = svm.SVC(kernel='rbf')
clf_ada_babbage_curie.fit(X_ada_babbage_curie, y_ada_babbage_curie)

y_pred_curie = clf_ada_babbage_curie.predict(X_curie)
print("Accuracy-train-ada-babbage-curie-test-curie:",metrics.accuracy_score(y_curie, y_pred_curie))

df_ada_babbage_curie = pd.concat([df_ada, df_babbage, df_curie], axis=0)
y_ada_babbage_curie = df_ada_babbage_curie['truthfulness']
X_ada_babbage_curie = df_ada_babbage_curie.drop(labels="truthfulness", axis=1)

clf_ada_babbage_curie = svm.SVC(kernel='rbf')
clf_ada_babbage_curie.fit(X_ada_babbage_curie, y_ada_babbage_curie)

y_pred_davinci = clf_ada_babbage_curie.predict(X_davinci)
print("Accuracy-train-ada-babbage-curie-test-davinci:",metrics.accuracy_score(y_davinci, y_pred_davinci))



print("\n")


"""
EXPERIMENT: Generalization across datasets
"""
df_davinci_truthful_QA = pd.read_csv("../outputs/davinci_output_labeled_features.csv")
y_davinci_truthful_QA = df_davinci_truthful_QA['truthfulness']
X_davinci_truthful_QA = df_davinci_truthful_QA.drop(labels="truthfulness", axis=1)

df_davinci_openbook_QA = pd.read_csv("../outputs/openbookqa_davinci_labelled_features.csv")
y_davinci_openbook_QA = df_davinci_openbook_QA['truthfulness']
X_davinci_openbook_QA = df_davinci_openbook_QA.drop(labels="truthfulness", axis=1)

clf_davinci = svm.SVC(kernel='rbf')
clf_davinci.fit(X_davinci_truthful_QA, y_davinci_truthful_QA)

y_davinci_openbook_QA_pred = clf_davinci.predict(X_davinci_openbook_QA)
print("Accuracy-train-davinciTruthfulQA-test-davinciOpenBookQA:",metrics.accuracy_score(y_davinci_openbook_QA, y_davinci_openbook_QA_pred))



clf_davinci = svm.SVC(kernel='rbf')
clf_davinci.fit(X_davinci_openbook_QA, y_davinci_openbook_QA)

y_davinci_truthful_QA_pred = clf_davinci.predict(X_davinci_truthful_QA)
print("Accuracy-train-davinciOpenBookQA-test-davinciTruthfulnessQA:",metrics.accuracy_score(y_davinci_truthful_QA, y_davinci_truthful_QA_pred))



df_davinci_openbook_QA = pd.read_csv("../outputs/openbookqa_davinci_labelled_features.csv")
y_davinci_openbook_QA = df_davinci_openbook_QA['truthfulness']
X_davinci_openbook_QA = df_davinci_openbook_QA.drop(labels="truthfulness", axis=1)
X_train_davinci_openbook_QA, X_test_davinci_openbook_QA, y_train_davinci_openbook_QA, y_test_davinci_openbook_QA = train_test_split(X_davinci_openbook_QA, y_davinci_openbook_QA, test_size=0.2,random_state=2023)

clf_davinci = svm.SVC(kernel='rbf')
clf_davinci.fit(X_train_davinci_openbook_QA, y_train_davinci_openbook_QA)

y_test_pred_davinci_openbook_QA = clf_davinci.predict(X_test_davinci_openbook_QA)
print("Accuracy-train-davinciOpenBookQA-test-davinciOpenBookQA:",metrics.accuracy_score(y_test_davinci_openbook_QA, y_test_pred_davinci_openbook_QA))

df_davinci_openbook_QA_original = pd.read_csv("../outputs/openbookqa_davinci_labelled.csv")
y_davinci_openbook_QA_original = df_davinci_openbook_QA_original['label']
X_davinci_openbook_QA_original = df_davinci_openbook_QA_original.drop(labels='label', axis=1)
X_train_davinci_openbook_QA_original, X_test_davinci_openbook_QA_original, y_train_davinci_openbook_QA_original, y_test_davinci_openbook_QA_original = train_test_split(X_davinci_openbook_QA_original, y_davinci_openbook_QA_original, test_size=0.2,random_state=2023)

data = {
    'response': X_test_davinci_openbook_QA_original.model_answer.values.tolist(),
    'label': y_test_davinci_openbook_QA_original.tolist(),
    'label-ignore': y_test_davinci_openbook_QA.tolist(),
    'pred': y_test_pred_davinci_openbook_QA.tolist()
}
df = pd.DataFrame(data)
df.to_csv('../outputs/pred_openbookqa_davinci.csv',index=False)




df_davinci_truthful_QA = pd.read_csv("../outputs/davinci_output_labeled_features.csv")
y_davinci_truthful_QA = df_davinci_truthful_QA['truthfulness']
X_davinci_truthful_QA = df_davinci_truthful_QA.drop(labels="truthfulness", axis=1)
X_train_davinci_truthful_QA, X_test_davinci_truthful_QA, y_train_davinci_truthful_QA, y_test_davinci_truthful_QA = train_test_split(X_davinci_truthful_QA, y_davinci_truthful_QA, test_size=0.2,random_state=2023)

clf_davinci = svm.SVC(kernel='rbf')
clf_davinci.fit(X_train_davinci_truthful_QA, y_train_davinci_truthful_QA)

y_test_pred_davinci_truthful_QA = clf_davinci.predict(X_test_davinci_truthful_QA)
print("Accuracy-train-davinciTruthfulQA-test-davinciTruthfulQA:",metrics.accuracy_score(y_test_davinci_truthful_QA, y_test_pred_davinci_truthful_QA))

df_davinci_truthful_QA_original = pd.read_csv("../outputs/davinci_output_labeled.csv")
y_davinci_truthful_QA_original = df_davinci_truthful_QA_original['pred']
X_davinci_truthful_QA_original = df_davinci_truthful_QA_original.drop(labels='pred', axis=1)
X_train_davinci_truthful_QA_original, X_test_davinci_truthful_QA_original, y_train_davinci_truthful_QA_original, y_test_davinci_truthful_QA_original = train_test_split(X_davinci_truthful_QA_original, y_davinci_truthful_QA_original, test_size=0.2,random_state=2023)

data = {
    'response': X_test_davinci_truthful_QA_original.model_answer.values.tolist(),
    'label': y_test_davinci_truthful_QA_original.tolist(),
    'label-ignore': y_test_davinci_truthful_QA.tolist(),
    'pred': y_test_pred_davinci_truthful_QA.tolist()
}
df = pd.DataFrame(data)
df.to_csv('../outputs/pred_truthfulqa_davinci.csv',index=False)