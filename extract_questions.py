import csv
from tqdm import tqdm

questions = {}
with open('train.csv', 'r') as f:
    data = csv.DictReader(f, delimiter=',')
    for row in tqdm(data):
        questions[row['qid1']] = row['question1']
        questions[row['qid2']] = row['question2']

for qid, question in questions.items():
    print(question)
