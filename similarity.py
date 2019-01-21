import csv
import spacy
from tqdm import tqdm
from math import log
from random import random
from scipy.special import softmax
import pickle
import re


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


def similarity(s1, s2):
    doc1 = nlp(s1)
    doc2 = nlp(s2)
    prob = clamp(abs(doc1.similarity(doc2)), 0, 1) / 2
    is_dup = prob > 0.6
    return (prob, is_dup, doc1.similarity(doc2))


def bag_of_words(s1, s2, extra_debug=None):
    words1 = ngrams(s1)
    len1 = len(words1)
    words2 = ngrams(s2)
    len2 = len(words2)

    if len1 == 0 and len2 == 0:
        return (0.5, 0)

    average_length = (len1 + len2) / 2.0
    common_words = words1.intersection(words2)

    prob = len(common_words) / average_length
    is_dup = prob > 0.5
    return (prob, is_dup, {
        'common': common_words,
        'len1': len1,
        'len2': len2,
        'extra': extra_debug
    })


def named_entities(s1, s2):
    doc1 = nlp2(s1)
    doc2 = nlp2(s2)
    w1 = doc1.ents
    w2 = doc2.ents
    if len(w1) == 0 and len(w2) == 0:
        return None
    s1 = ' '.join([str(w) for w in w1])
    s2 = ' '.join([str(w) for w in w2])
    return bag_of_words(s1, s2, s1 + ' | ' + s2)


def weighted_average(values, weights):
    return sum(values[i] * float(weights[i])
               for i in range(0, len(values))) / sum(weights)


def bag_of_words_100(s1, s2):
    words1 = set(s1.split()) - top_100
    len1 = len(words1)
    words2 = set(s2.split()) - top_100
    len2 = len(words2)

    average_length = (len1 + len2) / 2.0
    common_words = words1.intersection(words2)

    prob = len(common_words) / average_length
    is_dup = prob > 0.5
    return (prob, is_dup, {
        'common_words': common_words,
        'ave_len': average_length
    })


def ngrams(s):
    s = s.lower()
    s = re.sub('[^a-z]', ' ', s)
    s = re.sub('\s+', ' ', s)
    words = s.lower().split()
    ns = set(words)
    return ns
    for i, word in enumerate(words[:len(words) - 1]):
        next_word = words[i + 1]
        ns.add(word + '_' + next_word)
        if i < len(words) - 2:
            ns.add(word + '_' + next_word + '_' + words[i + 2])
    return ns


def tf_idf(s1, s2):
    words1 = ngrams(s1)
    words2 = ngrams(s2)

    different_words = words1 - words2

    difference_weight = 0
    debug = []
    for word in different_words:
        tf = 1
        idf = log(total_questions / (vocab[word]))
        difference_weight += tf * idf
        debug.append({'word': word, idf: idf})

    prob = clamp(10 / (1 + difference_weight), 0, 1)
    is_dup = prob > 0.5
    return (prob, is_dup)


def tf_idf_jaccard(s1, s2):
    words1 = ngrams(s1)
    words2 = ngrams(s2)

    all_words = words1.union(words2)
    all_weight = 0
    for word in all_words:
        tf = 1
        idf = log(total_questions / (vocab[word]))
        all_weight += tf * idf

    different_words = words1 - words2
    difference_weight = 0
    debug = []
    for word in different_words:
        tf = 1
        idf = log(total_questions / (vocab[word]))
        difference_weight += tf * idf
        debug.append({'word': word, idf: idf})

    prob = clamp(3.0 * (all_weight - difference_weight) / (4.0 * all_weight),
                 0, 1)
    is_dup = prob > 0.5
    return (prob, is_dup, debug)


def tf_idf_softmax(s1, s2):
    words1 = ngrams(s1)
    words2 = ngrams(s2)

    different_words = words1 - words2

    difference_weight = 0
    for word in different_words:
        tf = 1
        idf = log(total_questions / (vocab[word]))
        difference_weight += tf * idf

    common_words = words1.intersection(words2)
    common_weight = 0
    for word in common_words:
        tf = 1
        idf = log(total_questions / vocab[word])
        common_weight += tf * idf

    prob = softmax([difference_weight, common_weight])[1]
    is_dup = prob > 0.5
    return (prob, is_dup)


def preprocess(s):
    s = s.lower()
    s = re.sub('[^a-z]', ' ', s)
    s = re.sub('\s+', ' ', s)

    s = re.sub('^what is the (\w+ )?way', 'how', s)
    s = re.sub('^what can i do to', 'how do i', s)
    s = re.sub('^do you think of (.*) as', 'is \1 a', s)

    return s.strip()


def question_pairs_prob(s1, s2):
    p1 = preprocess(s1).split()[0]
    p2 = preprocess(s2).split()[0]
    if p1 > p2:
        p1, p2 = (p2, p1)
    if (p1, p2) not in question_pairs:
        return None
    r = question_pairs[(p1, p2)]
    prob = clamp(float(1.3 * r['duplicate']) / r['total'], 0, 1.0)
    is_dup = prob > 0.5
    return (prob, is_dup, (p1, p2))


def root_pairs_prob(s1, s2):
    s1 = preprocess(s1)
    s2 = preprocess(s2)
    doc1 = nlp2(s1)
    doc2 = nlp2(s2)
    root1 = str(next(doc1.sents).root)
    root2 = str(next(doc2.sents).root)
    if root1 > root2:
        root1, root2 = (root2, root1)
    if (root1, root2) not in root_pairs:
        return None
    r = root_pairs[(root1, root2)]
    prob = clamp(float(1.3 * r['duplicate']) / r['total'], 0, 1.0)
    is_dup = prob > 0.5
    return (prob, is_dup, (root1, root2))


def cross_entropy(prediction, target):
    if target == 1:
        if prediction == 0:
            return 10
        return -log(prediction)
    else:
        if prediction == 1:
            return 10
        return -log(1 - prediction)


print('loading vectors')
nlp = spacy.load('en_vectors_web_lg')
print('loading nlp core')
nlp2 = spacy.load('en_core_web_md')
print('loading question pairs')
question_pairs = pickle.load(open('question_pairs.obj', 'rb'))
print('loading root pairs')
root_pairs = pickle.load(open('root_pairs.obj', 'rb'))
total_pairs = 0
total_loss = 0
num_accurate = 0
#top_100 = set([s.strip() for s in open('top_100.txt').readlines()])
#top_100 = set(['is', 'a', 'the', 'be', 'will', 'of', 'to', 'on'])
top_100 = set()
vocab = {}
print('loading questions')
questions = set([s.strip() for s in open('questions.txt').readlines()])
total_questions = len(questions)
for q in tqdm(questions, total=537000):
    for word in ngrams(q):
        if word not in vocab:
            vocab[word] = 0
        vocab[word] += 1

if __name__ == '__main__':

    with open('train_small.csv', 'r') as f:
        data = csv.DictReader(f, delimiter=',')
        t = tqdm(data, total=1000)
        models = [
            bag_of_words, named_entities, question_pairs_prob, root_pairs_prob
        ]
        model_weights = [1, 1, 1, 1, 1, 1]
        predictions = [[] for m in models]
        losses = [0 for m in models]
        for row in t:
            total_pairs += 1
            if total_pairs > 1000:
                break
            s1 = row['question1']
            s2 = row['question2']
            target = float(row['is_duplicate'])
            current_predictions = []
            current_weights = []
            current_is_dups = []
            debug = [() for m in models]
            for i, model in enumerate(models):
                ret = model(s1, s2)
                if ret is None:
                    continue

                (prediction, is_dup, d) = ret
                debug[i] = d
                predictions[i].append(prediction)
                current_predictions.append(prediction)
                current_weights.append(model_weights[i])
                current_is_dups.append(is_dup)
                losses[i] += cross_entropy(prediction, target)
            """
            if sum(current_is_dups) / len(current_is_dups) > 0.5:
                is_dup = 1
            else:
                is_dup = 0
            """
            prediction = weighted_average(current_predictions, current_weights)
            if prediction > 0.5:
                is_dup = 1
            else:
                is_dup = 0
            total_loss += cross_entropy(prediction, target)
            if is_dup == target:
                num_accurate += 1
            accuracy = float(num_accurate) / total_pairs
            t.set_postfix({'loss': total_loss, 'accuracy': accuracy})
            if is_dup != target:
                t.write(str([s1, s2, prediction, int(is_dup), target]))
                t.write(str(current_predictions))
                t.write('debug info: ' + str(debug))
        print('losses', losses)
        print('loss=', total_loss, accuracy)
        print('avg prediction: ', [
            sum(model_predictions) / len(model_predictions)
            for model_predictions in predictions
        ])

# print(s1, s2, target, prediction, cross_entropy(prediction, target))
