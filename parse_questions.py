import spacy
import csv
import re
from math import log
from tqdm import tqdm

nlp = spacy.load('en_core_web_md')


def preprocess(s):
    s = s.lower()
    s = re.sub('[^a-z]', ' ', s)
    s = re.sub('\s+', ' ', s)

    s = re.sub('^what is the (\w+ )?way', 'how', s)
    s = re.sub('^what can i do to', 'how do i', s)
    s = re.sub('^do you think of (.*) as', 'is \1 a', s)

    return s.strip()


def cross_entropy(prediction, target):
    if target == 1:
        if prediction == 0:
            return 10
        return -log(prediction)
    else:
        if prediction == 1:
            return 10
        return -log(1 - prediction)


def ngrams(s):
    words = s.lower().split()
    ns = set(words)
    return ns
    for i, word in enumerate(words[:len(words) - 1]):
        next_word = words[i + 1]
        ns.add(word + '_' + next_word)
        if i < len(words) - 2:
            ns.add(word + '_' + next_word + '_' + words[i + 2])
    return ns


def bag_of_words(s1, s2):
    words1 = ngrams(s1)
    len1 = len(words1)
    words2 = ngrams(s2)
    len2 = len(words2)

    average_length = (len1 + len2) / 2.0
    common_words = words1.intersection(words2)

    prob = len(common_words) / average_length
    is_dup = prob > 0.5
    return (prob, is_dup)


def concat(ls):
    for l in ls:
        for x in l:
            yield x


question_pairs = {}
root_pairs = {}


def parse_node(n):
    lefts = []
    for nl in n.lefts:
        lefts.append(parse_node(nl))
    rights = []
    for nr in n.rights:
        rights.append(parse_node(nr))
    return {
        'word': str(n),
        'lemma': n.lemma_,
        'dep': n.dep_,
        'pos': n.pos_,
        'lefts': lefts,
        'rights': rights,
    }


class Node:
    def __init__(self, text):
        self.text = text

    def __repr__(self):
        return self.text

    def __str__(self):
        return self.text

    def similarity(self, other):
        return nlp(self.text).similarity(nlp(other.text))


class Subject(Node):
    pass


class Object(Node):
    pass


class Verb(Node):
    pass


class AdpositionalPhrase(Node):
    def __init__(self, root, doc):
        if len(list(root.rights)) == 1 and list(root.rights)[0].pos_ == 'NOUN':
            c_root = list(root.rights)[0]
            self.complement = doc[c_root.left_edge.i:c_root.right_edge.i + 1]
            self.adp = doc[root.left_edge.i:c_root.left_edge.i]
        else:
            self.complement = None
            self.adp = doc[root.left_edge.i:root.right_edge.i + 1]

    def __repr__(self):
        return str({
            'adp': self.adp.text,
            'complement': self.complement.__repr__()
        })

    def similarity(self, other):
        adp_similarity = self.adp.similarity(other.adp)
        print('adp_sim', adp_similarity)
        if self.complement is None and other.complement is None:
            print('both no compl')
            return adp_similarity
        elif self.complement is None or other.complement is None:
            print('one no compl')
            return adp_similarity / 2.0
        else:
            print('both compl')
            complement_similarity = self.complement.similarity(
                other.complement)
            return (adp_similarity + complement_similarity) / 2.0


class NounChunk(Node):
    def __init__(self, root, doc):
        if len(list(root.rights)) == 1 and list(root.rights)[0].pos_ == 'ADP':
            adp_root = list(root.rights)[0]
            self.noun_chunk = doc[root.left_edge.i:adp_root.left_edge.i]
            self.adposition = AdpositionalPhrase(adp_root, doc)
        else:
            self.noun_chunk = doc[root.left_edge.i:root.right_edge.i + 1]
            self.adposition = None
        self.text = self.noun_chunk.text
        Node.__init__(self, self.text)

    def __repr__(self):
        return str({
            'noun_chunk': self.noun_chunk.text,
            'adposition': self.adposition.__repr__()
        })

    def similarity(self, other):
        ncs = self.noun_chunk.similarity(other.noun_chunk)
        print('ncs', ncs)
        if self.adposition is None and other.adposition is None:
            print('no adps')
            return ncs
        elif self.adposition is None or other.adposition is None:
            print('one adp', ncs)
            return ncs / 2.0
        else:
            print('both adps')
            ads = self.adposition.similarity(other.adposition)
            return (ncs + ads) / 2.0


class Sentence:
    def __init__(self, s, v, o):
        self.s = s
        self.v = v
        self.o = o

    def similarity(self, other):
        return (self.s.similarity(other.s) + self.v.similarity(other.v) +
                self.o.similarity(other.o)) / 3.0


def parse_question(doc):
    sentences = []
    for sent in doc.sents:
        root = sent.root
        ls = ' '.join([d.text for l in root.lefts for d in l.subtree])
        rs = ' '.join([d.text for r in root.rights for d in r.subtree])
        if len(list(root.rights)) == 1 and list(
                root.rights)[0].pos_ == 'NOUN' and len(list(root.rights)) == 1:
            o = NounChunk(list(root.rights)[0], doc)
        else:
            o = Object(rs)
        sentences.append(Sentence(Subject(ls), Verb(root.text), o))
    return sentences[0]


def mean(l):
    return float(sum(l)) / len(l)


def similarity(s1, s2):
    return nlp(s1).similarity(nlp(s2))


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


if __name__ == '__main__':
    q1 = parse_question(
        nlp('Which is the best digital marketing institution in bangalore'))
    q2 = parse_question(
        nlp('Which is the best digital marketing institution in peru'))
    q3 = parse_question(nlp('Which is the best digital marketing institution'))
    print(q1.similarity(q2))
    print(q1.similarity(q3))
    print(q2.similarity(q3))
    exit(0)
    with open('train_small.csv') as f:
        data = csv.DictReader(f, delimiter=',')
        total = 0
        validation_loss_baseline = 0
        validation_loss = 0
        train_loss = 0
        #t = tqdm(enumerate(data), total=404000)
        epoch_size = 0
        total = 0.0
        loss = 0.0
        accurate = 0.0
        for i, sample in enumerate(data):
            total += 1
            target = int(sample['is_duplicate'])
            q1 = sample['question1']
            q2 = sample['question2']
            print(q1, q2, target)
            doc1 = nlp(q1)
            p1 = parse_question(doc1)
            doc2 = nlp(q2)
            p2 = parse_question(doc2)
            prediction = mean([
                p1[i].similarity(p2[i]) for i in range(max(len(p1), len(p2)))
            ])
            prediction = clamp(prediction, 0.0001, 0.999)
            is_dup = prediction > 0.5
            if is_dup == target:
                accurate += 1
            loss += cross_entropy(prediction, target)
            print(p1, p2, prediction, target)
            print('-------------------', total, float(accurate) / total, loss)
