# %%
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# %%
from utils import tf_idf

# %%
from models.naivebayes import NaiveBayes

# %%
from models.knn import Knn


# %%
class TextClassifier:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

    def evaluationStats(self, X_train=None, y_train=None, X_valid=None, y_valid=None, X_test=None, y_test=None):
        model = self.model
        if X_train is not None and y_train is not None:
            print('Train:', model.score(X_train, y_train))
            # print("Number of mislabeled samples out of a total %d samples in Train: %d"
            #       % (X_train.shape[0], (y_train != model.predict(X_train)).sum()))

        if X_valid is not None and y_valid is not None:
            print('Validation:', model.score(X_valid, y_valid))
            # print("Number of mislabeled samples out of a total %d samples in Validation: %d"
            #       % (X_valid.shape[0], (y_valid != model.predict(X_valid)).sum()))

        if X_test is not None and y_test is not None:
            print('Test:', model.score(X_test, y_test))
            # print("Number of mislabeled samples out of a total %d samples in Test: %d"
            #       % (X_test.shape[0], (y_test != model.predict(X_test)).sum()))


# %%
from dataset import TextDataSet

textds = TextDataSet(data_path='data/')
X_train, X_valid, X_test, y_train, y_valid, y_test = textds.generate_text_dataset(train_size=500,
                                                                                  valid_size=200,
                                                                                  test_size=500)

print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)
print(X_test.shape, y_test.shape)

# print(textds.embedding_from_text(['I love coffee']).shape)

# %%
knn_ds = {
    'hamming': ((X_train > 0).astype('float'), (X_valid > 0).astype('float')),
    'euclidean': (X_train, X_valid),
    'cosine': (tf_idf(X_train, alpha=1e-6, beta=1e-9), tf_idf(X_valid, alpha=1e-6, beta=1e-9)),
}

best_acc = 0
best_metric = None
best_k = 0
for metric, (X_train_, X_valid_) in knn_ds.items():
    for k in [1, 3, 5]:
        clf = TextClassifier(Knn(n_neighbors=k, metric=metric))
        clf.fit(X_train_, y_train)
        acc = clf.score(X_valid_, y_valid)
        print(metric, k, round(acc * 100, 2), sep=', ')

        if acc > best_acc:
            best_acc = acc
            best_metric = metric
            best_k = k
print(best_metric, best_k, round(best_acc * 100, 2), sep=', ')

# %%
from sklearn.neighbors import KNeighborsClassifier

clf = TextClassifier(KNeighborsClassifier(n_neighbors=5, metric='cosine'))
clf.fit(tf_idf(X_train, alpha=1e-6, beta=1e-9), y_train)
acc = clf.score(tf_idf(X_valid, alpha=1e-6, beta=1e-9), y_valid)
print('scikit', 'cosine', 'k=5', round(acc * 100, 2), sep=', ')

# %%
import numpy as np

best_acc = 0
best_smoothing_factor = 0
for s in [1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
    clf = TextClassifier(NaiveBayes(smoothing_factor=s))
    clf.fit(X_train, y_train)
    acc = clf.score(X_valid, y_valid)
    print('smoothing factor', s, round(acc * 100, 2), sep=', ')
    if acc > best_acc:
        best_acc = acc
        best_smoothing_factor = s
print('best_smoothing_factor', best_smoothing_factor, round(best_acc * 100, 2), sep=', ')
# %%
from sklearn.naive_bayes import MultinomialNB

clf = TextClassifier(MultinomialNB())
clf.fit(X_train, y_train)
clf.evaluationStats(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, X_test=X_test, y_test=y_test)

# %%
import math

kn = TextClassifier(Knn(n_neighbors=5, metric='cosine'))
kn.fit(tf_idf(X_train, alpha=1e-6, beta=1e-9), y_train)
nb = TextClassifier(NaiveBayes(smoothing_factor=1e-3))
nb.fit(X_train, y_train)

X_test_ = tf_idf(X_test, alpha=1e-6, beta=1e-9)

stat = []
print('Iter', 'KNN', 'Naive Bayes', sep=', ')

for b in range(50):
    X = []
    X_ = []
    y = []

    for d in range(int(math.ceil(len(y_test) / 500))):
        st = d * 500 + b * 10
        en = min(d * 500 + b * 10 + 10, len(y_test))

        if st >= en:
            break

        X.append(X_test[st:en, :])
        X_.append(X_test_[st:en, :])
        y.append(y_test[st:en])

    X = np.concatenate(X)
    X_ = np.concatenate(X_)
    y = np.concatenate(y)

    acc_nb = nb.score(X, y)
    acc_kn = kn.score(X_, y)

    print(b + 1, round(acc_kn * 100, 2), round(acc_nb * 100, 2), sep=', ')
    stat.append((b + 1, acc_kn, acc_nb))
print('Overall', round(np.mean([a[1] for a in stat]) * 100, 2), round(np.mean([a[2] for a in stat]) * 100, 2), sep=', ')

# %%
from scipy import stats

kns=np.array([a[1] for a in stat])
nbs=np.array([a[2] for a in stat])

stats.ttest_rel(kns, nbs)

# %%
nb = TextClassifier(NaiveBayes(smoothing_factor=0.0026))
nb.fit(X_train, y_train)

# %%
textds.class_label(nb.predict(
                    textds.embedding_from_text(['I love working with hardware',
                                                'Coffee is bad for health',
                                                'animes are deployable in adruino'])))
# textds.embedding_from_text(['I love coffee'])