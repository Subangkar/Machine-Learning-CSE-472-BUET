import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# %%
from dataset import generate_dataset
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
            print(model.score(X_train, y_train))
            print("Number of mislabeled samples out of a total %d samples in Train: %d"
                  % (X_train.shape[0], (y_train != model.predict(X_train)).sum()))

        if X_valid is not None and y_valid is not None:
            print(model.score(X_valid, y_valid))
            print("Number of mislabeled samples out of a total %d samples in Validation: %d"
                  % (X_valid.shape[0], (y_valid != model.predict(X_valid)).sum()))

        if X_test is not None and y_test is not None:
            print(model.score(X_test, y_test))
            print("Number of mislabeled samples out of a total %d samples in Test: %d"
                  % (X_test.shape[0], (y_test != model.predict(X_test)).sum()))


# %%
X_train, X_valid, X_test, y_train, y_valid, y_test, mapping = generate_dataset(valid_size=200)

print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)
print(X_test.shape, y_test.shape)

# %%
nb = NaiveBayes(smoothing_factor=1e-3)  # GaussianNB()
clf = TextClassifier(nb)
clf.fit(X_train, y_train)
clf.evaluationStats(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, X_test=X_test, y_test=y_test)

# %%
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
clf = TextClassifier(nb)
clf.fit(X_train, y_train)
clf.evaluationStats(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, X_test=X_test, y_test=y_test)

# %%
from sklearn.neighbors import KNeighborsClassifier

nb = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
clf = TextClassifier(nb)
clf.fit(X_train, y_train)
clf.evaluationStats(X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid, X_test=X_test, y_test=y_test)

# %%
X_train_ = X_train
X_valid_ = X_valid
X_test_ = X_test

clf = TextClassifier(Knn(n_neighbors=3, metric='euclidean'))
clf.fit(X_train_, y_train)
clf.evaluationStats(X_train=X_train_, y_train=y_train, X_valid=X_valid_, y_valid=y_valid, X_test=X_test_, y_test=y_test)

# %%
X_train_ = (X_train > 0).astype('float')
X_valid_ = (X_valid > 0).astype('float')
X_test_ = (X_test > 0).astype('float')

clf = TextClassifier(Knn(n_neighbors=3, metric='hamming'))
clf.fit(X_train_, y_train)
clf.evaluationStats(X_train=X_train_, y_train=y_train, X_valid=X_valid_, y_valid=y_valid, X_test=X_test_, y_test=y_test)

# %%
X_train_ = tf_idf(X_train, alpha=1e-6, beta=1e-9)
X_valid_ = tf_idf(X_valid, alpha=1e-6, beta=1e-9)
X_test_ = tf_idf(X_test, alpha=1e-6, beta=1e-9)

clf = TextClassifier(Knn(n_neighbors=3, metric='cosine'))
clf.fit(X_train_, y_train)
clf.evaluationStats(X_train=X_train_, y_train=y_train, X_valid=X_valid_, y_valid=y_valid, X_test=X_test_, y_test=y_test)
