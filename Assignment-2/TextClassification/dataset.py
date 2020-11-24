from utils import *
from sklearn.preprocessing import LabelEncoder


class TextDataSet:
    def __init__(self, data_path='data/'):
        self.data_path = data_path
        self.stem_to_idx_map = None
        self.idx_to_stem_map = None
        self.le = LabelEncoder()
        self.mapping = None

    def generate_text_dataset(self, train_size=500, valid_size=200, test_size=500, return_text=False):
        stem_set = set()

        train_set_line, valid_set_line, test_set_line = [], [], []
        train_label, valid_label, test_label = [], [], []

        with open(self.data_path + 'topics.txt', 'r', encoding='utf-8') as file:
            topics = [t.strip() for t in file.readlines()]

            for topic in topics:
                if topic == '3d_Printer':
                    continue

                with open(self.data_path + 'Training/%s.xml' % topic, 'r', encoding='utf-8') as file:
                    content = file.read()

                    texts = get_text_list_from_xml(content, n=None, remove_links=True)

                    lines = []
                    labels = []
                    for line in texts:
                        stem = get_stem_tokens(line)
                        if len(stem) > 0:
                            lines.append(line)
                            labels.append(topic)

                        if len(labels) >= train_size + valid_size + test_size:
                            break

                    train_set_line.extend(lines[:train_size])
                    valid_set_line.extend(lines[train_size:train_size + valid_size])
                    test_set_line.extend(lines[train_size + valid_size:train_size + valid_size + test_size])

                    train_label.extend(labels[:train_size])
                    valid_label.extend(labels[train_size:train_size + valid_size])
                    test_label.extend(labels[train_size + valid_size:train_size + valid_size + test_size])

                    for line in train_set_line:
                        stem_set.update(get_stem_tokens(line))
                    for line in valid_set_line:
                        stem_set.update(get_stem_tokens(line))

        self.stem_to_idx_map = {stem: i for i, stem in enumerate(stem_set)}
        self.idx_to_stem_map = {i: stem for stem, i in self.stem_to_idx_map.items()}

        self.le.fit(train_label)
        self.mapping = dict(zip(self.le.classes_, range(len(self.le.classes_))))

        assert (len(train_set_line) == len(train_label))
        assert (len(valid_set_line) == len(valid_label))
        assert (len(test_set_line) == len(test_label))

        if return_text:
            return train_set_line, valid_set_line, test_set_line, train_label, valid_label, test_label

        X_train, y_train = self.embedding_from_text(train_set_line, train_label)
        X_valid, y_valid = self.embedding_from_text(valid_set_line, valid_label)
        X_test, y_test = self.embedding_from_text(test_set_line, test_label)

        assert (X_train.shape[0] == y_train.shape[0])
        assert (X_valid.shape[0] == y_valid.shape[0])
        assert (X_test.shape[0] == y_test.shape[0])

        return X_train, X_valid, X_test, y_train, y_valid, y_test

    def embedding_from_text(self, X, y=None):
        X_stem = [get_stem_tokens(text) for text in X]
        return embeding_from_stem(self.stem_to_idx_map, X_stem) if y is None else (
            embeding_from_stem(self.stem_to_idx_map, X_stem), self.le.transform(y))

    def embedding_from_stem(self, X, y=None):
        return embeding_from_stem(self.stem_to_idx_map, X) if y is None else (
            embeding_from_stem(self.stem_to_idx_map, X), self.le.transform(y))

    def class_label(self, class_id):
        return self.mapping[class_id]
