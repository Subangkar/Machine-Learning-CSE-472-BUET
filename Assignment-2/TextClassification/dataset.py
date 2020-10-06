from utils import *
from sklearn.preprocessing import LabelEncoder


def generate_dataset(data_path='data/', train_size=500, valid_size=200, test_size=500):
    stem_set = set()
    stem_to_idx_map = dict()
    idx_to_stem_map = dict()

    train_set_word_X, valid_set_word_X, test_set_word_X = [], [], []

    train_set_stem_X, valid_set_stem_X, test_set_stem_X = [], [], []

    train_label, valid_label, test_label = [], [], []
    X_train, X_valid, X_test = None, None, None
    y_train, y_valid, y_test = None, None, None

    with open(data_path + 'topics.txt', 'r', encoding='utf-8') as file:
        topics = [t.strip() for t in file.readlines()]

        for topic in topics:

            if topic == '3d_Printer':
                continue

            with open(data_path + 'Training/%s.xml' % topic, 'r', encoding='utf-8') as file:
                content = file.read()

                texts = get_text_list_from_xml(content, n=train_size + valid_size + test_size, remove_links=True)

                # train_set_word_X.extend(texts[:train_size])
                # train_label.extend([topic] * train_size)
                # valid_set_word_X.extend(texts[train_size: train_size+valid_size])
                # valid_label.extend([topic] * valid_size)
                # test_set_word_X.extend(texts[train_size+valid_size: train_size+valid_size+test_size])
                # test_label.extend([topic] * test_size)

                for line in texts[:train_size]:
                    stem = get_stem_tokens(line)
                    stem_set.update(stem)

                    if len(stem) > 0:
                        train_set_word_X.append(line)
                        train_set_stem_X.append(stem)
                        train_label.append(topic)

                for line in texts[train_size: train_size + valid_size]:
                    stem = get_stem_tokens(line)
                    stem_set.update(stem)

                    if len(stem) > 0:
                        valid_set_word_X.append(line)
                        valid_set_stem_X.append(stem)
                        valid_label.append(topic)

                for line in texts[train_size + valid_size: train_size + valid_size + test_size]:
                    stem = get_stem_tokens(line)
                    # stem_set.update(stem)

                    if len(stem) > 0:
                        test_set_word_X.append(line)
                        test_set_stem_X.append(stem)
                        test_label.append(topic)

    # print(len(stem_set))
    stem_to_idx_map = {stem: i for i, stem in enumerate(stem_set)}
    idx_to_stem_map = {i: stem for stem, i in stem_to_idx_map.items()}
    # print(stem_set)

    X_train = embeding_from_stem(stem_to_idx_map, train_set_stem_X)
    X_valid = embeding_from_stem(stem_to_idx_map, valid_set_stem_X)
    X_test = embeding_from_stem(stem_to_idx_map, test_set_stem_X)

    le = LabelEncoder()
    le.fit(train_label)

    y_train = le.transform(train_label)
    y_valid = le.transform(valid_label)
    y_test = le.transform(test_label)

    assert (X_train.shape[0] == y_train.shape[0])
    assert (X_valid.shape[0] == y_valid.shape[0])
    assert (X_test.shape[0] == y_test.shape[0])

    mapping = dict(zip(le.classes_, range(len(le.classes_))))

    return X_train, X_valid, X_test, y_train, y_valid, y_test, mapping
