# coding: utf-8
# create by tongshiwei on 2019/4/12

import numpy as np
import warnings
from longling import loading, path_append
import mxnet as mx
from tqdm import tqdm
from gluonnlp.data import FixedBucketSampler, PadSequence
from CangJie.utils.embeddings import load_embedding, token_to_idx

__all__ = ["extract", "transform", "etl", "pseudo_data_iter"]


def pseudo_data_iter(_cfg):
    def pseudo_data_generation():
        # 在这里定义测试用伪数据流
        import random
        random.seed(10)

        from CangJie.utils.testing import pseudo_sentence
        from CangJie import tokenize, characterize, token2radical

        sentences = pseudo_sentence(1000, 20)

        def feature2num(token):
            return random.randint(0, 10)

        w = [feature2num(tokenize(s)) for s in sentences]
        rw = [feature2num(token2radical(_w)) for _w in w]
        c = [feature2num(characterize(s)) for s in sentences]
        rc = [feature2num(token2radical(_c)) for _c in c]

        labels = [random.randint(0, 32) for _ in sentences]
        features = [w, rw, c, rc]

        return features, labels

    return load(transform(pseudo_data_generation(), _cfg), _cfg)


def extract(data_src, embeddings):
    word_feature = []
    word_radical_feature = []
    char_feature = []
    char_radical_feature = []
    features = [word_feature, word_radical_feature, char_feature,
                char_radical_feature]
    labels = []
    for ds in tqdm(loading(data_src), "loading data from %s" % data_src):
        label = ds['label']

        w = token_to_idx(embeddings["w"], ds["w"]) if embeddings.get("w") and "w" in ds else []
        rw = token_to_idx(embeddings["rw"], ds["rw"]) if embeddings.get("rw") and "rw" in ds else []
        c = token_to_idx(embeddings["rw"], ds["rw"]) if embeddings.get("c") and "c" in ds else []
        rc = token_to_idx(embeddings["rw"], ds["rw"]) if embeddings.get("rc") and "rc" in ds else []
        try:
            assert len(w) == len(rw), "some word miss radical"
            assert len(c) == len(rc), "some char miss radical"
        except AssertionError as e:
            warnings.warn("%s" % e)
            continue
        word_feature.append(w)
        word_radical_feature.append(rw)
        char_feature.append(c)
        char_radical_feature.append(rc)
        labels.append(label)

    return features, labels


def transform(raw_data, params):
    # 定义数据转换接口
    # raw_data --> batch_data

    batch_size = params.batch_size
    padding = params.padding
    num_buckets = params.num_buckets
    fixed_length = params.fixed_length

    features, labels = raw_data
    word_feature, word_radical_feature, char_feature, char_radical_feature = features
    batch_idxes = FixedBucketSampler(
        [len(word_f) for word_f in word_feature],
        batch_size, num_buckets=num_buckets
    )
    batch = []
    for batch_idx in batch_idxes:
        batch_features = [[] for _ in range(len(features))]
        batch_labels = []
        for idx in batch_idx:
            for i, feature in enumerate(batch_features):
                batch_features[i].append(features[i][idx])
            batch_labels.append(labels[idx])
        batch_data = []
        word_mask = []
        char_mask = []
        for i, feature in enumerate(batch_features):
            max_len = max(
                [len(fea) for fea in feature]
            ) if not fixed_length else fixed_length
            padder = PadSequence(max_len, pad_val=padding)
            feature, mask = zip(*[(padder(fea), len(fea)) for fea in feature])
            if i == 0:
                word_mask = mask
            elif i == 2:
                char_mask = mask
            batch_data.append(mx.nd.array(feature))
        batch_data.append(mx.nd.array(word_mask))
        batch_data.append(mx.nd.array(char_mask))
        batch_data.append(mx.nd.array(batch_labels, dtype=np.int))
        batch.append(batch_data)
    return batch[::-1]


def load(transformed_data, params):
    return transformed_data


def etl(filename, embeddings, params):
    raw_data = extract(filename, embeddings)
    transformed_data = transform(raw_data, params)
    return load(transformed_data, params)


if __name__ == '__main__':
    from longling.lib.structure import AttrDict
    from CangJie import PAD_TOKEN
    import os

    filename = "../../../../data/ctc32/train.json"
    print(os.path.abspath(filename))

    vec_root = "../../../../data/vec/"
    _embeddings = load_embedding(
        {
            "w": path_append(vec_root, "word.vec.dat"),
            "rw": path_append(vec_root, "word_radical.vec.dat"),
            "c": path_append(vec_root, "char.vec.dat"),
            "rc": path_append(vec_root, "char_radical.vec.dat")
        }
    )

    for data in tqdm(extract(filename, _embeddings)):
        pass

    parameters = AttrDict({"batch_size": 128, "padding": PAD_TOKEN}, num_buckets=100, fixed_length=None)
    for data in tqdm(etl(filename, _embeddings, params=parameters)):
        pass
