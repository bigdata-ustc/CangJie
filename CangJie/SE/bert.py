# coding: utf-8
# 2020/1/8 @ tongshiwei
from mxnet.gluon.data import Dataset
import logging
import gluonnlp
import mxnet as mx
import os
from mxnet.gluon.data import DataLoader
from gluonnlp.data import BERTTokenizer, BERTSentenceTransform, BERTSPTokenizer

logger = logging


def get_home_dir():
    """Get home directory for storing datasets/models/pre-trained word embeddings"""
    _home_dir = os.environ.get('MXNET_HOME', os.path.join('~', '.mxnet'))
    # expand ~ to actual path
    _home_dir = os.path.expanduser(_home_dir)
    return _home_dir


class BertEmbedding(object):
    """
    Encoding from BERT model.

    Parameters
    ----------
    ctx : Context.
        running BertEmbedding on which gpu device id.
    dtype: str
        data type to use for the model.
    model : str, default bert_12_768_12.
        pre-trained BERT model
    dataset_name : str, default book_corpus_wiki_en_uncased.
        pre-trained model dataset
    params_path: str, default None
        path to a parameters file to load instead of the pretrained model.
    max_seq_length : int, default 25
        max length of each sequence
    batch_size : int, default 256
        batch size
    sentencepiece : str, default None
        Path to the sentencepiece .model file for both tokenization and vocab
    root : str, default '$MXNET_HOME/models' with MXNET_HOME defaults to '~/.mxnet'
        Location for keeping the model parameters.
    """

    def __init__(self, ctx=mx.cpu(), dtype='float32', model='bert_12_768_12',
                 dataset_name='wiki_cn_cased', params_path=None,
                 max_seq_length=25, batch_size=256, sentencepiece=None,
                 root=os.path.join(get_home_dir(), 'models')):
        self.ctx = ctx
        self.dtype = dtype
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.dataset_name = dataset_name

        # use sentencepiece vocab and a checkpoint
        # we need to set dataset_name to None, otherwise it uses the downloaded vocab
        if params_path and sentencepiece:
            dataset_name = None
        else:
            dataset_name = self.dataset_name
        if sentencepiece:
            vocab = gluonnlp.vocab.BERTVocab.from_sentencepiece(sentencepiece)
        else:
            vocab = None
        self.bert, self.vocab = gluonnlp.model.get_model(model,
                                                         dataset_name=dataset_name,
                                                         pretrained=params_path is None,
                                                         ctx=self.ctx,
                                                         use_pooler=False,
                                                         use_decoder=False,
                                                         use_classifier=False,
                                                         root=root, vocab=vocab)

        self.bert.cast(self.dtype)
        if params_path:
            logger.info('Loading params from %s', params_path)
            self.bert.load_parameters(params_path, ctx=ctx, ignore_extra=True, cast_dtype=True)

        lower = 'uncased' in self.dataset_name
        if sentencepiece:
            self.tokenizer = BERTSPTokenizer(sentencepiece, self.vocab, lower=lower)
        else:
            self.tokenizer = BERTTokenizer(self.vocab, lower=lower)
        self.transform = BERTSentenceTransform(tokenizer=self.tokenizer,
                                               max_seq_length=self.max_seq_length,
                                               pair=False)

    def cls(self, sentence):
        transform = self.transform
        model = self.bert

        sample = transform((sentence,))
        words, valid_len, segments = mx.nd.array([sample[0]]), mx.nd.array([sample[1]]), mx.nd.array([sample[2]])
        seq_encoding, cls_encoding = model(words, segments, valid_len)
        return cls_encoding

    def __call__(self, sentences, oov_way='avg'):
        return self.embedding(sentences, oov_way='avg')

    def embedding(self, sentences, oov_way='avg'):
        """
        Get tokens, tokens embedding

        Parameters
        ----------
        sentences : List[str]
            sentences for encoding.
        oov_way : str, default avg.
            use **avg**, **sum** or **last** to get token embedding for those out of
            vocabulary words

        Returns
        -------
        List[(List[str], List[ndarray])]
            List of tokens, and tokens embedding
        """
        data_iter = self.data_loader(sentences=sentences)
        batches = []
        for token_ids, valid_length, token_types in data_iter:
            token_ids = token_ids.as_in_context(self.ctx)
            valid_length = valid_length.as_in_context(self.ctx)
            token_types = token_types.as_in_context(self.ctx)
            sequence_outputs = self.bert(token_ids, token_types,
                                         valid_length.astype(self.dtype))
            for token_id, sequence_output in zip(token_ids.asnumpy(),
                                                 sequence_outputs.asnumpy()):
                batches.append((token_id, sequence_output))
        return self.oov(batches, oov_way)

    def data_loader(self, sentences, shuffle=False):
        """Load, tokenize and prepare the input sentences."""
        dataset = BertEmbeddingDataset(sentences, self.transform)
        return DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=shuffle)

    def oov(self, batches, oov_way='avg'):
        """
        How to handle oov. Also filter out [CLS], [SEP] tokens.

        Parameters
        ----------
        batches : List[(tokens_id, sequence_outputs)].
            batch   token_ids shape is (max_seq_length,),
                    sequence_outputs shape is (max_seq_length, dim)
        oov_way : str
            use **avg**, **sum** or **last** to get token embedding for those out of
            vocabulary words

        Returns
        -------
        List[(List[str], List[ndarray])]
            List of tokens, and tokens embedding
        """
        sentences = []
        padding_idx, cls_idx, sep_idx = None, None, None
        if self.vocab.padding_token:
            padding_idx = self.vocab[self.vocab.padding_token]
        if self.vocab.cls_token:
            cls_idx = self.vocab[self.vocab.cls_token]
        if self.vocab.sep_token:
            sep_idx = self.vocab[self.vocab.sep_token]
        for token_ids, sequence_outputs in batches:
            tokens = []
            tensors = []
            oov_len = 1
            for token_id, sequence_output in zip(token_ids, sequence_outputs):
                # [PAD] token, sequence is finished.
                if padding_idx and token_id == padding_idx:
                    break
                # [CLS], [SEP]
                if cls_idx and token_id == cls_idx:
                    continue
                if sep_idx and token_id == sep_idx:
                    continue
                token = self.vocab.idx_to_token[token_id]
                if not self.tokenizer.is_first_subword(token):
                    tokens.append(token)
                    if oov_way == 'last':
                        tensors[-1] = sequence_output
                    else:
                        tensors[-1] += sequence_output
                    if oov_way == 'avg':
                        oov_len += 1
                else:  # iv, avg last oov
                    if oov_len > 1:
                        tensors[-1] /= oov_len
                        oov_len = 1
                    tokens.append(token)
                    tensors.append(sequence_output)
            if oov_len > 1:  # if the whole sentence is one oov, handle this special case
                tensors[-1] /= oov_len
            sentences.append((tokens, tensors))
        return sentences


class BertEmbeddingDataset(Dataset):
    """Dataset for BERT Embedding

    Parameters
    ----------
    sentences : List[str].
        Sentences for embeddings.
    transform : BERTDatasetTransform, default None.
        transformer for BERT input format
    """

    def __init__(self, sentences, transform=None):
        """Dataset for BERT Embedding

        Parameters
        ----------
        sentences : List[str].
            Sentences for embeddings.
        transform : BERTDatasetTransform, default None.
            transformer for BERT input format
        """
        self.sentences = sentences
        self.transform = transform

    def __getitem__(self, idx):
        sentence = (self.sentences[idx], 0)
        if self.transform:
            return self.transform(sentence)
        else:
            return sentence

    def __len__(self):
        return len(self.sentences)


class CharBert(object):
    def __init__(self, ctx=mx.cpu()):
        model, vocab = gluonnlp.model.get_model('bert_12_768_12', dataset_name='wiki_cn_cased',
                                                use_classifier=False, use_decoder=False, ctx=ctx)
        tokenizer = gluonnlp.data.BERTTokenizer(vocab, lower=True)
        transform = gluonnlp.data.BERTSentenceTransform(tokenizer, max_seq_length=512, pair=False, pad=False)
        self.tokenizer = tokenizer
        self.model = model
        self.transform = transform
        self.ctx = ctx

    def __call__(self, sentence):
        transform = self.transform
        model = self.model

        sample = transform((sentence,))
        words, valid_len = mx.nd.array([sample[0]], ctx=self.ctx), mx.nd.array([sample[1]], ctx=self.ctx)
        segments = mx.nd.array([sample[2]], ctx=self.ctx)
        seq_encoding, cls_encoding = model(words, segments, valid_len)
        return seq_encoding, cls_encoding

    def tokenize(self, sentence):
        return self.tokenizer(sentence)


if __name__ == '__main__':
    # wb = BertEmbedding()
    # print(wb(["天安门"]))
    # print(wb.cls("再给我两分钟"))
    cb = CharBert()
    print(cb("再给我两分钟"))
    print(cb.tokenize("再给我两分钟"))
