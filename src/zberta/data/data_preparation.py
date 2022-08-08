from ..legacytorch.field import Field, LabelField
from ..legacytorch.dataset import TabularDataset
from ..legacytorch.iterator import BucketIterator
from ..utils.utils import init_tokenizer
from abc import ABC, abstractclassmethod


class DataPreparation(ABC):

    tokenizer = None
    max_input_length = None
    train_eval_split = None

    '''
    Personalize dataset creation
    '''
    def __init__(self, tokenizer, device, train_eval_split, batch_size):
        self.tokenizer = tokenizer if type(tokenizer) is not str else init_tokenizer(tokenizer)
        self.device = device
        self.train_eval_split = train_eval_split
        self.__class__.train_eval_split = self.train_eval_split
        self.batch_size = batch_size
        self.init_token_idx = self.tokenizer.cls_token_id
        self.eos_token_idx = self.tokenizer.sep_token_id
        self.pad_token_idx = self.tokenizer.pad_token_id
        self.unk_token_idx = self.tokenizer.unk_token_id
        self.init_token = self.tokenizer.cls_token
        self.eos_token = self.tokenizer.sep_token
        self.pad_token = self.tokenizer.pad_token
        self.unk_token = self.tokenizer.unk_token
        self.max_input_length = self.tokenizer.max_model_input_sizes['bert-base-uncased']
        self.__class__.tokenizer = self.tokenizer
        self.__class__.max_input_length = self.max_input_length
        self.LABEL = None

    @classmethod
    def tokenize_bert(cls, sentence):
        tokens = cls.tokenizer.tokenize(sentence)
        return tokens

    @classmethod
    def split_and_cut(cls, sentence):
        tokens = sentence.strip().split(" ")
        tokens = tokens[:cls.max_input_length-1]
        return tokens

    @classmethod
    def trim_sentence(cls, sent):
        try:
            sent = sent.split()
            sent = sent[:128]
            return " ".join(sent)
        except:
            return sent

    @classmethod
    def get_sent1_token_type(cls, sent):
        try:
            return [0]* len(sent)
        except:
            return []

    @classmethod
    def get_sent2_token_type(cls, sent):
        try:
            return [1]* len(sent)
        except:
            return []

    @classmethod
    def combine_seq(cls, seq):
        return " ".join(seq)

    @classmethod
    def combine_mask(cls, mask):
        mask = [str(m) for m in mask]
        return " ".join(mask)

    @classmethod
    @abstractclassmethod
    def _prepare_df(cls, df_train, df_test):
        pass

    def output_dim(self):
        if self.LABEL is not None:
            return len(self.LABEL.vocab)
        else:
            return 2
    
    def labels(self):
        if self.LABEL is not None:
            return len(self.LABEL.vocab.itos)
        else:
            return 2

    def nli_labels(self):
        return {"entailment": 0, "contradiction": 1} if self.LABEL is None \
            else [v for v in self.LABEL.vocab.itos.values()][0] if type(self.LABEL.vocab.itos) is dict \
            else dict(zip(self.LABEL.vocab.itos, [0, 1, 2]))

    '''
    Preparing dataset for NLI training through tokenization, mask creation and adding of BERT tokens [CLS], [SEP], ...
    '''
    @classmethod
    def _prepare_dataset(cls, df_train, df_test):
        df_train = df_train.sample(frac=1).reset_index(drop=True)

        df_train, df_dev, df_test = cls._prepare_df(df_train, df_test)

        df_train['sent1_t'] = df_train['sent1'].apply(cls.tokenize_bert)
        df_train['sent2_t'] = df_train['sent2'].apply(cls.tokenize_bert)
        df_dev['sent1_t'] = df_dev['sent1'].apply(cls.tokenize_bert)
        df_dev['sent2_t'] = df_dev['sent2'].apply(cls.tokenize_bert)
        df_test['sent1_t'] = df_test['sent1'].apply(cls.tokenize_bert)
        df_test['sent2_t'] = df_test['sent2'].apply(cls.tokenize_bert)

        df_train['sent1_token_type'] = df_train['sent1_t'].apply(cls.get_sent1_token_type)
        df_train['sent2_token_type'] = df_train['sent2_t'].apply(cls.get_sent2_token_type)
        df_dev['sent1_token_type'] = df_dev['sent1_t'].apply(cls.get_sent1_token_type)
        df_dev['sent2_token_type'] = df_dev['sent2_t'].apply(cls.get_sent2_token_type)
        df_test['sent1_token_type'] = df_test['sent1_t'].apply(cls.get_sent1_token_type)
        df_test['sent2_token_type'] = df_test['sent2_t'].apply(cls.get_sent2_token_type)

        df_train['sequence'] = df_train['sent1_t'] + df_train['sent2_t']
        df_dev['sequence'] = df_dev['sent1_t'] + df_dev['sent2_t']
        df_test['sequence'] = df_test['sent1_t'] + df_test['sent2_t']

        df_train['attention_mask'] = df_train['sequence'].apply(cls.get_sent2_token_type)
        df_dev['attention_mask'] = df_dev['sequence'].apply(cls.get_sent2_token_type)
        df_test['attention_mask'] = df_test['sequence'].apply(cls.get_sent2_token_type)

        df_train['token_type'] = df_train['sent1_token_type'] + df_train['sent2_token_type']
        df_dev['token_type'] = df_dev['sent1_token_type'] + df_dev['sent2_token_type']
        df_test['token_type'] = df_test['sent1_token_type'] + df_test['sent2_token_type']

        df_train['sequence'] = df_train['sequence'].apply(cls.combine_seq)
        df_dev['sequence'] = df_dev['sequence'].apply(cls.combine_seq)
        df_test['sequence'] = df_test['sequence'].apply(cls.combine_seq)

        df_train['attention_mask'] = df_train['attention_mask'].apply(cls.combine_mask)
        df_dev['attention_mask'] = df_dev['attention_mask'].apply(cls.combine_mask)
        df_test['attention_mask'] = df_test['attention_mask'].apply(cls.combine_mask)

        df_train['token_type'] = df_train['token_type'].apply(cls.combine_mask)
        df_dev['token_type'] = df_dev['token_type'].apply(cls.combine_mask)
        df_test['token_type'] = df_test['token_type'].apply(cls.combine_mask)

        df_train = df_train[['gold_label', 'sequence', 'attention_mask', 'token_type']]
        df_dev = df_dev[['gold_label', 'sequence', 'attention_mask', 'token_type']]
        df_test = df_test[['gold_label', 'sequence', 'attention_mask', 'token_type']]

        df_train = df_train.loc[df_train['gold_label'].isin(['entailment','contradiction','neutral'])]
        df_dev = df_dev.loc[df_dev['gold_label'].isin(['entailment','contradiction','neutral'])]
        df_test = df_test.loc[df_test['gold_label'].isin(['entailment','contradiction','neutral'])]

        df_train.to_csv('train.csv', index=False)
        df_dev.to_csv('dev.csv', index=False)
        df_test.to_csv('test.csv', index=False)

        return df_train, df_dev, df_test

    def convert_to_int(self, tok_ids):
        tok_ids = [int(x) for x in tok_ids]
        return tok_ids

    def create_iterator(self, df_train, df_test):
        df_train, df_dev, df_test = self._prepare_dataset(df_train, df_test)

        TEXT = Field(batch_first = True,
                        use_vocab = False,
                        tokenize = self.split_and_cut,
                        preprocessing = self.tokenizer.convert_tokens_to_ids,
                        pad_token = self.pad_token_idx,
                        unk_token = self.unk_token_idx)

        self.LABEL = LabelField()

        ATTENTION = Field(batch_first = True,
                        use_vocab = False,
                        tokenize = self.split_and_cut,
                        preprocessing = self.convert_to_int,
                        pad_token = self.pad_token_idx)

        TTYPE = Field(batch_first = True,
                        use_vocab = False,
                        tokenize = self.split_and_cut,
                        preprocessing = self.convert_to_int,
                        pad_token = 1)
        
        fields = [('label', self.LABEL), ('sequence', TEXT), ('attention_mask', ATTENTION), ('token_type', TTYPE)]

        train_data, valid_data, test_data = TabularDataset.splits(
                                                path ='',
                                                train = 'train.csv',
                                                validation = 'dev.csv',
                                                test = 'test.csv',
                                                format = 'csv',
                                                fields = fields,
                                                skip_header = True)

        self.LABEL.build_vocab(train_data)

        train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
            (train_data, valid_data, test_data), 
            batch_size = self.batch_size,
            sort_key = lambda x: len(x.sequence),
            sort_within_batch = False, 
            device = self.device)
        
        return train_iterator, valid_iterator, test_iterator
