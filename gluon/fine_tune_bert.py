import warnings
warnings.filterwarnings('ignore')

import io
import random
import numpy as np
import mxnet as mx
import gluonnlp as nlp
from gluonnlp.calibration import BertLayerCollector
from sentence_embedding.bert import data

nlp.utils.check_version('0.8.1')

class FineTuneBert:
    np.random.seed(100)
    random.seed(100)
    mx.random.seed(10000)
    # change `ctx` to `mx.cpu()` if no GPU is available.
    # ctx = mx.gpu(0)
    ctx = mx.cpu(0)

    def preProcess(self):

        # Get Bert

        # include the pooler layer of the pre-trained model by setting use_pooler to True
        bert_base, vocabulary = nlp.model.get_model('bert_12_768_12',
                                                    dataset_name='book_corpus_wiki_en_uncased',
                                                    pretrained=True, ctx=self.ctx, use_pooler=True,
                                                    use_decoder=False, use_classifier=False)
        # print(bert_base)

        # Transform the model for SentencePair classification

        # The BERTClassifier class uses a BERT base model to encode sentence representation, followed by a nn.Dense layer for classification.
        self.bert_classifier = nlp.model.BERTClassifier(bert_base, num_classes=2, dropout=0.1)
        # only need to initialize the classifier layer.
        self.bert_classifier.classifier.initialize(init=mx.init.Normal(0.02), ctx=self.ctx)
        self.bert_classifier.hybridize(static_alloc=True)

        # softmax cross entropy loss for classification
        self.loss_function = mx.gluon.loss.SoftmaxCELoss()
        self.loss_function.hybridize(static_alloc=True)

        self.metric = mx.metric.Accuracy()

        # Loading the dataset

        # tsv_file = io.open('sentence_embedding/dev.tsv', encoding='utf-8')

        # ﻿Quality	#1 ID	#2 ID	#1 String	#2 String
        # 1	1355540	1355592	He said the foodservice pie business doesn 't fit the company 's long-term growth strategy .	" The foodservice pie business does not fit our long-term growth strategy .
        # for i in range(5):
        #     print(tsv_file.readline())

        # Skip the first line, which is the schema
        num_discard_samples = 1
        # Split fields by tabs
        field_separator = nlp.data.Splitter('\t')
        # Fields to select from the file
        field_indices = [3, 4, 0]
        self.data_train_raw = nlp.data.TSVDataset(filename='sentence_embedding/dev.tsv',
                                             field_separator=field_separator,
                                             num_discard_samples=num_discard_samples,
                                             field_indices=field_indices)
        sample_id = 0
        # Sentence A
        print(self.data_train_raw[sample_id][0])
        # Sentence B
        print(self.data_train_raw[sample_id][1])
        # 1 means equivalent, 0 means not equivalent
        print(self.data_train_raw[sample_id][2])

        #  tokenize the input sequences - insert [CLS] at the beginning - insert [SEP] between sentence A and sentence B, and at the end - generate segment ids to indicate whether a token belongs to the first sequence or the second sequence. - generate valid length

        # Use the vocabulary from pre-trained model for tokenization
        bert_tokenizer = nlp.data.BERTTokenizer(vocabulary, lower=True)

        # The maximum length of an input sequence
        max_len = 128

        # The labels for the two classes [(0 = not similar) or  (1 = similar)]
        all_labels = ["0", "1"]

        # whether to transform the data as sentence pairs.
        # for single sentence classification, set pair=False
        # for regression task, set class_labels=None
        # for inference without label available, set has_label=False
        pair = True
        transform = data.transform.BERTDatasetTransform(bert_tokenizer, max_len,
                                                        class_labels=all_labels,
                                                        has_label=True,
                                                        pad=True,
                                                        pair=pair)
        self.data_train = self.data_train_raw.transform(transform)

        print('vocabulary used for tokenization = \n%s' % vocabulary)
        print('%s token id = %s' % (vocabulary.padding_token, vocabulary[vocabulary.padding_token]))
        print('%s token id = %s' % (vocabulary.cls_token, vocabulary[vocabulary.cls_token]))
        print('%s token id = %s' % (vocabulary.sep_token, vocabulary[vocabulary.sep_token]))
        print('token ids = \n%s' % self.data_train[sample_id][0])
        print('segment ids = \n%s' % self.data_train[sample_id][1])
        print('valid length = \n%s' % self.data_train[sample_id][2])
        print('label = \n%s' % self.data_train[sample_id][3])

    # use a fixed learning rate and skip the validation steps. For the optimizer, we leverage the ADAM optimizer which performs very well for NLP data and for BERT models in particular.
    def fineTune(self):
        batch_size = 32
        lr = 5e-6

        # The FixedBucketSampler and the DataLoader for making the mini-batches
        #   Assign each data sample to a fixed bucket based on its length.
        #   The bucket keys are either given or generated from the input sequence lengths
        #   example can be found from sampler_test.py
        train_sampler = nlp.data.FixedBucketSampler(lengths=[int(item[2]) for item in self.data_train],
                                                    batch_size=batch_size,
                                                    shuffle=True)
        bert_dataloader = mx.gluon.data.DataLoader(self.data_train, batch_sampler=train_sampler)

        trainer = mx.gluon.Trainer(self.bert_classifier.collect_params(), 'adam',
                                   {'learning_rate': lr, 'epsilon': 1e-9})

        # Collect all differentiable parameters
        # `grad_req == 'null'` indicates no gradients are calculated (e.g. constant parameters)
        # The gradients for these params are clipped later
        params = [p for p in self.bert_classifier.collect_params().values() if p.grad_req != 'null']
        grad_clip = 1

        # Training the model with only three epochs
        log_interval = 4
        num_epochs = 3
        for epoch_id in range(num_epochs):
            self.metric.reset()
            step_loss = 0
            for batch_id, (token_ids, segment_ids, valid_length, label) in enumerate(bert_dataloader):
                with mx.autograd.record():

                    # Load the data to the CPU
                    token_ids = token_ids.as_in_context(self.ctx)
                    valid_length = valid_length.as_in_context(self.ctx)
                    segment_ids = segment_ids.as_in_context(self.ctx)
                    label = label.as_in_context(self.ctx)

                    # Forward computation
                    out = self.bert_classifier(token_ids, segment_ids, valid_length.astype('float32'))
                    ls = self.loss_function(out, label).mean()

                # And backwards computation
                ls.backward()

                # Gradient clipping
                trainer.allreduce_grads()
                nlp.utils.clip_grad_global_norm(params, 1)
                trainer.update(1)

                step_loss += ls.asscalar()
                self.metric.update([label], [out])

                # Printing vital information
                if (batch_id + 1) % (log_interval) == 0:
                    print('[Epoch {} Batch {}/{}] loss={:.4f}, lr={:.7f}, acc={:.3f}'
                          .format(epoch_id, batch_id + 1, len(bert_dataloader),
                                  step_loss / log_interval,
                                  trainer.learning_rate, self.metric.get()[1]))
                    step_loss = 0

if __name__ == "__main__":
    nlp.utils.check_version('0.8.1')

    # test import correctly
    # print(len(data.embedding.BertEmbeddingDataset("asdf asdf")))

    fineTuneBert = FineTuneBert()

    fineTuneBert.preProcess()
    # fineTuneBert.fineTune()
