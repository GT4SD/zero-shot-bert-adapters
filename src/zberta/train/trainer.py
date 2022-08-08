import torch
import torch.nn as nn
import transformers
from tqdm import tqdm
import math
import time


class Trainer:

    def __init__(self, model, train_iterator, valid_iterator, test_iterator, device, fp16=False, max_grad_norm=1,
                 epochs=6, batch=8):
        self.model = model
        self.train_iterator = train_iterator
        self.valid_iterator = valid_iterator
        self.test_iterator = test_iterator
        self.device = device
        self.fp16 = fp16
        self.max_grad_norm = max_grad_norm
        self.N_EPOCHS = epochs
        self.BATCH_SIZE = batch
        self.warmup_percent = 0.2
        self.train_data_len = len(self.train_iterator)

        self.total_steps = math.ceil(self.N_EPOCHS * self.train_data_len * 1. / self.BATCH_SIZE)
        self.warmup_steps = int(self.total_steps * self.warmup_percent)
        self.optimizer = transformers.AdamW(model.parameters(), lr=2e-5, eps=1e-6, correct_bias=False)
        self.scheduler = transformers.get_constant_schedule_with_warmup(self.optimizer,
                                                                        num_warmup_steps=self.warmup_steps)
        self.criterion = nn.CrossEntropyLoss().to(device)

        if fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1')

    def categorical_accuracy(self, preds, y):
        max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
        correct = (max_preds.squeeze(1) == y).float()
        return correct.sum() / len(y)

    def train(self):
        epoch_loss = 0
        epoch_acc = 0

        self.model.train()
        with tqdm(total=len(self.train_iterator)) as progress_bar:
            for batch in self.train_iterator:

                self.optimizer.zero_grad()  # clear gradients first
                torch.cuda.empty_cache()  # releases all unoccupied cached memory

                sequence = batch.sequence
                attn_mask = batch.attention_mask
                token_type = batch.token_type
                label = batch.label

                predictions = self.model(sequence, attn_mask, token_type)

                loss = self.criterion(predictions, label)

                acc = self.categorical_accuracy(predictions, label)

                if self.fp16:
                    try:
                        from apex import amp
                    except ImportError:
                        raise ImportError(
                            "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.max_grad_norm)
                else:
                    loss.backward()

                self.optimizer.step()
                self.scheduler.step()

                epoch_loss += loss.item()
                epoch_acc += acc.item()
                progress_bar.update(1)

            print("Epoch result:")
            print("Loss: " + str(epoch_loss))
            print("Accuracy: " + str(epoch_acc))
            print("-----------")

        return epoch_loss / len(self.train_iterator), epoch_acc / len(self.train_iterator)

    def evaluate(self, iterator):
        epoch_loss = 0
        epoch_acc = 0

        self.model.eval()

        with torch.no_grad():
            with tqdm(total=len(iterator)) as progress_bar:
                for batch in iterator:
                    sequence = batch.sequence
                    attn_mask = batch.attention_mask
                    token_type = batch.token_type
                    labels = batch.label

                    predictions = self.model(sequence, attn_mask, token_type)

                    loss = self.criterion(predictions, labels)

                    acc = self.categorical_accuracy(predictions, labels)

                    epoch_loss += loss.item()
                    epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def start_training(self):
        best_valid_loss = float('inf')

        for epoch in range(self.N_EPOCHS):

            start_time = time.time()

            train_loss, train_acc = self.train()
            valid_loss, valid_acc = self.evaluate(self.valid_iterator)

            end_time = time.time()

            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), 'bert-nli.pt')
                print("New model version saved!")

            print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

            return (train_loss, train_acc), (valid_loss, valid_acc)

    def start_testing(self, path="bert-nli.pt", load=True):
        if load:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        test_loss, test_acc = self.evaluate(self.test_iterator)
        print(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc * 100:.2f}%')
        return test_loss, test_acc
