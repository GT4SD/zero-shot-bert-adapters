from src.zberta.data.data_snli import DataSNLI
from src.zberta.data.data_banking import DataBanking
import src.zberta.model.model as model
from src.zberta.train.trainer import Trainer
import torch
from src.zberta.intent_discovery.unknown_intents import unknown_intents_set
from src.zberta.intent_discovery.zberta import ZBERTA

if __name__ == '__main__':
    model_name = "bert-base-uncased"
    training = False
    testing = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if training:
        snli = DataSNLI(model_name, device)
        train_iterator, valid_iterator, test_iterator = snli.iterators()
        berta = model.instantiate_model(snli.labels(), snli.output_dim(), device, model_name, snli.nli_labels())
        trainer = Trainer(berta, train_iterator, valid_iterator, test_iterator, device)
        trainer.start_training()
        if testing:
            trainer.start_testing()
    z_banking = DataBanking(model_name, device)
    z_dataset = z_banking.z_iterator()
    z_intents = unknown_intents_set("en_core_web_trf", z_dataset['test']['text'])
    berta = model.instantiate_model(z_banking.labels(), z_banking.output_dim(), device, model_name,
                                    z_banking.nli_labels(), path="model.pt", dict=True)
    zberta = ZBERTA(berta, model_name, z_dataset['test']['text'], z_intents, z_dataset['test']['category'])
    z_acc = zberta.zero_shot_intents()
    print(z_acc)
