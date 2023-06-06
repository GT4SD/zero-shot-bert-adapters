# Zero-shot-BERT-adapters
Repository containing code for Zero-shot-BERT-adapters (Z-BERT-A), a two-stage method for multilingual intent discovery relying on a Transformer architecture, fine-tuned with Adapters, which is initially trained for Natural Language Inference (NLI) and later applied for unknown intent classification in a zero-shot setting.

<img src="https://user-images.githubusercontent.com/6382701/195600358-24c3f136-96f0-4a8a-8eaf-220970ca0604.png" width=523 height=466>

## Deployment example of an use-case of Zero-shot-BERT-adapters pipeline

![image](https://i.ibb.co/RTrm66c/image.png)



## Important information
This repository makes use of the module [Adapter Transformers](https://github.com/adapter-hub/adapter-transformers).<br>
The [wget](https://www.gnu.org/software/wget/) tool is required for this package, for Windows users please make sure to install it accordingly before running this module.

## Installation
In order to install Zero-shot-BERT-adapters it is just needed to execute the following pip command which will make sure everything is accordingly installed.
Zero-shot-BERT-adapters uses spaCy with 'en_core_web_trf', this package will try to install it automatically itself if not present but make sure you have it installed through the suggested way of [spaCy](https://github.com/explosion/spaCy).

```console
pip install git+https://github.com/GT4SD/zero-shot-bert-adapters.git
```

## Usage information

In order to reproduce the results here there is the sample code which can also be found in an example [Jupyter Notebook](https://colab.research.google.com/drive/1k-MqbwbU870wGlpcSbaRNV3sx7h-rW9_?usp=sharing).

```python
import torch
from zberta.data.data_snli import DataSNLI
from zberta.data.data_banking import DataBanking
import zberta.model.model as model
from zberta.train.trainer import Trainer
from zberta.intent_discovery.unknown_intents import unknown_intents_set
from zberta.intent_discovery.zberta import ZBERTA

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
z_intents = unknown_intents_set(z_dataset['test']['text'])
berta = model.instantiate_model(z_banking.labels(), z_banking.output_dim(), device, model_name,
                                z_banking.nli_labels(), path="model.pt", dict=True)
zberta = ZBERTA(berta, model_name, z_dataset['test']['text'], z_intents, z_dataset['test']['category'])
z_acc = zberta.zero_shot_intents()
print(z_acc)
```

For simple usage of the Zero-shot-BERT-adapters pipeline instead it's just needed to load the pipeline and model through this simple code which encapsulate all the complexity behind.

```python
import torch
from zberta.intent_discovery.zberta import ZBERTA
import zberta.model.model as model

model_name = "bert-base-uncased"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
berta = model.instantiate_model(2, 2, device, model_name,
                                {"entailment": 0, "contradiction": 1}, path="model.pt", dict=True)
zberta = ZBERTA(berta, model_name)
```

As example run:

```python
zberta.find_new_intents(["I want to buy a book but I lost all my money, where can I make a withdrawal?"])
```
```
Output: ['make withdrawal']
```
