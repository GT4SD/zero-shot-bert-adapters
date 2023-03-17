import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import numpy as np
from os import listdir
from os.path import isfile, join

with open("BANKING77-OOS/id-oos/test/seq.in", "r") as seqs, open(
    "BANKING77-OOS/id-oos/test/label_original", "r"
) as labels:
    seq = seqs.readlines()
    label = labels.readlines()

with open("BANKING77-OOS/id-oos/test/test.csv", "w") as data:
    data.write("text,category\n")
    for s, l in zip(seq, label):
        data.write(s.strip().replace(",", "").replace(".", "") + "," + l.strip() + "\n")

df = pd.read_csv("BANKING77-OOS/id-oos/test/test.csv")
df = df.dropna()
df.to_csv("BANKING77-OOS/id-oos/test/test.csv", index=False)

dataset = load_dataset(
    "csv",
    data_files={
        "train": "BANKING77-OOS/id-oos/train/train.csv",
        "test": "BANKING77-OOS/id-oos/test/test.csv",
    },
    encoding="ISO-8859-1",
)

model = SentenceTransformer(
    "sentence-transformers/distiluse-base-multilingual-cased-v1"
)
sim_pred = []

filenames = [f for f in listdir("../results/") if isfile(join("../results/", f))]

for filename in filenames:
    df = pd.read_csv(filename, error_bad_lines=False)
    df = df.dropna()
    df = df.reset_index(drop=True)
    for i in tqdm(range(0, len(df))):
        label = dataset["test"][i]["category"].replace("_", " ")

        embedding_1 = model.encode(label, convert_to_tensor=True)
        embedding_2 = model.encode(
            " ".join(df["prediction"][i].split()[:2]), convert_to_tensor=True
        )

        sim_pred.append(util.pytorch_cos_sim(embedding_1, embedding_2).item())

    avg = np.mean(sim_pred)
    t = 0.5 if avg <= 0.5 else avg + 0.5 * np.var(sim_pred)

    count = 0
    for x in sim_pred:
        if x > t:
            count += 1
    print(filename + " score: " + str(count / len(sim_pred)))
