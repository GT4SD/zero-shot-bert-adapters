import pandas as pd
import ast
import click
from transformers import pipeline
from tqdm import tqdm


@click.command()
@click.option("--datafile", type=str, required=True, help="Path to the datafile")
@click.option(
    "--model_name_or_path",
    type=str,
    default=None,
    help="Model to be used for the generation.",
)
@click.option(
    "--output_path", type=str, required=True, help="Path to save the generations."
)
def execute(datafile, output_path, model_name_or_path=None):

    if model_name_or_path is not None:
        classifier = pipeline("zero-shot-classification", model=model_name_or_path)
    else:
        classifier = pipeline("zero-shot-classification")

    data = pd.read_csv(datafile)
    data = data.to_dict("records")

    utterances = []
    classes = []
    generations = []

    for case in tqdm(data):

        labels = ast.literal_eval(case["classes"])

        utterances.append(case["utterance"])
        classes.append(labels)

        res = classifier(case["utterance"], labels)

        label = "Unknown"
        if res["scores"][0] > 0.5:
            label = res["labels"][0]

        generations.append(label)

    df = pd.DataFrame.from_dict(
        {"utterance": utterances, "classes": classes, "snli": generations}
    )
    df.to_csv(output_path)


if __name__ == "__main__":
    execute()
