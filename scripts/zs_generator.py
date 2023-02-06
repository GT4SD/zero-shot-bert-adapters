import pandas as pd
import click
from tqdm import tqdm
from zs_model_generators import t0_generator, gpt_generator, keybert_generator


def find_model_class(model_name_or_path):

    if "t0" in model_name_or_path.lower():
        return t0_generator
    elif "gpt" in model_name_or_path.lower():
        return gpt_generator
    elif "keybert" in model_name_or_path.lower():
        return keybert_generator
    else:
        raise ValueError(f"{model_name_or_path} is not supported")


@click.command()
@click.option("--dataset", type=str, required=True, help="Dataset's path")
@click.option("--prompt", type=str, required=True, help="Prompt to be used.")
@click.option(
    "--model_name_or_path",
    type=str,
    required=True,
    help="Model to be used for the generation.",
)
@click.option(
    "--cache_dir", type=str, required=True, help="Cache directory for the model."
)
@click.option(
    "--output_path", type=str, required=True, help="Path to save the generations."
)
def main(dataset, prompt, model_name_or_path, cache_dir, output_path):

    model_class = find_model_class(model_name_or_path)
    model = model_class(model_name_or_path, cache_dir)

    data = pd.read_csv(dataset)

    prompts = []
    generations = []

    for _, case in tqdm(data.iterrows()):

        input_text = prompt.format(case["text"])
        prompts.append(prompt)

        generated = model.generate_text(input_text)

        generations.append(generated[0])

    df = pd.DataFrame.from_dict(
        {
            "utterance": data["text"].to_list(),
            "prompt": prompts,
            model_name_or_path: generations,
            "category": data["category"].to_list(),
        }
    )
    df.to_csv(output_path)


if __name__ == "__main__":
    main()
