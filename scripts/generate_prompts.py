import os
from typing import List
from urllib.request import urlopen


def get_prompts(url: str) -> List[str]:
    cases = []
    with urlopen(url) as response:
        for line in response.readlines():
            line = line.strip().decode()
            if line:
                cases.append(line.title())
    return cases


if __name__ == "__main__":
    print(
        os.linesep.join(
            get_prompts(
                "https://raw.githubusercontent.com/jianguoz/Few-Shot-Intent-Detection/main/Datasets/BANKING77-OOS/id-oos/test/seq.in"
            )
        )
        + os.linesep
    )
