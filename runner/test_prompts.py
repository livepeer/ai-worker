from app.pipelines.utils.utils import split_prompt

if __name__ == "__main__":
    input_prompt = "A photo of a cat.|"
    test = split_prompt(input_prompt)

    input_prompt2 = ""
    test2 = split_prompt(input_prompt2)

    input_pormpt3 = "A photo of a cat.|A photo of a dog.|A photo of a bird."
    test3 = split_prompt(input_pormpt3)

