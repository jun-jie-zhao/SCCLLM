import openai
import pandas as pd
from tqdm import tqdm
import random
import time
import math

random.seed(1234)
openai.api_base = 'https://xxxxxxx'
openai.api_key = 'sk-xxxxxxxxxx'

def comment(example_source1, example_target1, example_source2,example_target2, example_source3, example_target3,example_source4, example_target4,example_source5,example_target5, code, example_choose):
    prompt = '#example code 1:' + example_source1 + \
             '\n#example summarization 1:' + example_target1 + \
             '\n#example code 2:' + example_source2 + \
             '\n#example summarization 2:' + example_target2 + \
             '\n#example code 3:' + example_source3 + \
             '\n#example summarization 3:' + example_target3 + \
             '\n#example code 4:' + example_source4 + \
             '\n#example summarization 4:' + example_target4 + \
             '\n#example code 5:' + example_source5 + \
             '\n#example summarization 5:' + example_target5 + \
             '\n#a smart contract code:' + code + \
             '\n#Generated summarization(The length should not exceed ['+ example_choose +']):\n'
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "To generate a short summarization in one sentence for smart contract code.To alleviate the difficulty of this task, I will give you five examples.Please learn from them."},
            {"role": "user", "content": prompt},
        ],
        temperature=0
    )
    return response['choices'][0]['message']['content']


if __name__ == '__main__':
    df = pd.read_csv('data/example_all.csv')
    example_code1 = df['code1'].tolist()
    df = pd.read_csv('data/example_all.csv')
    example_comment1 = df['comment1'].tolist()
    df = pd.read_csv('data/example_all.csv')
    example_code2 = df['code2'].tolist()
    df = pd.read_csv('data/example_all.csv')
    example_comment2 = df['comment2'].tolist()
    df = pd.read_csv('data/example_all.csv')
    example_code3 = df['code3'].tolist()
    df = pd.read_csv('data/example_all.csv')
    example_comment3 = df['comment3'].tolist()
    df = pd.read_csv('data/example_all.csv')
    example_code4 = df['code4'].tolist()
    df = pd.read_csv('data/example_all.csv')
    example_comment4 = df['comment4'].tolist()
    df = pd.read_csv('data/example_all.csv')
    example_code5 = df['code5'].tolist()
    df = pd.read_csv('data/example_all.csv')
    example_comment5 = df['comment5'].tolist()

    #df = pd.read_csv('dataset/number2.csv', header=None)
    #number = df[0].tolist()

    df = pd.read_csv('data/test_base_function.csv', header=None)
    source_codes = df[0].tolist()

    df = pd.read_csv('dataset/Csim.csv', header=None)
    example = df[0].tolist()

    num_batches = math.ceil(len(source_codes) / 100)
    for batch_index in range(num_batches):
        start_index = batch_index * 100
        end_index = min(start_index + 100, len(source_codes))

        source_batch = source_codes[start_index:end_index]
        example_batch = example[start_index:end_index]

        python_codes = []
        for i in tqdm(range(len(source_batch))):
            python_codes.append(
                comment(example_code1[i], example_comment1[i], example_code2[i], example_comment2[i], example_code3[i],
                        example_comment3[i], example_code4[i], example_comment4[i], example_code5[i],
                        example_comment5[i], source_batch[i], example_batch[i]))
            time.sleep(2)

        df = pd.DataFrame(python_codes)
        if batch_index == 0:
            df.to_csv('result/sml.csv', index=False, header=None, line_terminator='\n')
        else:
            with open('result/sml.csv', 'a') as f:
                df.to_csv(f, index=False, header=None, line_terminator='\n')
