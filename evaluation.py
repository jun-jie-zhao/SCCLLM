from nlgeval import compute_metrics
import sys
from io import StringIO
from rouge import Rouge
import statistics
from nltk.translate import bleu_score


output_buffer = StringIO()
sys.stdout = output_buffer

#计算所有指标
compute_metrics(hypothesis='baseline/SCCLLM.csv',
                references=['baseline/nl.csv'],
                no_skipthoughts=True,
                no_glove=True)
sys.stdout = sys.__stdout__

output = output_buffer.getvalue()
rouge_l_result = None
lines = output.split('\n')
for line in lines:
    if line.startswith('ROUGE_L:'):
        rouge_l_result = line

def read_csv(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

reference_file_path = 'baseline/nl.csv'  # Replace it with your own path
generated_file_path = 'baseline/SCCLLM.csv'  # Replace it with your own path

reference_annotations = read_csv(reference_file_path)
generated_annotations = read_csv(generated_file_path)

# 初始化 ROUGE 对象
rouge = Rouge()

# 计算 ROUGE-1 和 ROUGE-2 分数
scores = rouge.get_scores(generated_annotations, reference_annotations, avg=True)

# 输出分数
avg_rouge1 = scores['rouge-1']['f']
avg_rouge2 = scores['rouge-2']['f']


chencherry = bleu_score.SmoothingFunction()

# Evaluate perfect prediction & BLEU score of our approach

k=1
path_targets = 'baseline/nl.csv'
path_predictions = 'baseline/SCCLLM.csv'

tgt = [line.strip() for line in open(path_targets)]
pred = [line.strip() for line in open(path_predictions)]



count_perfect = 0
BLEUscore = []
for i in range(len(tgt)):
        best_BLEU = 0
        target = tgt[i]
        for prediction in pred[i * k:i * k + k]:
            if " ".join(prediction.split()) == " ".join(target.split()):
                count_perfect += 1
                best_BLEU = bleu_score.sentence_bleu([target], prediction, smoothing_function=chencherry.method1)
                break
            current_BLEU = bleu_score.sentence_bleu([target], prediction, smoothing_function=chencherry.method1)
            if current_BLEU > best_BLEU:
                best_BLEU = current_BLEU
        BLEUscore.append(best_BLEU)
print(f'BLEU: ', statistics.mean(BLEUscore))
print("ROUGE-1:", avg_rouge1)
print("ROUGE-2:", avg_rouge2)
print(rouge_l_result)


