# %%
import csv


def get_accuracy(filename):
    num_correct = 0
    num_total = 0

    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for idx, row in enumerate(reader):
            if idx>0:
                p_true = row[1]
                p_false = row[2]
                label = int(float(row[3]))

                pred = p_true > p_false
                correct = (pred == label) #bool

                num_correct += correct
                num_total += 1
    acc = num_correct / num_total
    print(acc)


get_accuracy('inference_output_honest.csv')
get_accuracy('inference_output_liar.csv')
# %%
