# %%
import csv
import matplotlib.pyplot as plt


def get_accuracy(filename, threshold=0):
    num_correct = 0
    num_total = 0
    acc = 0
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for idx, row in enumerate(reader):
            if idx>0:
                p_true = float(row[1])
                p_false = float(row[2])
                if p_true > threshold or p_false > threshold:
                    label = int(float(row[3]))
                    
                    pred = p_true > p_false
                    correct = (pred == label) #bool

                    num_correct += correct
                    num_total += 1
    if num_total > 0:
        acc = num_correct / num_total
    return acc, num_total

threshs = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9]
honest_accs = []
honest_total = []
liar_accs = []
liar_total = []

for thresh in threshs:
    h_acc, h_tot = get_accuracy('inference_output_honest.csv', threshold=thresh)
    l_acc, l_tot = get_accuracy('inference_output_liar.csv', threshold=thresh)
    honest_accs.append(h_acc)
    honest_total.append(h_tot)
    liar_accs.append(l_acc)
    liar_total.append(l_tot)

fig, axs = plt.subplots(2)

# plot list1 and list2 on the first plot
axs[0].plot(threshs, honest_accs, label='honest')
axs[0].plot(threshs, liar_accs, label='liar')
axs[0].set_xlabel('threshold')
axs[0].set_ylabel('accuracy')
axs[0].legend()

# plot list3 and list4 on the second plot
axs[1].plot(threshs, honest_total, label='honest')
axs[1].plot(threshs, liar_total, label='liar')
axs[1].set_xlabel('threshold')
axs[1].set_ylabel('total data points')
axs[1].legend()

plt.show()
# 
# get_accuracy('inference_output_honest_7b.csv')
# get_accuracy('inference_output_liar_7b.csv')
# # %%
# get_accuracy('inference_output_honest_13b.csv', threshold=.8)
# get_accuracy('inference_output_liar_13b.csv', threshold=0)
# 

# %%
