import word
import words_model
import random
import lwa
import numpy as np
import streamlit
import matplotlib.pyplot as plt

model = words_model.words_11
grades = []
with streamlit.container():
    col1, col2 = streamlit.columns(2)
    odd = True
    for w in model["words"].keys():
        occur_amount = 0
        with col1 if odd else col2:
             occur_amount = int(streamlit.text_input(f"Number of occurrences for '{w}' ", "0"))
        grades += [w] * occur_amount
        odd = not odd
if (len(grades) == 0):
    streamlit.warning("At least 1 grade is required!")
    exit()
W = []
for item in model['words']:
    W.append(grades.count(item))

h = min(item['lmf'][-1] for item in model['words'].values())
m = 50
intervals_umf = lwa.alpha_cuts_intervals(m)
intervals_lmf = lwa.alpha_cuts_intervals(m, h)


res_lmf = lwa.y_lmf(intervals_lmf, model, W)
res_umf = lwa.y_umf(intervals_umf, model, W)
res = lwa.construct_dit2fs(np.arange(*model['x']), intervals_lmf, res_lmf, intervals_umf, res_umf)
#result_figure, result_ax = res.plot()

sm = []
model = words_model.words_15
for title, fou in model['words'].items():
    sm.append((title,
               word.Word(title, model['x'], fou['lmf'], fou['umf']),
               res.similarity_measure(word.Word(None, model['x'], fou['lmf'], fou['umf']))))
res_word = max(sm, key=lambda item: item[2])
#estimated_figure, estimated_ax = res_word[1].plot()

fig, (ax1, ax2) = plt.subplots(1, 2)
res.plot_on_ax(ax1)
res_word[1].plot_on_ax(ax2, color="lightpink")
fig.suptitle(res_word[1].title + " " + str(res_word[2]))

fig1, ax = plt.subplots()
res.plot_on_ax(ax)
res_word[1].plot_on_ax(ax, color="lightpink")

plt.grid(True)
streamlit.pyplot(fig)
plt.grid(True)
streamlit.pyplot(fig1)

print(res_word)
