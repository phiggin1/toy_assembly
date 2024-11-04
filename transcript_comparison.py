from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os.path
import numpy as np

def gen_plot(data, labels, title, size):
    print(f"{title}, {labels}, {size}")
    plt.figure(figsize=size)
    fig, ax = plt.subplots()
    im = ax.imshow(data, cmap='cool')

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels=labels)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels=labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
 
    # Loop over data dimensions and create text annotations.
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax.text(j, i, f"{data[i, j]:.2}", ha="center", va="center", color="w")

    ax.set_title(title)
    '''
    txt="Cosine Similarity of TF-IDF of language used in each part of both conditions"
    fig.text(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
    '''
    fig.tight_layout()
    plt.show()
    plt.savefig(os.path.join("/home/rivr/", f"ffidf_cosine_sim.png"), bbox_inches='tight')
 

text_files = [
    "/home/rivr/text_image_data/VR_First/vr_first_transcripts.csv",
    "/home/rivr/text_image_data/VR_First/real_second_transcripts.csv",
    "/home/rivr/text_image_data/Real_First/real_first_transcripts.txt",
    "/home/rivr/text_image_data/Real_First/vr_second_transcripts.txt"
]

all_text = []
for fname in text_files:
    print(fname)
    data_frames = pd.read_csv(fname, index_col=False)  
    print(data_frames.shape)
    text = ""
    #print(data_frames["text"])
    for frame in data_frames["text"]:
        text = text + str(frame)
    all_text.append(text)
    
tfidf_vectorizer = TfidfVectorizer(input='content', stop_words='english')
tfidf_vector = tfidf_vectorizer.fit_transform(all_text)

tfidf_df = pd.DataFrame(tfidf_vector.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

print(tfidf_df)

# Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_df, tfidf_df)

print(cosine_sim)

gen_plot(data=cosine_sim, title="Comparison of Speech Across Conditions", size=(5,5), labels=["VR First","Real Seccond","Real First","VR Second"])