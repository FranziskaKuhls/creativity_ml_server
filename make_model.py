import pandas as pd 
df = pd.read_csv("~/Desktop/Masterarbeit/data/argumentative-creative-essays/Corpus-creative-essays-climate-change.csv")
df.columns =['id', 'text']
print(len(df))
df.head(1)

"""## Preprocess
We split the esssays into sentences.
"""

import itertools
print(len(df['text'].values))
docs_list = [df['text'][i].split(".") for i in range(len(df))]
our_docs = list(itertools.chain.from_iterable(docs_list))
len(our_docs)

"""We keep the sentences with more than one word. We are removing empty enters. """

docs_filtered = [x for x in our_docs if len(x.split())>1]
len(docs_filtered)

"""## Topic Modeling"""

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

# Remove stop words from definition not from model
vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")

# Seed the model
umap_model = UMAP(n_neighbors=10, n_components=5, 
                  min_dist=0.0, metric='cosine', random_state=42)

topic_model = BERTopic(language="english", 
                       min_topic_size = 10,
                       umap_model=umap_model,
                       vectorizer_model=vectorizer_model, 
                       calculate_probabilities=True, 
                       verbose=True)

# Fit the model with our data
topics, probs = topic_model.fit_transform(docs_filtered)

"""### View results"""

topic_model.visualize_barchart(top_n_topics=20)

topic_model.visualize_distribution(probs[200], min_probability=0.015)

topic_model.visualize_heatmap(n_clusters=None, width=1000, height=1000)

# Save the model
topic_model.save("climate_change_model")

"""## Predictions

Load the model
"""

from bertopic import BERTopic
preds_model = BERTopic.load("climate_change_model")

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def preprocess(text):
  sentences = text.split('.') 
  sentences = [x for x in sentences if len(x.split())>1]
  return sentences


def get_fluency(preds_model, pred_topics):
  """
  The general idea is to sum up the ‘effort’ that each topic represents, which is expressed as the distance of each topic to zero. If you just count, the assumption is that the effort of each topic is 1.

  Returns: Value between 0 and 1 
  1 means that the user wrote about ALL the topics
  0 the user did not write about any of the topics
  """
  list_topics = list(np.unique(pred_topics))
  # -1 are outliers
  found_topics = [x for x in list_topics if x>-1]
  all_topics = preds_model.get_topics()
  fluency = len(found_topics)/len(all_topics) 
  return found_topics, fluency


def compute_flexibility(found_topics, similarity_matrix):
  """
  Range 0-1. Where 1 is that they are very distant and 0 that they are very close
  Greater is best

  If there is only one element it will return zero
  """
  global_similarity = 0


  if len(found_topics)>1:
    for i in found_topics:
      other_topics = [x for x in found_topics if x != i]
      topic_similarity = 0
      for j in other_topics:
        topic_similarity +=  similarity_matrix[i,j]

      topic_similarity = topic_similarity/len(other_topics)
      global_similarity += topic_similarity/len(found_topics)
  else:
    global_similarity = 1


  return 1 - global_similarity


def get_flexibility(preds_model, found_topics):
  """
  Average pairwise distance between all user topics
  """
  embeddings = np.array(preds_model.topic_embeddings)

  # remove -1 (outliers)
  embeddings = embeddings[1:]

  similarity_matrix = cosine_similarity(embeddings)
  flexibility = compute_flexibility(found_topics, similarity_matrix)
  return flexibility

def get_originality(pred_topics):
  """
  How many were outliers TBD
  """
  originality = pred_topics.count(-1)/len(pred_topics)
  return originality


climate_model = BERTopic.load("climate_change_model")

## MAIN 
def get_metrics(text, dataset = "climate_change"):
  if dataset == "climate_change":
    preds_model = climate_model
  sentences = preprocess(text)
  pred_topics, pred_prob = preds_model.transform(sentences)
  found_topics, fluency = get_fluency(preds_model, pred_topics)
  flexibility = get_flexibility(preds_model, found_topics)
  originality = get_originality(pred_topics)
  return fluency, flexibility, originality

### Tests
text = df['text'].iloc[0]
get_metrics(text)

text  = """
The shoelace formula, shoelace algorithm, or shoelace method (also known as Gauss's area formula and the surveyor's formula)[1] is a mathematical algorithm to determine the area of a simple polygon whose vertices are described by their Cartesian coordinates in the plane.[2] It is called the shoelace formula because of the constant cross-multiplying for the coordinates making up the polygon, like threading shoelaces.[2] It has applications in surveying and forestry,[3] among other areas.
The formula was described by Albrecht Ludwig Friedrich Meister (1724–1788) in 1769[4] and is based on the trapezoid formula which was described by Carl Friedrich Gauss and C.G.J. Jacobi.[5] The triangle form of the area formula can be considered to be a special case of Green's theorem.
The area formula can also be applied to self-overlapping polygons since the meaning of area is still clear even though self-overlapping polygons are not generally simple.[6] Furthermore, a self-overlapping polygon can have multiple "interpretations" but the Shoelace formula can be used to show that the polygon's area is the same regardless of the interpretation.[7]
"""
get_metrics(text)
