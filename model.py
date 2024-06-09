import os

import pandas as pd
from bertopic.representation import MaximalMarginalRelevance, KeyBERTInspired
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

from data_utils import extract_text_from_document, get_document_paths

def load_or_instantiate(saved_path: str = None):
    #rep_key = KeyBERTInspired()
    # we add this to remove stopwords
    vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")
    rep_max = MaximalMarginalRelevance(diversity=0.3)
    representation_models = [rep_max]
    if saved_path is None:
        return BERTopic(vectorizer_model=vectorizer_model, representation_model=representation_models)
    return BERTopic.load(saved_path)
    
def save_model(model, path="model"):
    if not os.path.isdir(path):
        os.makedirs(path)
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    model.save(path, serialization="pytorch", save_ctfidf=True, save_embedding_model=embedding_model)

def fit_model(docs, names, model, res):
    #docs = get_document_paths(dir_path)
    docs = extract_text_from_document(docs)
    topics, probs = model.fit_transform(docs)
    df = pd.DataFrame({"Documents": names, "Topic": topics})
    res.append(df)
    res.append(probs)
    res.append(model)
    return res

def fit_and_merge(docs, model_to_fit, new_model):
    df, probs, new_model = fit_model(list(docs.values()), list(docs.keys()), model_to_fit)
    merged_model =  BERTopic.merge_models([new_model, new_model])
    return df, probs, merged_model

def organize_by_term(term, model, doc_score_df, all_docs=False):
    num_topics = len(model.get_topics())
    topics, scores = model.find_topics(search_term=term, top_n=num_topics)
    not_valid = set(doc_score_df["Topic"]).difference(set(topics))
    for not_val in not_valid:
        doc_score_df["Topic"] = doc_score_df["Topic"].replace(not_val, -1)
    doc_score_df["Topic"] = [topics.index(x) for x in doc_score_df["Topic"]]
    doc_score_df = doc_score_df.sort_values(by='Topic')
    doc_score_df["Topic"] = [topics[x] for x in doc_score_df["Topic"]]
    return scores, doc_score_df
