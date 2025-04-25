from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from bert_score import score as bert_score
# import moverscore_v2 as moverscore
from nltk.stem.snowball import SnowballStemmer
import numpy as np

# Initialize German stemmer
stemmer = SnowballStemmer("german")

def preprocess_text(text):
    """
    Preprocess text by applying stemming for German.
    :param text: Input text (string).
    :return: Stemmed text (string).
    """
    return " ".join([stemmer.stem(word) for word in text.split()])

def compute_rouge(reference, generated):
    """
    Compute ROUGE scores.
    ROUGE measures n-gram overlap and sequence similarity between the generated and reference summaries.
    Ideal Score: Closer to 1.0 (or 100%) indicates high similarity.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return [scorer.score(preprocess_text(ref), preprocess_text(gen)) for ref, gen in zip(reference, generated)]

def compute_bleu(reference, generated):
    """
    Compute BLEU scores.
    BLEU measures n-gram precision between the generated and reference summaries.
    Ideal Score: Closer to 1.0 (or 100%) indicates high similarity.
    """
    return [sentence_bleu([preprocess_text(ref).split()], preprocess_text(gen).split()) for ref, gen in zip(reference, generated)]

def compute_meteor(reference, generated):
    """
    Compute METEOR scores.
    METEOR measures precision, recall, and semantic similarity, accounting for stemming and synonyms.
    Ideal Score: Closer to 1.0 (or 100%) indicates high similarity.
    """
    # return [meteor_score([ref], gen) for ref, gen in zip(reference, generated)]
    return [
        meteor_score([word_tokenize(ref)], word_tokenize(gen))
        for ref, gen in zip(reference, generated)
    ]

def compute_bertscore(reference, generated):
    """
    Compute BERTScore.
    BERTScore measures semantic similarity using contextual embeddings from BERT.
    Ideal Score: Closer to 1.0 (or 100%) indicates high semantic similarity.
    """
    P, R, F1 = bert_score(generated, reference, lang="de", model_type="bert-base-multilingual-cased")
    return {'Precision': P.mean().item(), 'Recall': R.mean().item(), 'F1': F1.mean().item()}

# def compute_moverscore(reference, generated):
#     """
#     Compute MoverScore.
#     MoverScore measures semantic similarity and word alignment using contextual embeddings and Earth Mover's Distance.
#     Ideal Score: Closer to 1.0 (or 100%) indicates high similarity.
#     """
#     return moverscore.get_scores(generated, reference, model_type="xlm-roberta-base")

def evaluate_summaries(reference, generated):
    """
    Evaluate the quality of generated summaries using various metrics.
    :param reference: List of reference summaries.
    :param generated: List of generated summaries.
    :return: Dictionary of metric scores.
    """
    results = {}

    # Compute each metric
    # Compute each metric
    rouge_scores = compute_rouge(reference, generated)
    results['ROUGE'] = {
        'rouge1': np.mean([score['rouge1'].fmeasure for score in rouge_scores]),
        'rouge2': np.mean([score['rouge2'].fmeasure for score in rouge_scores]),
        'rougeL': np.mean([score['rougeL'].fmeasure for score in rouge_scores]),
    }
    results['BLEU'] = np.mean(compute_bleu(reference, generated))
    results['METEOR'] = np.mean(compute_meteor(reference, generated))
    bert_scores = compute_bertscore(reference, generated)
    results['BERTScore'] = {
        'Precision': bert_scores['Precision'],
        'Recall': bert_scores['Recall'],
        'F1': bert_scores['F1'],
    }
    # results['MoverScore'] = compute_moverscore(reference, generated)

    return results