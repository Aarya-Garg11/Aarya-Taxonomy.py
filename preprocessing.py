
# <a href="https://colab.research.google.com/github/shashankcodes1903/Ontology-Project-HPE-/blob/main/hemanthcs-ontology.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%

import csv
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer


nlp = spacy.load("en_core_web_sm")

_LEADING_STRIP = re.compile(r"^(the|a|an|this|that|these|those|its|their|his|her|our)\s+", re.I)




def preprocess_text(text: str) -> list[str]:
    doc = nlp(text)
    tokens = []

    for token in doc:
        if token.is_stop or token.is_punct or token.like_num or token.is_space:
            continue
        if token.pos_ in ("PRON", "DET", "PART", "INTJ", "SYM", "X"):
            continue

        if token.pos_ == "PROPN":
            tokens.append(token.text.lower())
        elif token.pos_ in ("NOUN", "VERB", "ADJ", "ADV"):
            tokens.append(token.lemma_.lower())
        else:
            tokens.append(token.lower_)

    return tokens




def extract_tfidf_terms(text: str, top_k: int = 20) -> dict[str, float]:
    """TF-IDF over sentence-level corpus for meaningful scores."""
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents if sent.text.strip()]
    if len(sentences) < 2:
        sentences = [text]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    X = vectorizer.fit_transform(sentences)
    feature_names = vectorizer.get_feature_names_out()
    max_scores = X.max(axis=0).toarray()[0]

    sorted_terms = sorted(zip(feature_names, max_scores), key=lambda x: x[1], reverse=True)
    return dict(sorted_terms[:top_k])


def _clean_phrase(phrase: str) -> str:
    phrase = phrase.strip().lower()
    phrase = _LEADING_STRIP.sub("", phrase)
    phrase = re.sub(r"\s+", " ", phrase)
    return phrase


def extract_noun_phrases(text: str) -> list[str]:
    doc = nlp(text)
    seen = set()
    phrases = []
    for chunk in doc.noun_chunks:
        cleaned = _clean_phrase(chunk.text)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            phrases.append(cleaned)
    return phrases


def _remove_substrings(terms: list[str]) -> list[str]:
    """Drop a term if it is a proper substring of another term."""
    terms_sorted = sorted(terms, key=len, reverse=True)
    kept = []
    for candidate in terms_sorted:
        if not any(candidate != longer and candidate in longer for longer in kept):
            kept.append(candidate)
    return kept


def extract_terms(text: str) -> tuple[list[str], dict[str, float]]:
    tfidf_scores = extract_tfidf_terms(text, top_k=20)
    noun_phrases = extract_noun_phrases(text)

    raw_terms = list(tfidf_scores.keys()) + noun_phrases
    seen = set()
    cleaned = []
    for t in raw_terms:
        t = _clean_phrase(t)
        if t and t not in seen and len(t) > 1:
            seen.add(t)
            cleaned.append(t)

    final_terms = _remove_substrings(cleaned)
    return final_terms, tfidf_scores




def extract_concepts(terms: list[str], tfidf_scores: dict[str, float], min_len: int = 2) -> list[dict]:
    concept_data: dict[str, dict] = {}

    for term in terms:
        doc = nlp(term)
        for token in doc:
            if token.pos_ not in ("NOUN", "PROPN"):
                continue
            lemma = token.lemma_.lower()
            if len(lemma) < min_len:
                continue

            source = term if " " in term else concept_data.get(lemma, {}).get("source", term)
            score = tfidf_scores.get(lemma, tfidf_scores.get(term, 0.0))

            if lemma not in concept_data or score > concept_data[lemma]["tfidf_score"]:
                concept_data[lemma] = {
                    "concept": lemma,
                    "pos": token.pos_,
                    "source": source,
                    "tfidf_score": round(score, 4),
                }

    ranked = sorted(concept_data.values(), key=lambda x: (-x["tfidf_score"], x["concept"]))
    for i, entry in enumerate(ranked, start=1):
        entry["rank"] = i

    return ranked


def export_concepts_to_csv(concepts: list[dict], filepath: str = "concepts.csv") -> None:
    fieldnames = ["rank", "concept", "pos", "source", "tfidf_score"]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(concepts)
    print(f"\n\u2705  Concepts exported to '{filepath}'  ({len(concepts)} rows)")
    print(f"Saved to {filepath}")


def ontology_pipeline(text: str, csv_path: str = "concepts.csv") -> dict:
    sep = "=" * 60

    print(f"\n{sep}\nINPUT TEXT\n{sep}")
    print(text.strip())

    # Step 1
    tokens = preprocess_text(text)
    print(f"\n{'─'*60}\nSTEP 1 — CLEAN TOKENS\n{'─'*60}")
    print(tokens)

    # Step 2
    terms, tfidf_scores = extract_terms(text)
    print(f"\n{'-'*60}\nSTEP 2 - EXTRACTED TERMS\n{'-'*60}")
    for t in sorted(terms):
        score = tfidf_scores.get(t, 0.0)
        print(f"  {t:<35}  tfidf={score:.4f}" if score else f"  {t}")

    # Step 3
    concepts = extract_concepts(terms, tfidf_scores)
    print(f"\n{'-'*60}\nSTEP 3 - CONCEPTS\n{'-'*60}")
    header = f"  {'Rank':<6}{'Concept':<25}{'POS':<8}{'TF-IDF':<10}Source"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for c in concepts:
        print(f"  {c['rank']:<6}{c['concept']:<25}{c['pos']:<8}{c['tfidf_score']:<10}{c['source']}")

    export_concepts_to_csv(concepts, filepath=csv_path)

    return {
        "tokens": tokens,
        "terms": terms,
        "tfidf_scores": tfidf_scores,
        "concepts": concepts,
    }



if __name__ == "__main__":
    text = """
   The all-new HPE ProLiant Compute EL2000 chassis is purpose-built for some of the most rugged and  size, weight, and power (SWaP)-constrained environments in national security, manufacturing, retail, and telecommunications. The platform is based on Intel Xeon 6 processors, ideal for demanding edge environments. Supporting up to two HPE ProLiant Compute EL220 Gen12 servers or one EL240 Gen12 server, the chassis helps deliver rugged performance and modular flexibility. The new servers, available only with the HPE ProLiant Compute EL2000, features:

Scalability from 8 up to 144 Intel Xeon 6 cores1
Support for space-saving CPU Thermal Design Power up to 350 watts to achieve higher performance
Reliable operation in extreme temperatures ranging from -40 degrees Celsius to 55 degrees Celsius, as well as up to 95% humidity2, and
Durability in environments with heavy vibration from aircraft or ground vehicles, environmental contaminants, and electromagnetic interference (EMI)
Available with NVIDIA RTX PRO 4500 or NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs (only on EL240 Gen12 server)
Support for NVIDIA AI Enterprise, with government-ready software to meet rigorous security standards and high-assurance environments
    """

    ontology_pipeline(text, csv_path="concepts.csv")

# %%



