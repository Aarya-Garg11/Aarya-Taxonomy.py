import re
import inspect
import spacy
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import preprocessing

def load_text_from_preprocessing() -> str:
    src = inspect.getsource(preprocessing)
    match = re.search(r'text\s*=\s*"""(.+?)"""', src, re.DOTALL)
    if not match:
        raise ValueError("Could not find `text = \"\"\"...\"\"\"` in preprocessing.py")
    return match.group(1)

nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

BAD_PARENTS = {
    "some", "based", "ideal", "and", "for",
    "way", "type", "kind", "thing",
}

BAD_TERMS = {
    "job", "type", "kind", "way", "thing", "method", "methods",
    "technique", "techniques", "include", "includes",
    "feature", "features", "support", "supports",
}

JUNK_EDGE = {"and", "or", "the", "a", "an", "is", "are", "was", "were", "of", "in"}

INVALID_PHRASES = {
    "based on", "ideal for", "platform is", "for demanding", "and modular",
    "chassis helps", "processors ideal", "on intel", "is based", "is a",
    "up to", "such as", "as well", "only with", "available with",
    "ranging from", "available only", "helps deliver",
}

#PHRASE VALIDATION HELPERS
def _has_noun(term: str) -> bool:
    """Return True if the term contains at least one NOUN or PROPN token."""
    doc = nlp(term)
    return any(t.pos_ in ("NOUN", "PROPN") for t in doc)

def _starts_or_ends_with_stopword(term: str) -> bool:
    """Return True if the first or last word is a spaCy stop word."""
    words = term.strip().split()
    if not words:
        return True
    first = nlp(words[0])[0]
    last  = nlp(words[-1])[0]
    return first.is_stop or last.is_stop


def _is_valid_phrase(term: str) -> bool:
    term = term.strip().lower()
    if not term or len(term) <= 2:
        return False
    if term in INVALID_PHRASES:
        return False
    if term in BAD_TERMS:
        return False
    if re.fullmatch(r"[\d\s\-\.]+", term):  
        return False
    if _starts_or_ends_with_stopword(term):
        return False
    if not _has_noun(term):
        return False
    return True


def valid_parent(term: str) -> bool:
    term = term.strip().lower()
    if not _is_valid_phrase(term):
        return False
    if term in BAD_PARENTS:
        return False

    doc = nlp(term)
    content_tokens = [t for t in doc if not t.is_stop and not t.is_punct]
    if not content_tokens:
        return False
    verb_adj_count = sum(1 for t in content_tokens if t.pos_ in ("VERB", "ADJ", "ADP", "ADV"))
    if verb_adj_count > len(content_tokens) / 2:
        return False

    return True

def is_reasonable_hierarchy(child: str, parent: str) -> bool:
    child  = child.strip().lower()
    parent = parent.strip().lower()

    if child == parent:
        return False
    if not valid_parent(parent):
        return False

    if len(parent.split()) > len(child.split()):
        return False

    return True
# 3. NOUN-HEAD FALLBACK
BAD_HEADS = {
    "thing", "type", "kind", "way", "method", "use", "item",
    "number", "amount", "level", "part", "set", "form",
}


def extract_headword_parent(term: str) -> str | None:
    term = term.strip().lower()
    words = term.split()
    if len(words) < 2:          
        return None

    doc = nlp(term)
    #ROOT first
    head = None
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ in ("NOUN", "PROPN"):
            head = token.lemma_.lower()
            break
    # NOUN/PROPN
    if not head:
        for token in reversed(list(doc)):
            if token.pos_ in ("NOUN", "PROPN"):
                head = token.lemma_.lower()
                break
    if not head:
        return None
    if head == term:
        return None
    if len(head) <= 2:
        return None
    if head in BAD_HEADS or head in BAD_PARENTS:
        return None

    return head

def extract_headword_pairs(terms: list[str]) -> set[tuple[str, str]]:
    """
    For every multi-word term in the list, try to produce a
    (term, headword) is-a pair via extract_headword_parent.
    Returns a set of (child, parent) tuples.
    """
    pairs = set()
    for term in terms:
        head = extract_headword_parent(term)
        if head and _is_valid_phrase(term):
            pairs.add((term, head))
    return pairs

#GENERAL HELPERS
def _clean(phrase: str) -> str:
    """Strip stray connector/aux words from the ends of a phrase."""
    parts = phrase.strip().lower().split()
    while parts and parts[0] in JUNK_EDGE:
        parts.pop(0)
    while parts and parts[-1] in JUNK_EDGE:
        parts.pop()
    return " ".join(parts)


def _is_valid(term: str) -> bool:
    """Quick check: not empty, long enough, not in generic BAD_TERMS."""
    return bool(term) and len(term) > 2 and term not in BAD_TERMS

# HEARST PATTERN EXTRACTION
HEARST_PATTERNS = [
    re.compile(r"(?P<hypo>[\w\-]+(?:\s[\w\-]+){0,2}) (?:is a kind of|is a type of|is an|is a) (?P<hyper>[\w\-]+(?:\s[\w\-]+){0,2})"),
    re.compile(r"(?P<hyper>[\w\-]+(?:\s[\w\-]+){0,2}) (?:such as|including|like) (?P<hypo>[\w\-]+(?:\s[\w\-]+){0,2}(?:\s*,\s*(?:and\s+)?[\w\-]+(?:\s[\w\-]+){0,2})*)"),
    re.compile(r"(?P<hypo>[\w\-]+(?:\s[\w\-]+){0,2}) and other (?P<hyper>[\w\-]+(?:\s[\w\-]+){0,2})"),
    re.compile(r"(?P<hypo>[\w\-]+(?:\s[\w\-]+){0,2}) or other (?P<hyper>[\w\-]+(?:\s[\w\-]+){0,2})"),
]

def extract_hearst_patterns(text: str) -> set:
    relations = set()
    doc = nlp(text)

    for sent in doc.sents:
        sent_lower = sent.text.lower()

        for pattern in HEARST_PATTERNS:
            for match in pattern.finditer(sent_lower):
                hypo  = _clean(match.group("hypo"))
                hyper = _clean(match.group("hyper"))

                if "," in hypo:
                    for part in hypo.split(","):
                        part = _clean(part)
                        if _is_valid_phrase(part) and is_reasonable_hierarchy(part, hyper):
                            relations.add((part, hyper))
                else:
                    if _is_valid_phrase(hypo) and is_reasonable_hierarchy(hypo, hyper):
                        relations.add((hypo, hyper))

    return relations

#NOUN PHRASE EXTRACTION (spaCy)
def get_noun_phrases(text: str) -> list:
    """
    Extracts unique, valid noun phrases from text using spaCy noun chunks.
    Applies _is_valid_phrase to drop fragment chunks.
    """
    doc = nlp(text)
    seen = set()
    phrases = []

    for chunk in doc.noun_chunks:
        clean = chunk.text.strip().lower()
        clean = re.sub(r"^(the|a|an|this|that)\s+", "", clean)
        if _is_valid_phrase(clean) and clean not in seen:
            seen.add(clean)
            phrases.append(clean)

    return phrases

#EMBEDDINGS + CLUSTERING (KMeans)
def cluster_terms(terms: list, n_clusters: int = 5) -> dict:
    """
    Embeds terms with SentenceTransformers and clusters with KMeans.
    Returns {term -> cluster_id}.
    Only valid phrases enter the pool (garbage in → garbage out).
    """
    valid_terms = [t for t in terms if _is_valid_phrase(t)]
    if not valid_terms:
        return {}

    k = min(n_clusters, len(valid_terms))
    embeddings = embedder.encode(valid_terms)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    return {term: int(label) for term, label in zip(valid_terms, labels)}

#TAXONOMY BUILDER
def build_taxonomy(hearst_pairs: set, term_clusters: dict,
                   headword_pairs: set = None,
                   tfidf_scores: dict = None) -> dict:
    taxonomy = {}
    #Hearst patterns
    for child, parent in hearst_pairs:
        if valid_parent(parent) and _is_valid_phrase(child):
            taxonomy[child] = parent
    #Noun-head fallback (fills gaps Hearst misses)
    for child, parent in (headword_pairs or set()):
        if child not in taxonomy and valid_parent(parent) and _is_valid_phrase(child):
            taxonomy[child] = parent
    #Cluster-based fallback
    if term_clusters:
        clusters = defaultdict(list)
        for term, cid in term_clusters.items():
            clusters[cid].append(term)

        for cid, members in clusters.items():
            members = [m for m in members if _is_valid_phrase(m)]
            if len(members) < 2:
                continue

            #single-noun term
            single_nouns = [
                m for m in members
                if len(m.split()) == 1 and _has_noun(m)
            ]
            parent = min(single_nouns, key=len) if single_nouns else min(members, key=len)
            if not valid_parent(parent):
                continue
            for term in members:
                if term == parent or term in taxonomy:
                    continue
                if is_reasonable_hierarchy(term, parent):
                    taxonomy[term] = parent
    return taxonomy

#RELATIONSHIP EXTRACTION (pattern-based)
FOUNDED_BY_PATTERNS = [
    re.compile(r"(?P<org>[\w\-]+(?:\s[\w\-]+){0,3}) (?:was founded by|founded by|was started by|was created by) (?P<person>[\w\-]+(?:\s[\w\-]+){0,3})", re.IGNORECASE),
    re.compile(r"(?P<person>[\w\-]+(?:\s[\w\-]+){0,3}) (?:founded|co-founded|started|created) (?P<org>[\w\-]+(?:\s[\w\-]+){0,3})", re.IGNORECASE),
]

WORKS_AT_PATTERNS = [
    re.compile(r"(?P<person>[\w\-]+(?:\s[\w\-]+){0,3}),?\s+(?:who works at|working at|employed at|an employee of) (?P<org>[\w\-]+(?:\s[\w\-]+){0,3})", re.IGNORECASE),
    re.compile(r"(?P<person>[\w\-]+(?:\s[\w\-]+){0,2}) is (?:a|an)? [\w\-]+ at (?P<org>[\w\-]+(?:\s[\w\-]+){0,3})", re.IGNORECASE),
]

LOCATED_IN_PATTERNS = [
    re.compile(r"(?P<entity>[\w\-]+(?:\s[\w\-]+){0,3}) (?:is located in|is based in|headquartered in|located in|based in) (?P<place>[\w\-]+(?:\s[\w\-]+){0,2})", re.IGNORECASE),
]

def _apply_patterns(text: str, patterns: list, group1: str, group2: str, relation: str) -> set:
    """Run a list of compiled regex patterns on text, return relation triples."""
    results = set()
    for pattern in patterns:
        for match in pattern.finditer(text):
            e1 = match.group(group1).strip().lower()
            e2 = match.group(group2).strip().lower()
            if e1 and e2:
                results.add((e1, relation, e2))
    return results

def extract_relationships(text: str) -> set:
    """
    Extracts (subject, relation, object) triples via keyword pattern matching.
    Runs per sentence and validates with spaCy NER labels.
    """
    doc = nlp(text)
    raw = set()
    for sent in doc.sents:
        s = sent.text
        raw |= _apply_patterns(s, FOUNDED_BY_PATTERNS, "org",    "person", "founded_by")
        raw |= _apply_patterns(s, WORKS_AT_PATTERNS,   "person", "org",    "works_at")
        raw |= _apply_patterns(s, LOCATED_IN_PATTERNS, "entity", "place",  "located_in")

    persons = {e.text.lower() for e in doc.ents if e.label_ == "PERSON"}
    orgs    = {e.text.lower() for e in doc.ents if e.label_ in ("ORG", "PRODUCT")}
    places  = {e.text.lower() for e in doc.ents if e.label_ in ("GPE", "LOC", "FAC")}

    def hits(text_, ents):
        t = text_.lower()
        return any(t in n or n in t for n in ents)

    valid = set()
    for s, r, o in raw:
        if r == "founded_by" and hits(s, orgs) and hits(o, persons):
            valid.add((s, r, o))
        elif r == "works_at" and hits(s, persons) and hits(o, orgs):
            valid.add((s, r, o))
        elif r == "located_in" and hits(o, places):
            valid.add((s, r, o))
    return valid

#GRAPH BUILDER (networkx)
def build_graph(taxonomy: dict, relationships) -> nx.DiGraph:
    """
    Builds a directed graph:
      - taxonomy pairs  → "is_a" edges
      - relation triples → typed edges
    """
    G = nx.DiGraph()
    for child, parent in taxonomy.items():
        G.add_edge(child, parent, relation="is_a")
    for subj, rel, obj in relationships:
        G.add_edge(subj, obj, relation=rel)
    return G

def summarize_graph(G: nx.DiGraph) -> dict:
    """Returns a simple summary dict of the graph."""
    return {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "nodes": list(G.nodes()),
        "edges": [(u, v, G[u][v]["relation"]) for u, v in G.edges()],
    }

#VISUALIZATION
REL_COLORS = {
    "is_a":       "#7f8c8d",   # grey  (taxonomy backbone)
    "founded_by": "#e74c3c",   # red
    "works_at":   "#27ae60",   # green
    "located_in": "#f39c12",   # orange
}

def visualize_graph(G: nx.DiGraph, save_path: str = "taxonomy_graph.png") -> None:
    """Draws the taxonomy/relation graph and saves it as a PNG."""
    if G.number_of_nodes() == 0:
        print("Graph is empty, nothing to draw.")
        return
    
    plt.figure(figsize=(13, 9))
    pos = nx.spring_layout(G, k=1.6, seed=42)

    nx.draw_networkx_nodes(G, pos, node_color="#cce5ff",
                           node_size=1600, edgecolors="#1f4e79", linewidths=1.2)
    nx.draw_networkx_labels(G, pos, font_size=8)

    for rel, color in REL_COLORS.items():
        edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation") == rel]
        if edges:
            nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=color,
                                   width=1.6, arrows=True, arrowsize=14,
                                   connectionstyle="arc3,rad=0.05")

    handles = [plt.Line2D([0], [0], color=c, lw=2.5, label=r)
               for r, c in REL_COLORS.items()]
    plt.legend(handles=handles, loc="upper left", fontsize=9, frameon=True)

    plt.title("Taxonomy + Relationship Graph", fontsize=13)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Graph saved to {save_path}")

#PIPELINE
def run_pipeline(text: str, pre_output, n_clusters: int = 5):
    """
    Builds taxonomy + relations + graph using the output of
    preprocessing.ontology_pipeline().

    pre_output can be:
      - dict: {"tokens", "terms", "tfidf_scores", "concepts"}
      - list: just the concepts list
    """
    if isinstance(pre_output, dict):
        concepts     = pre_output.get("concepts", [])
        terms        = pre_output.get("terms", [])
        tfidf_scores = pre_output.get("tfidf_scores", {})
        tokens       = pre_output.get("tokens", [])
    else:
        concepts     = pre_output
        terms        = []
        tfidf_scores = {}
        tokens       = []

    concept_names = [c["concept"] for c in concepts]

    # ── Hearst patterns (primary taxonomy evidence) ───────────────
    hearst_pairs = extract_hearst_patterns(text)
    print(f"Hearst pairs: {len(hearst_pairs)}")

    doc = nlp(text)
    person_tokens = set()
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            for w in ent.text.lower().split():
                person_tokens.add(w)

    vocab = list(dict.fromkeys(concept_names + terms))
    clusterable = [
        t for t in vocab
        if t not in person_tokens and _is_valid_phrase(t)
    ]
    print(f"Cluster terms: {len(clusterable)}")
    term_clusters = cluster_terms(clusterable, n_clusters=n_clusters)
    # Run on all valid multi-word phrases from vocab
    all_phrases = [t for t in vocab if _is_valid_phrase(t) and len(t.split()) > 1]
    headword_pairs = extract_headword_pairs(all_phrases)
    #Taxonomy: Hearst → headword → clusters ────────────────────
    taxonomy = build_taxonomy(hearst_pairs, term_clusters, headword_pairs, tfidf_scores)
    print(f"Taxonomy edges: {len(taxonomy)}")
    print("Sample taxonomy:", list(taxonomy.items())[:10])
    #Relationships (NER-validated)
    relations = extract_relationships(text)
    #Graph
    G = build_graph(taxonomy, relations)
    graph_summary = summarize_graph(G)
    graph_summary["doc_token_count"]   = len(tokens)
    graph_summary["doc_unique_tokens"] = len(set(tokens))

    return taxonomy, relations, graph_summary, G

# EXAMPLE USAGE
if __name__ == "__main__":
    sample_text = load_text_from_preprocessing()

    # let preprocessing do its full job (tokens, terms, tfidf, concepts, csv export)
    pre_output = preprocessing.ontology_pipeline(sample_text, csv_path="concepts.csv")

    # taxonomy reuses preprocessing's full output — no second preprocessing pass
    taxonomy, relations, graph_summary, G = run_pipeline(
        sample_text, pre_output, n_clusters=4
    )

    print("=== TAXONOMY (child -> parent) ===")
    for child, parent in taxonomy.items():
        print(f"  {child} --> {parent}")

    print("\n=== RELATIONS ===")
    for subj, rel, obj in relations:
        print(f"  ({subj})  --[{rel}]-->  ({obj})")

    print("\n=== GRAPH SUMMARY ===")
    print(f"  Nodes : {graph_summary['num_nodes']}")
    print(f"  Edges : {graph_summary['num_edges']}")
    print(f"  Doc tokens : {graph_summary['doc_token_count']} ({graph_summary['doc_unique_tokens']} unique)")
    for u, v, rel in graph_summary["edges"]:
        print(f"  {u} --[{rel}]--> {v}")
    # draw it
    visualize_graph(G, save_path="taxonomy_graph.png")
