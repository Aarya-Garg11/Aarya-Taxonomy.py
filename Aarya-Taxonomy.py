"""
BOOFS: Bootstrapped Ontology and Object Frame Semantics
========================================================

Universal Ontology Learning System - Complete Implementation

Novel algorithms:
1. Bootstrapped Distant Supervision (auto-seed generation)
2. Frame-Based Semantic Slot Filling (universal frames)
3. Unsupervised Relation Discovery (distributional clustering)
4. Active Learning (smart example selection)
5. Distributional Fact Completion (entity similarity transfer)

Research paper: "Universal Ontology Learning from Unstructured Text"

------------------------------------------------------------------------------
NOTE ON THIS REVISION
------------------------------------------------------------------------------
This file keeps the original BOOFS architecture and 7-stage pipeline fully
intact (coreference -> concept extraction -> distant supervision -> frame slot
filling -> unsupervised relation discovery -> consolidation -> KG embeddings).
The changes applied here are *incremental hardening only* and are tagged inline
as [IMPROVEMENT N]. No stage was removed or replaced. A full change log appears
at the bottom of the file.
"""

import csv
import re
import logging
from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Set, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════════
# [IMPROVEMENT 11 + 4] CENTRALIZED CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
#
# Previously, every tunable number (confidence values, DBSCAN eps, entity
# distance, spaCy model name) was a magic literal scattered across the file.
# That made calibration impossible and the spaCy model impossible to change
# without editing module-load code. All such knobs now live in one dataclass.
# Defaults preserve the original numeric behavior except where an improvement
# deliberately changes it (documented per field).

@dataclass
class BOOFSConfig:
    # --- spaCy model selection [IMPROVEMENT 4] ---
    spacy_model_preferred: str = "en_core_web_lg"   # prefer the larger, more accurate model
    spacy_model_fallback: str = "en_core_web_sm"    # graceful fallback if lg isn't installed

    # --- entity pairing [IMPROVEMENT 13] ---
    # Originally a character heuristic (max_entity_distance * 10 chars). Now an
    # explicit *token* distance, which is far more stable across entity lengths.
    max_entity_token_distance: int = 10

    # --- clustering ---
    dbscan_eps_distant: float = 0.35
    dbscan_eps_unsup: float = 0.4

    # --- negation handling [IMPROVEMENT 7] ---
    # If True, a negated frame ("did not marry") does NOT assert its positive
    # canonical relation. If False, the relation is still emitted but its
    # confidence is multiplied by negation_confidence_penalty.
    skip_negated: bool = True
    negation_confidence_penalty: float = 0.4

    # --- KG embedding evaluation [IMPROVEMENT 9] ---
    # Below this many triples, an honest held-out split is not meaningful, so
    # Hits@K reporting is disabled instead of evaluating on the training set.
    min_triples_for_eval: int = 20
    kg_test_ratio: float = 0.2

    # --- confidence constants [IMPROVEMENT 11/12] ---
    conf_ner: float = 0.9
    conf_noun_chunk: float = 0.6
    conf_distant_base: float = 0.7
    conf_distributional: float = 0.6
    # [IMPROVEMENT 3] entity-similarity edges are hypotheses, not relations:
    conf_similarity: float = 0.25
    # [IMPROVEMENT 12] canonical frame relations are scaled from their slots'
    # own confidences rather than a single hardcoded literal:
    conf_frame_scale: float = 0.95
    conf_partial_scale: float = 0.9

    # [IMPROVEMENT 12] role-assignment confidence gradient. The strength of a
    # slot fill now depends on *how* its role was inferred, instead of always
    # collapsing to a single value (the old code had a dead 0.9/0.6 ternary).
    role_conf_direct: float = 1.0     # entity attached directly to the trigger
    role_conf_prep: float = 0.85      # role inferred through a preposition chain
    role_conf_fallback: float = 0.7   # role from generic dependency mapping
    role_conf_context: float = 0.5    # no specific role found

    # base confidence a slot fill is scaled from
    slot_base_entity: float = 0.9
    slot_base_noun: float = 0.72


CONFIG = BOOFSConfig()


# ════════════════════════════════════════════════════════════════════════════
# [IMPROVEMENT 4] CONFIGURABLE spaCy MODEL LOADER
# ════════════════════════════════════════════════════════════════════════════
#
# Original code hardcoded `spacy.load("en_core_web_sm")`. Every downstream
# heuristic (NER, dependency-walking role assignment, noun chunks) is only as
# good as this parse, and `sm` is weakest on the long narrative sentences BOOFS
# targets. We now try the larger model first and fall back gracefully, with no
# changes to any other stage.

def load_spacy_model(preferred: str = None, fallback: str = None):
    """Load the preferred spaCy model, falling back to a smaller one if needed."""
    preferred = preferred or CONFIG.spacy_model_preferred
    fallback = fallback or CONFIG.spacy_model_fallback
    for name in (preferred, fallback):
        try:
            model = spacy.load(name)
            if name != preferred:
                logger.warning(f"Preferred model '{preferred}' unavailable; using '{name}'.")
            else:
                logger.info(f"Loaded spaCy model '{name}'.")
            return model
        except OSError:
            continue
    logger.error(
        f"No spaCy model found. Install one with: "
        f"python -m spacy download {preferred}  (or {fallback})"
    )
    raise OSError(f"Neither '{preferred}' nor '{fallback}' is installed.")


nlp = load_spacy_model()


# ════════════════════════════════════════════════════════════════════════════
# COREFERENCE RESOLUTION (PREPROCESSING STAGE) — unchanged in behavior
# ════════════════════════════════════════════════════════════════════════════
#
# Resolves pronouns to their canonical mentions so downstream modules receive
# cleaner text. Purely additive: with no backend available, resolve() returns
# the original text unchanged. (Only the bare-except hardening from
# [IMPROVEMENT 8] is applied inside this class; logic is otherwise untouched.)

try:
    from fastcoref import FCoref
    _COREF_BACKEND = "fastcoref"
except ImportError:
    try:
        import spacy_experimental  # noqa: F401  # type: ignore
        _COREF_BACKEND = "spacy_experimental"
    except ImportError:
        try:
            import neuralcoref  # noqa: F401  # type: ignore
            _COREF_BACKEND = "neuralcoref"
        except ImportError:
            _COREF_BACKEND = None


class CoreferenceResolver:
    """
    Resolves pronouns to canonical entity mentions before BOOFS extraction.

    Tries fastcoref first, then spacy-experimental, then neuralcoref, then a
    lightweight rule-based resolver. This guarantees the module always works
    while preferring more accurate neural backends when present.
    """

    def __init__(self):
        self.backend = _COREF_BACKEND
        self._coref_nlp = None
        self._fcoref_model = None

        if self.backend == "fastcoref":
            try:
                self._fcoref_model = FCoref()
            except Exception:
                logger.warning("fastcoref failed to initialize; falling back to next backend.")
                self.backend = None

        if self.backend is None:
            try:
                import spacy_experimental  # noqa: F401  # type: ignore
                self.backend = "spacy_experimental"
            except ImportError:
                try:
                    import neuralcoref  # noqa: F401  # type: ignore
                    self.backend = "neuralcoref"
                except ImportError:
                    self.backend = None

        if self.backend == "spacy_experimental":
            try:
                self._coref_nlp = spacy.load("en_coreference_web_trf")
            except Exception:
                logger.warning("spacy-experimental coref model not found; falling back to rule-based resolver.")
                self.backend = None

        elif self.backend == "neuralcoref":
            try:
                neuralcoref.add_to_pipe(nlp)
                self._coref_nlp = nlp
            except Exception:
                logger.warning("neuralcoref failed to attach; falling back to rule-based resolver.")
                self.backend = None

    def resolve(self, text: str) -> str:
        """Replace pronouns in `text` with their resolved canonical mentions."""
        if self.backend == "fastcoref" and self._fcoref_model is not None:
            return self._resolve_fastcoref(text)
        if self.backend == "spacy_experimental" and self._coref_nlp is not None:
            return self._resolve_spacy_experimental(text)
        if self.backend == "neuralcoref" and self._coref_nlp is not None:
            return self._resolve_neuralcoref(text)
        return self._resolve_rule_based(text)

    def _resolve_fastcoref(self, text: str) -> str:
        try:
            preds = self._fcoref_model.predict(texts=[text])
            clusters = preds[0].get_clusters(as_strings=False)
        except Exception as e:
            logger.warning(f"fastcoref prediction failed ({e}); returning original text.")
            return text

        span_clusters = []
        for cluster in clusters:
            spans = [type("Span", (), {"start_char": s, "end_char": e, "text": text[s:e]})() for s, e in cluster]
            span_clusters.append(spans)
        return self._apply_clusters(text, span_clusters)

    def _resolve_spacy_experimental(self, text: str) -> str:
        doc = self._coref_nlp(text)
        clusters = [v for k, v in doc.spans.items() if k.startswith("coref_clusters")]
        return self._apply_clusters(text, clusters)

    def _resolve_neuralcoref(self, text: str) -> str:
        doc = self._coref_nlp(text)
        if doc._.has_coref:
            return doc._.coref_resolved
        return text

    def _apply_clusters(self, text: str, clusters) -> str:
        """
        Canonical-mention selection prefers the shortest proper-name-looking
        mention, falling back to longest string. Overlapping replacement spans
        are dropped (keep-first) to avoid corrupting the output.
        """
        replacements = []
        for cluster in clusters:
            if not cluster:
                continue

            def _looks_like_proper_name(span_text: str) -> bool:
                words = span_text.split()
                return bool(words) and words[0][0:1].isupper() and len(words) <= 4

            proper_candidates = [s for s in cluster if _looks_like_proper_name(s.text)]
            if proper_candidates:
                main = min(proper_candidates, key=lambda s: len(s.text))
            else:
                main = max(cluster, key=lambda s: len(s.text))

            for mention in cluster:
                if mention.text.lower() != main.text.lower():
                    replacements.append((mention.start_char, mention.end_char, main.text))

        replacements.sort(key=lambda r: r[0], reverse=True)

        accepted = []
        for start, end, repl in replacements:
            if any(not (end <= a_start or start >= a_end) for a_start, a_end, _ in accepted):
                continue
            accepted.append((start, end, repl))

        resolved = text
        for start, end, repl in accepted:
            resolved = resolved[:start] + repl + resolved[end:]
        return resolved

    PRONOUNS = {'he', 'him', 'his', 'she', 'her', 'hers', 'they', 'them', 'their', 'it', 'its'}
    PERSONAL_PRONOUNS = {'he', 'him', 'his', 'she', 'her', 'hers'}
    IMPERSONAL_PRONOUNS = {'it', 'its'}

    def _resolve_rule_based(self, text: str) -> str:
        """
        Minimal fallback: replace each pronoun with the nearest preceding entity
        of the grammatically-required type (PERSON vs non-PERSON).
        """
        doc = nlp(text)
        last_person = None
        last_nonperson = None
        replacements = []

        token_to_entity = {}
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "ORG", "GPE", "PRODUCT"):
                for tok in ent:
                    token_to_entity[tok.i] = ent

        for token in doc:
            word = token.text.lower()
            if word in self.PRONOUNS:
                if word in self.PERSONAL_PRONOUNS and last_person:
                    replacements.append((token.idx, token.idx + len(token.text), last_person))
                elif word in self.IMPERSONAL_PRONOUNS and last_nonperson:
                    replacements.append((token.idx, token.idx + len(token.text), last_nonperson))
                elif word not in self.PERSONAL_PRONOUNS and word not in self.IMPERSONAL_PRONOUNS:
                    fallback = last_person or last_nonperson
                    if fallback:
                        replacements.append((token.idx, token.idx + len(token.text), fallback))

            ent = token_to_entity.get(token.i)
            if ent is not None:
                if ent.label_ == "PERSON":
                    last_person = ent.text
                else:
                    last_nonperson = ent.text

        replacements.sort(key=lambda r: r[0], reverse=True)
        resolved = text
        for start, end, repl in replacements:
            resolved = resolved[:start] + repl + resolved[end:]
        return resolved


# ════════════════════════════════════════════════════════════════════════════
# UNIVERSAL FRAMES DEFINITION
# ════════════════════════════════════════════════════════════════════════════
#
# [IMPROVEMENT 6] Two changes to every frame, no slots touched:
#   (a) `triggers` are now LEMMAS ONLY. The original lists mixed lemmas with
#       inflected forms ('founded', 'studied', 'employed', 'located', 'based')
#       and even duplicates ('founded' twice, 'establish' twice in FOUNDING).
#       Since matching is done against `token.lemma_`, those inflected entries
#       could NEVER fire — they were dead weight that overstated coverage.
#   (b) A new `trigger_pos` set gates which part-of-speech may activate the
#       frame. Verb-driven frames fire only on VERBs (kills false positives like
#       "serve dinner" / a noun "position"); FAMILY also accepts NOUNs so that
#       relational nouns ('son', 'sister', 'spouse') still trigger it.

UNIVERSAL_FRAMES = {
    'EMPLOYMENT': {
        # lemmas only; verb-driven
        'triggers': ['work', 'employ', 'hire', 'join', 'serve'],
        'trigger_pos': {'VERB'},
        'description': 'Someone works at organization in a role',
        'slots': {
            'EMPLOYEE': {'role': 'AGENT', 'ner_types': ['PERSON']},
            # GPE included so a country/government can be an "employer" for
            # civic roles (e.g. "served as PM of India").
            'EMPLOYER': {'role': 'PATIENT', 'ner_types': ['ORG', 'PRODUCT', 'GPE']},
            'POSITION': {'role': 'ATTRIBUTE', 'ner_types': ['NOUN']},
            'START_TIME': {'role': 'TEMPORAL', 'ner_types': ['DATE']},
            'END_TIME': {'role': 'TEMPORAL', 'ner_types': ['DATE']},
            'LOCATION': {'role': 'LOCATION', 'ner_types': ['GPE', 'LOC']},
        }
    },
    'FOUNDING': {
        'triggers': ['found', 'establish', 'create', 'start', 'launch', 'form'],
        'trigger_pos': {'VERB'},
        'description': 'Someone/organization founds/establishes an entity',
        'slots': {
            'FOUNDER': {'role': 'AGENT', 'ner_types': ['PERSON', 'ORG']},
            'FOUNDED_ENTITY': {'role': 'PATIENT', 'ner_types': ['ORG', 'PRODUCT']},
            'TIME': {'role': 'TEMPORAL', 'ner_types': ['DATE']},
            'LOCATION': {'role': 'LOCATION', 'ner_types': ['GPE', 'LOC']},
        }
    },
    'EDUCATION': {
        'triggers': ['study', 'graduate', 'attend', 'major', 'enroll', 'educate'],
        'trigger_pos': {'VERB'},
        'description': 'Someone studies at institution',
        'slots': {
            'STUDENT': {'role': 'AGENT', 'ner_types': ['PERSON']},
            'INSTITUTION': {'role': 'LOCATION', 'ner_types': ['ORG']},
            'FIELD': {'role': 'ATTRIBUTE', 'ner_types': ['NOUN']},
            'DEGREE': {'role': 'ATTRIBUTE', 'ner_types': ['NOUN']},
            'TIME': {'role': 'TEMPORAL', 'ner_types': ['DATE']},
        }
    },
    'FAMILY': {
        # marry/divorce are verbs; the rest are relational nouns -> allow both POS
        'triggers': ['marry', 'divorce', 'parent', 'child', 'spouse',
                     'sibling', 'brother', 'sister', 'son', 'daughter'],
        'trigger_pos': {'VERB', 'NOUN'},
        'description': 'Family relationships',
        'slots': {
            'PERSON1': {'role': 'AGENT', 'ner_types': ['PERSON']},
            'PERSON2': {'role': 'PATIENT', 'ner_types': ['PERSON']},
            'RELATION_TYPE': {'role': 'ATTRIBUTE', 'ner_types': ['NOUN']},
            'TIME': {'role': 'TEMPORAL', 'ner_types': ['DATE']},
        }
    },
    'LOCATION': {
        'triggers': ['locate', 'base', 'situate', 'headquarter', 'reside', 'live'],
        'trigger_pos': {'VERB'},
        'description': 'Entity is located at place',
        'slots': {
            'ENTITY': {'role': 'AGENT', 'ner_types': ['ORG', 'PERSON']},
            'PLACE': {'role': 'LOCATION', 'ner_types': ['GPE', 'LOC']},
            'TIME': {'role': 'TEMPORAL', 'ner_types': ['DATE']},
        }
    },
}

# Universal semantic role mapping (language-independent)
SEMANTIC_ROLE_MAPPING = {
    'nsubj': 'AGENT',
    'nsubjpass': 'PATIENT',
    'dobj': 'PATIENT',
    'iobj': 'BENEFICIARY',
    'pobj': 'CONTEXT',
    'attr': 'ATTRIBUTE',
    'prep': 'CONTEXT',
    'npadvmod': 'CONTEXT',
}


# ════════════════════════════════════════════════════════════════════════════
# CORE DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════

class ConceptExtract:
    """Extracted concept with metadata."""
    def __init__(self, text: str, entity_type: str, surface: str, confidence: float = 0.5):
        self.text = text.lower()
        self.type = entity_type
        self.surface = surface
        self.confidence = confidence
        self.sources = []

    def to_dict(self):
        return {
            'concept': self.text,
            'type': self.type,
            'surface': self.surface,
            'confidence': round(self.confidence, 3),
            'sources': ','.join(set(self.sources))
        }


class RelationExtract:
    """Extracted relation with metadata."""
    def __init__(self, subject: str, relation: str, object_: str, confidence: float = 0.5):
        self.subject = subject.lower()
        self.relation = relation
        self.object = object_.lower()
        self.confidence = confidence
        self.source = None
        self.evidence = None

    def to_dict(self):
        return {
            'subject': self.subject,
            'relation': self.relation,
            'object': self.object,
            'confidence': round(self.confidence, 3),
            'source': self.source,
            'evidence': self.evidence
        }

    def __hash__(self):
        return hash((self.subject, self.relation, self.object))

    def __eq__(self, other):
        return (self.subject == other.subject and
                self.relation == other.relation and
                self.object == other.object)


class FrameInstance:
    """Detected frame with filled slots."""
    def __init__(self, frame_type: str, trigger: str):
        self.frame_type = frame_type
        self.trigger = trigger
        self.slots = {}  # slot_name -> (value, confidence)
        self.sentence = None
        # [IMPROVEMENT 7] polarity flag set during slot filling when the trigger
        # carries a `neg` dependency. Consumed in consolidation.
        self.negated = False

    def add_slot(self, slot_name: str, value: str, confidence: float = 0.5):
        if slot_name not in self.slots:
            self.slots[slot_name] = (value, confidence)
        elif confidence > self.slots[slot_name][1]:
            self.slots[slot_name] = (value, confidence)

    def to_dict(self):
        return {
            'frame_type': self.frame_type,
            'trigger': self.trigger,
            'slots': self.slots,
            'sentence': self.sentence,
            'negated': self.negated,
        }


# ════════════════════════════════════════════════════════════════════════════
# ALGORITHM 1: BOOTSTRAPPED DISTANT SUPERVISION
# ════════════════════════════════════════════════════════════════════════════

class DistantSupervisionModule:
    """
    Automatically generates training pairs from text. No manual seed annotation.
    """

    def __init__(self, max_entity_distance: int = None):
        # [IMPROVEMENT 13] distance is now a TOKEN count from config (default 10),
        # not a character heuristic.
        self.max_entity_distance = (max_entity_distance
                                     if max_entity_distance is not None
                                     else CONFIG.max_entity_token_distance)
        self.entity_pairs = []
        self.discovered_relations = defaultdict(list)

    def extract_entity_pairs(self, doc: spacy.tokens.Doc) -> List[Dict]:
        """Find all entity pairs within the token-distance threshold."""
        pairs = []

        for sent in doc.sents:
            entities = list(sent.ents)

            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    ent1, ent2 = entities[i], entities[j]

                    # [IMPROVEMENT 13] token distance between the end of ent1 and
                    # the start of ent2. `ent.start`/`ent.end` are token offsets,
                    # so this is invariant to how many characters each entity
                    # spans (the old `start_char`/`end_char * 10` heuristic let a
                    # pair of long entities slip the threshold and short ones fail
                    # it, inconsistently).
                    token_distance = ent2.start - ent1.end
                    if 0 <= token_distance <= self.max_entity_distance:
                        context = self._extract_context(sent, ent1, ent2)

                        pairs.append({
                            'entity1': ent1.text.lower(),
                            'type1': ent1.label_,
                            'entity2': ent2.text.lower(),
                            'type2': ent2.label_,
                            'context': context,
                            'sentence': sent.text,
                            'confidence': CONFIG.conf_distant_base
                        })

        self.entity_pairs = pairs
        logger.info(f"[Distant Supervision] Found {len(pairs)} entity pairs")
        return pairs

    def _extract_context(self, sent, ent1, ent2):
        """Extract lemmatized context between two entities."""
        try:
            if ent1.start_char < ent2.start_char:
                between_text = sent.text[ent1.end_char:ent2.start_char].strip()
            else:
                between_text = sent.text[ent2.end_char:ent1.start_char].strip()

            if between_text:
                context_doc = nlp(between_text)
                lemmas = [t.lemma_ for t in context_doc if not t.is_stop and not t.is_punct]
                return " ".join(lemmas)
            return ""
        # [IMPROVEMENT 8] Catch only the exceptions that can realistically arise
        # from span/text handling, and log them, instead of a bare `except:` that
        # silently swallowed *every* error (including bugs) and returned "".
        except (AttributeError, ValueError, IndexError) as e:
            logger.debug(f"[Distant Supervision] context extraction failed: {e}")
            return ""

    def discover_relations_via_clustering(self) -> Dict[str, List]:
        """
        Cluster entity pairs by context similarity to discover relation types.
        """
        if len(self.entity_pairs) < 3:
            logger.warning("Need at least 3 entity pairs for clustering")
            return {}

        by_signature = defaultdict(list)
        for pair in self.entity_pairs:
            sig = f"{pair['type1']}-{pair['type2']}"
            by_signature[sig].append(pair)

        # [IMPROVEMENT 2] Accumulate into a defaultdict(list) instead of a plain
        # dict keyed by relation name. The original `discovered[rel_name] = ...`
        # OVERWROTE any earlier cluster that produced the same auto-generated
        # name (e.g. two different type-signatures both naming themselves
        # "REL_FOUND"), silently dropping relations. We also namespace the name
        # by the type signature so distinct signatures never collide, and use
        # .extend() so even same-signature same-name clusters merge rather than
        # clobber.
        discovered = defaultdict(list)

        for signature, sig_pairs in by_signature.items():
            if len(sig_pairs) < 2:
                continue

            contexts = [p['context'] for p in sig_pairs if p['context']]
            if not contexts or len(set(contexts)) < 2:
                continue

            try:
                vectorizer = TfidfVectorizer(max_features=50, min_df=1, max_df=10)
                X = vectorizer.fit_transform(contexts)

                clustering = DBSCAN(eps=CONFIG.dbscan_eps_distant, min_samples=1, metric='cosine')
                labels = clustering.fit_predict(X.toarray())

                for cluster_id in set(labels):
                    if cluster_id == -1:
                        continue

                    cluster_pairs = [sig_pairs[i] for i, l in enumerate(labels) if l == cluster_id]

                    cluster_contexts = [p['context'] for p in cluster_pairs]
                    all_words = " ".join(cluster_contexts).split()
                    most_common = Counter(all_words).most_common(3)

                    if most_common:
                        # signature-namespaced name -> no cross-signature collision
                        rel_name = f"{signature}_REL_{most_common[0][0].upper()}"
                        discovered[rel_name].extend(cluster_pairs)

            except (ValueError, AttributeError) as e:
                # [IMPROVEMENT 8] specific exceptions only (TF-IDF/DBSCAN raise
                # ValueError on degenerate input); logged at debug.
                logger.debug(f"Clustering failed for {signature}: {e}")
                continue

        self.discovered_relations = discovered
        logger.info(f"[Distant Supervision] Discovered {len(discovered)} relation types")
        return discovered


# ════════════════════════════════════════════════════════════════════════════
# ALGORITHM 2: FRAME-BASED SEMANTIC SLOT FILLING
# ════════════════════════════════════════════════════════════════════════════

class FrameSlottingModule:
    """
    Frame semantics: detect frames and fill slots using universal semantic roles.
    """

    def __init__(self, frames: Dict = None):
        self.frames = frames or UNIVERSAL_FRAMES
        self.detected_frames = []

    def detect_and_fill_frames(self, doc: spacy.tokens.Doc) -> List[FrameInstance]:
        """Detect and fill frames in document."""
        filled_frames = []

        for sent in doc.sents:
            # [IMPROVEMENT 5] A sentence may activate MORE THAN ONE frame.
            # The original `_detect_frame_in_sentence` returned after the first
            # trigger, so a sentence like "...took a job with GE... where he
            # married Lucile..." yielded employment OR marriage, never both,
            # losing recall on exactly the dense biographical sentences BOOFS
            # targets. We now fill every detected frame; slot-filling and
            # consolidation are unchanged because they already operate per-frame.
            for frame in self._detect_frames_in_sentence(sent):
                self._fill_frame_slots(frame, sent)
                if frame.slots:  # only keep frames that actually filled something
                    filled_frames.append(frame)

        self.detected_frames = filled_frames
        logger.info(f"[Frame Filling] Detected and filled {len(filled_frames)} frames")
        return filled_frames

    def _detect_frames_in_sentence(self, sent) -> List[FrameInstance]:
        """
        [IMPROVEMENT 5 + 6] Return ALL frames triggered in the sentence (deduped
        by frame type, keeping the first triggering token), with POS gating.

        POS gating: a trigger token only activates a frame when its
        part-of-speech is in that frame's `trigger_pos` set. This is what stops
        "serve dinner" (VERB 'serve' is allowed, but EMPLOYMENT needs an ORG/GPE
        employer to actually emit a relation downstream) and, more importantly,
        stops noun homographs from firing verb frames.
        """
        found = {}
        for token in sent:
            for frame_name, frame_def in self.frames.items():
                if frame_name in found:
                    continue  # already have this frame type for this sentence
                allowed_pos = frame_def.get('trigger_pos', {'VERB'})
                if token.pos_ in allowed_pos and token.lemma_ in frame_def['triggers']:
                    found[frame_name] = FrameInstance(frame_name, token.lemma_)
        return list(found.values())

    # [IMPROVEMENT 5] Backward-compatible shim: any external caller of the old
    # singular method still works (returns the first detected frame or None).
    def _detect_frame_in_sentence(self, sent) -> Optional[FrameInstance]:
        frames = self._detect_frames_in_sentence(sent)
        return frames[0] if frames else None

    def _fill_frame_slots(self, frame: FrameInstance, sent):
        """Fill slots based on semantic roles."""
        frame.sentence = sent.text
        entities = {ent.root: ent for ent in sent.ents}

        # Locate the trigger token for this frame.
        trigger_token = None
        for token in sent:
            if token.lemma_ == frame.trigger and \
               token.pos_ in self.frames[frame.frame_type].get('trigger_pos', {'VERB', 'NOUN'}):
                trigger_token = token
                break

        if not trigger_token:
            return

        # [IMPROVEMENT 7] Negation detection. If the trigger verb has a `neg`
        # child ("did NOT marry", "never worked"), mark the frame negated so
        # consolidation can suppress the positive assertion (or down-weight it).
        if any(child.dep_ == "neg" for child in trigger_token.children):
            frame.negated = True

        # --- Pass 1: named entities -> slots, scored by role-confidence -------
        for token in sent:
            if token in entities:
                entity = entities[token]
                role, role_conf = self._assign_semantic_role(token, trigger_token)

                for slot_name, slot_def in self.frames[frame.frame_type]['slots'].items():
                    required_role = slot_def['role']
                    allowed_types = slot_def['ner_types']

                    if role == required_role and entity.label_ in allowed_types:
                        # [IMPROVEMENT 10 + 12] The original line
                        #   confidence = 0.9 if entity.label_ in allowed_types else 0.6
                        # sat INSIDE a branch already guaranteeing the condition
                        # true, so it was always 0.9 and the 0.6 was dead. We now
                        # scale a base confidence by the *role* confidence
                        # gradient, so a directly-attached subject scores higher
                        # than a preposition-inferred one.
                        confidence = round(CONFIG.slot_base_entity * role_conf, 3)
                        frame.add_slot(slot_name, entity.text, confidence)

        # --- Pass 2: noun chunks -> NOUN-typed slots (POSITION/FIELD/etc.) ----
        # These slots use ner_types=['NOUN'] (a POS tag, never an NER label), so
        # they are unfillable from sent.ents alone. We match noun chunks by role,
        # skipping any chunk already covered as a named entity.
        entity_spans = {(ent.start, ent.end) for ent in entities.values()}
        for chunk in sent.noun_chunks:
            if (chunk.start, chunk.end) in entity_spans:
                continue
            role, role_conf = self._assign_semantic_role(chunk.root, trigger_token)
            for slot_name, slot_def in self.frames[frame.frame_type]['slots'].items():
                if slot_name in frame.slots:
                    continue
                if slot_def['role'] == role and 'NOUN' in slot_def['ner_types']:
                    text = chunk.text
                    if chunk[0].pos_ == 'DET' and len(chunk) > 1:
                        text = chunk[1:].text  # strip leading "the"/"a"/"an"
                    # [IMPROVEMENT 12] noun-chunk fills are scored from a lower
                    # base (a chunk role is a weaker signal than a named entity).
                    frame.add_slot(slot_name, text, round(CONFIG.slot_base_noun * role_conf, 3))

    def _assign_semantic_role(self, entity_token, trigger_token):
        """
        Assign a semantic role AND a confidence for that assignment.

        [IMPROVEMENT 12] Return signature changed from `role` to
        `(role, role_confidence)`. The confidence encodes HOW the role was
        inferred:
          - direct attachment to the trigger  -> role_conf_direct  (strongest)
          - inferred through a preposition     -> role_conf_prep
          - generic dependency-label fallback  -> role_conf_fallback
          - nothing specific found             -> role_conf_context (weakest)
        This feeds slot-fill calibration; it does not change which role is chosen.

        [IMPROVEMENT 10] The original method had TWO trailing `return "CONTEXT"`
        statements; the second was unreachable dead code and is removed.
        """
        # Direct attachment to the trigger is the strongest, least ambiguous signal.
        if entity_token.head == trigger_token:
            if entity_token.dep_ == 'nsubj':
                return ('AGENT', CONFIG.role_conf_direct)
            if entity_token.dep_ == 'nsubjpass':
                return ('PATIENT', CONFIG.role_conf_direct)
            if entity_token.dep_ == 'dobj':
                return ('PATIENT', CONFIG.role_conf_direct)
            if entity_token.dep_ == 'attr':
                return ('ATTRIBUTE', CONFIG.role_conf_direct)

        current = entity_token
        prep_chain = []
        connected = False
        # Clause-boundary guard: stop if we cross into a different clause
        # (relcl/advcl/ccomp) or pass through another finite verb, so an entity
        # from a marriage clause can't fill an employment slot via the conj chain.
        for _ in range(6):
            if current == trigger_token:
                connected = True
                break
            if current.dep_ in ('relcl', 'advcl', 'ccomp'):
                break
            if current.pos_ == 'VERB' and current != trigger_token:
                break
            if current.dep_ == 'prep':
                prep_chain.append(current.lemma_.lower())
            nxt = current.head
            if nxt == current:
                break
            current = nxt

        if connected and prep_chain:
            nearest_prep = prep_chain[0]
            ent_type = entity_token.ent_type_

            if ent_type == 'DATE' and nearest_prep in ('since', 'in', 'on', 'during', 'until', 'from', 'to'):
                return ('TEMPORAL', CONFIG.role_conf_prep)
            if ent_type in ('GPE', 'LOC') and nearest_prep in ('at', 'in', 'near'):
                return ('LOCATION', CONFIG.role_conf_prep)

            if nearest_prep == 'as':
                return ('ATTRIBUTE', CONFIG.role_conf_prep)
            if nearest_prep in ('of', 'with', 'for', 'by', 'at'):
                return ('PATIENT', CONFIG.role_conf_prep)

        # Fallback: original direct dependency-label mapping.
        dep = entity_token.dep_
        if dep in SEMANTIC_ROLE_MAPPING:
            return (SEMANTIC_ROLE_MAPPING[dep], CONFIG.role_conf_fallback)

        return ('CONTEXT', CONFIG.role_conf_context)


# ════════════════════════════════════════════════════════════════════════════
# ALGORITHM 3: UNSUPERVISED RELATION DISCOVERY  (unchanged)
# ════════════════════════════════════════════════════════════════════════════

class UnsupervisedRelationDiscoveryModule:
    """Discover relation patterns via distributional clustering."""

    def __init__(self):
        self.entity_contexts = defaultdict(list)
        self.discovered_patterns = []

    def build_context_profiles(self, doc: spacy.tokens.Doc):
        for sent in doc.sents:
            entities = [(ent.text.lower(), ent.label_, ent.root) for ent in sent.ents]
            for entity_text, ent_type, root_token in entities:
                context = {
                    'text': entity_text,
                    'type': ent_type,
                    'left': [t.lemma_ for t in root_token.lefts if not t.is_stop],
                    'right': [t.lemma_ for t in root_token.rights if not t.is_stop],
                    'sentence': sent.text
                }
                self.entity_contexts[entity_text].append(context)

    def discover_relations(self, min_cluster_size: int = 2) -> List[Dict]:
        if len(self.entity_contexts) < 3:
            logger.warning("Need at least 3 entities for discovery")
            return []

        entity_names = list(self.entity_contexts.keys())

        feature_vectors = []
        for entity in entity_names:
            contexts = self.entity_contexts[entity]
            all_words = []
            for ctx in contexts:
                all_words.extend(ctx['left'])
                all_words.extend(ctx['right'])
            feature_vectors.append(" ".join(all_words) if all_words else "EMPTY")

        try:
            vectorizer = TfidfVectorizer(max_features=50)
            X = vectorizer.fit_transform(feature_vectors)

            clustering = DBSCAN(eps=CONFIG.dbscan_eps_unsup, min_samples=min_cluster_size, metric='cosine')
            labels = clustering.fit_predict(X.toarray())

            patterns = []
            for cluster_id in set(labels):
                if cluster_id == -1:
                    continue

                cluster_entities = [entity_names[i] for i, l in enumerate(labels) if l == cluster_id]

                all_contexts = []
                for entity in cluster_entities:
                    for ctx in self.entity_contexts[entity]:
                        all_contexts.extend(ctx['left'])
                        all_contexts.extend(ctx['right'])

                signature = [w for w, c in Counter(all_contexts).most_common(3)]

                patterns.append({
                    'cluster_id': cluster_id,
                    'entities': cluster_entities,
                    'signature': signature,
                    'size': len(cluster_entities)
                })

            self.discovered_patterns = patterns
            logger.info(f"[Unsupervised Discovery] Found {len(patterns)} patterns")
            return patterns

        except (ValueError, AttributeError) as e:
            # [IMPROVEMENT 8] specific exceptions + debug log instead of bare except.
            logger.debug(f"Discovery failed: {e}")
            return []


# ════════════════════════════════════════════════════════════════════════════
# ALGORITHM 4: ACTIVE LEARNING  (unchanged)
# ════════════════════════════════════════════════════════════════════════════

class ActiveLearningModule:
    """Select the most informative examples for (optional) human labeling."""

    @staticmethod
    def compute_uncertainty(example: Dict) -> float:
        score = 0.0
        context = example.get('context', '')
        if len(context.split()) < 2:
            score += 0.3
        type1 = example.get('type1', '')
        type2 = example.get('type2', '')
        if type1 == type2:
            score += 0.3
        conf = example.get('confidence', 0.5)
        score += (1.0 - conf) * 0.4
        return min(score, 1.0)

    @staticmethod
    def select_informative_examples(candidates: List[Dict], k: int = 5) -> List[Dict]:
        if len(candidates) <= k:
            return candidates
        scored = [(ActiveLearningModule.compute_uncertainty(c), c) for c in candidates]
        scored.sort(reverse=True, key=lambda x: x[0])
        return [item[1] for item in scored[:k]]


# ════════════════════════════════════════════════════════════════════════════
# MAIN ONTOLOGY LEARNING SYSTEM
# ════════════════════════════════════════════════════════════════════════════

class BOOFSOntologyLearner:
    """
    Complete Ontology Learning System.
    Bootstrapped Ontology and Object Frame Semantics.
    """

    def __init__(self, frames: Dict = None):
        self.distant_supervisor = DistantSupervisionModule()
        self.frame_filler = FrameSlottingModule(frames)
        self.relation_discoverer = UnsupervisedRelationDiscoveryModule()
        self.active_learner = ActiveLearningModule()
        self.coref_resolver = CoreferenceResolver()
        self.kg_embedder = None

        self.concepts = []
        self.relations = []
        self.frames = []
        # [IMPROVEMENT 3] entity-similarity hypotheses kept separate from
        # asserted relations (also still mirrored into self.relations as
        # low-confidence SIMILAR_TO edges for backward-compatible single-file
        # consumers).
        self.similarity_hypotheses = []
        self.raw_text = None
        self.resolved_text = None

    def process(self, text: str, use_active_learning: bool = False, verbose: bool = True,
                resolve_coreference: bool = True):
        """Process text and extract ontology."""
        if verbose:
            print("\n" + "=" * 70)
            print("BOOFS: UNIVERSAL ONTOLOGY LEARNING SYSTEM")
            print("=" * 70)

        # Stage 0: coreference resolution.
        self.raw_text = text
        if resolve_coreference:
            if verbose: print("\n[0] Resolving coreferences...")
            text = self.coref_resolver.resolve(text)
            if verbose: print(f"    ✓ Coreference backend: {self.coref_resolver.backend or 'rule-based fallback'}")
        self.resolved_text = text

        doc = nlp(text)

        if verbose: print("\n[1] Extracting concepts...")
        self.concepts = self._extract_concepts(doc)
        if verbose: print(f"    ✓ Found {len(self.concepts)} concepts")

        if verbose: print("\n[2] Distant supervision - discovering entity pairs...")
        pairs = self.distant_supervisor.extract_entity_pairs(doc)
        discovered_rels = self.distant_supervisor.discover_relations_via_clustering()
        if verbose: print(f"    ✓ Discovered {len(discovered_rels)} relation types")

        if verbose: print("\n[3] Frame-based slot filling...")
        filled_frames = self.frame_filler.detect_and_fill_frames(doc)
        self.frames = filled_frames
        if verbose: print(f"    ✓ Detected {len(filled_frames)} frames with slots")

        if verbose: print("\n[4] Unsupervised relation discovery...")
        self.relation_discoverer.build_context_profiles(doc)
        patterns = self.relation_discoverer.discover_relations()
        if verbose: print(f"    ✓ Found {len(patterns)} distributional patterns")

        if verbose: print("\n[5] Consolidating relations...")
        self.relations = self._consolidate_relations(discovered_rels, filled_frames, patterns)
        if verbose: print(f"    ✓ Consolidated {len(self.relations)} unique relations")

        if verbose: print("\n[5b] Training knowledge graph embeddings...")
        triples = [(r.subject, r.relation, r.object) for r in self.relations]
        self.kg_embedder = KGEmbeddingModule()
        try:
            self.kg_embedder.train(triples)
            if verbose: print(f"    ✓ Trained embeddings on {len(triples)} triples")
        except Exception as e:
            logger.warning(f"KG embedding training skipped: {e}")

        if use_active_learning and len(pairs) > 5:
            if verbose: print("\n[6] Active learning...")
            to_label = self.active_learner.select_informative_examples(pairs, k=5)
            if verbose:
                print("    Please label these 5 examples:")
                for i, ex in enumerate(to_label, 1):
                    print(f"    {i}. ({ex['entity1']}) <-> ({ex['entity2']})")
                    print(f"       Context: {ex['context'][:50]}...")

        if verbose:
            print("\n" + "=" * 70)
            print("EXTRACTION COMPLETE")
            print("=" * 70)

        return {
            'concepts': self.concepts,
            'relations': self.relations,
            'frames': self.frames,
            'patterns': patterns,
            'similarity_hypotheses': self.similarity_hypotheses,  # additive output key
        }

    def _extract_concepts(self, doc) -> List[ConceptExtract]:
        concepts_dict = {}

        for ent in doc.ents:
            concept_id = ent.text.lower()
            if concept_id not in concepts_dict:
                c = ConceptExtract(concept_id, ent.label_, ent.text, confidence=CONFIG.conf_ner)
                c.sources.append('NER')
                concepts_dict[concept_id] = c

        for chunk in doc.noun_chunks:
            concept_id = chunk.lemma_.lower()
            if concept_id not in concepts_dict and len(concept_id.split()) > 1:
                c = ConceptExtract(concept_id, 'CONCEPT', chunk.text, confidence=CONFIG.conf_noun_chunk)
                c.sources.append('NOUN_CHUNK')
                concepts_dict[concept_id] = c

        return list(concepts_dict.values())

    # ------------------------------------------------------------------------
    # [IMPROVEMENT 1 + 12] canonical relation emission for ALL frame types
    # ------------------------------------------------------------------------
    # Maps each frame type to (subject_slot, RELATION_NAME, object_slot). The
    # original code only had hand-written branches for EMPLOYMENT and FOUNDING,
    # so EDUCATION/FAMILY/LOCATION frames — though correctly detected and filled
    # — never produced a clean relation and instead fell through to the generic
    # partial fallback (e.g. "EDUCATION_INSTITUTION"). This table makes all five
    # first-class while keeping the partial fallback strictly as a fallback.
    _CANONICAL_FRAME_RELATIONS = {
        'EMPLOYMENT': ('EMPLOYEE', 'WORKS_FOR', 'EMPLOYER'),
        'FOUNDING':   ('FOUNDER', 'FOUNDED', 'FOUNDED_ENTITY'),
        'EDUCATION':  ('STUDENT', 'STUDIED_AT', 'INSTITUTION'),
        'FAMILY':     ('PERSON1', 'FAMILY_RELATION', 'PERSON2'),
        'LOCATION':   ('ENTITY', 'LOCATED_IN', 'PLACE'),
    }

    @staticmethod
    def _calibrated_rel_conf(*slot_confidences) -> float:
        """[IMPROVEMENT 12] Derive a relation's confidence from the confidences
        of the slots that produced it (conservative: weakest slot dominates),
        instead of a single hardcoded literal like 0.85."""
        if not slot_confidences:
            return CONFIG.conf_frame_scale
        return round(min(slot_confidences) * CONFIG.conf_frame_scale, 3)

    def _consolidate_relations(self, discovered_rels, frames, patterns) -> List[RelationExtract]:
        relations_set = set()
        relations_list = []

        def _add(rel):
            if rel not in relations_set:
                relations_set.add(rel)
                relations_list.append(rel)

        # --- From distant supervision -----------------------------------------
        for rel_name, dpairs in discovered_rels.items():
            for pair in dpairs:
                rel = RelationExtract(pair['entity1'], rel_name, pair['entity2'],
                                      confidence=pair.get('confidence', CONFIG.conf_distant_base))
                rel.source = 'distant_supervision'
                rel.evidence = pair.get('context', '')
                _add(rel)

        # --- From frames ------------------------------------------------------
        for frame in frames:
            # [IMPROVEMENT 7] Negation gate. A negated frame must not assert its
            # positive fact. With skip_negated=True we drop it entirely; otherwise
            # we keep it but multiply confidence by the penalty so it ranks low.
            if frame.negated and CONFIG.skip_negated:
                continue
            neg_mult = CONFIG.negation_confidence_penalty if frame.negated else 1.0

            canonical = self._CANONICAL_FRAME_RELATIONS.get(frame.frame_type)
            emitted_canonical = False

            # [IMPROVEMENT 1] table-driven canonical emission for every frame type
            if canonical:
                subj_slot, rel_name, obj_slot = canonical
                if subj_slot in frame.slots and obj_slot in frame.slots:
                    subj_val, subj_conf = frame.slots[subj_slot]
                    obj_val, obj_conf = frame.slots[obj_slot]
                    conf = self._calibrated_rel_conf(subj_conf, obj_conf) * neg_mult
                    rel = RelationExtract(subj_val, rel_name, obj_val, confidence=round(conf, 3))
                    rel.source = 'frame_based' if not frame.negated else 'frame_based_negated'
                    rel.evidence = frame.sentence
                    _add(rel)
                    emitted_canonical = True

            # Partial fallback ONLY when no canonical relation was emitted, so a
            # frame's entities never silently vanish from the graph.
            if not emitted_canonical:
                frame_def = UNIVERSAL_FRAMES.get(frame.frame_type, {})
                slot_roles = frame_def.get('slots', {})
                agent_slot = next((s for s in frame.slots
                                   if slot_roles.get(s, {}).get('role') == 'AGENT'), None)
                if agent_slot is not None:
                    agent_value, agent_conf = frame.slots[agent_slot]
                    other_slots = [s for s in frame.slots if s != agent_slot]
                    if other_slots:
                        for other_slot in other_slots:
                            other_value, other_conf = frame.slots[other_slot]
                            conf = min(agent_conf, other_conf) * CONFIG.conf_partial_scale * neg_mult
                            rel = RelationExtract(agent_value,
                                                  f"{frame.frame_type}_{other_slot}",
                                                  other_value, confidence=round(conf, 3))
                            rel.source = 'frame_based_partial'
                            rel.evidence = frame.sentence
                            _add(rel)
                    else:
                        conf = agent_conf * 0.7 * neg_mult
                        rel = RelationExtract(agent_value, frame.frame_type, frame.trigger,
                                              confidence=round(conf, 3))
                        rel.source = 'frame_based_partial'
                        rel.evidence = frame.sentence
                        _add(rel)

        # --- From distributional patterns -------------------------------------
        # [IMPROVEMENT 3] The clustering groups entities by SIMILAR context, i.e.
        # it finds entities that are *alike* (all people, all companies) — it does
        # NOT find a relation BETWEEN them. The original code asserted a directed
        # REL_<id> edge between consecutive cluster members, manufacturing one
        # false edge per extra member. We keep the clustering and its output, but
        # re-label these as SIMILAR_TO hypotheses at low confidence, and also
        # record them separately in self.similarity_hypotheses. The module is NOT
        # removed; only the (incorrect) interpretation of its output changes.
        self.similarity_hypotheses = []
        for pattern in patterns:
            ents = pattern['entities']
            if len(ents) >= 2:
                for i in range(len(ents)):
                    for j in range(i + 1, len(ents)):
                        rel = RelationExtract(ents[i], 'SIMILAR_TO', ents[j],
                                              confidence=CONFIG.conf_similarity)
                        rel.source = 'distributional_similarity'
                        rel.evidence = f"cluster {pattern['cluster_id']} signature={pattern['signature']}"
                        self.similarity_hypotheses.append(rel)
                        _add(rel)

        return relations_list

    # ------------------------------------------------------------------------
    # EXPORTS  (existing format preserved for backward compatibility)
    # ------------------------------------------------------------------------
    def export_to_csv(self, prefix: str = "ontology"):
        """Export ontology to CSV files (same columns/format as before)."""
        with open(f"{prefix}_concepts.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["concept", "type", "surface", "confidence", "sources"])
            writer.writeheader()
            writer.writerows([c.to_dict() for c in self.concepts])
        logger.info(f"✓ Exported {len(self.concepts)} concepts to {prefix}_concepts.csv")

        with open(f"{prefix}_relations.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["subject", "relation", "object", "confidence", "source", "evidence"])
            writer.writeheader()
            writer.writerows([r.to_dict() for r in self.relations])
        logger.info(f"✓ Exported {len(self.relations)} relations to {prefix}_relations.csv")

        frame_rows = []
        for frame in self.frames:
            for slot_name, (slot_value, confidence) in frame.slots.items():
                frame_rows.append({
                    'frame_type': frame.frame_type,
                    'trigger': frame.trigger,
                    'slot_name': slot_name,
                    'slot_value': slot_value,
                    'confidence': round(confidence, 3),
                    'negated': frame.negated,  # additive column [IMPROVEMENT 7]
                })

        with open(f"{prefix}_frames.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["frame_type", "trigger", "slot_name", "slot_value", "confidence", "negated"])
            writer.writeheader()
            writer.writerows(frame_rows)
        logger.info(f"✓ Exported {len(frame_rows)} frame slots to {prefix}_frames.csv")

    # [IMPROVEMENT 3] dedicated export for the similarity hypotheses, so a
    # consumer can keep them out of the asserted-relations table if desired.
    def export_similarity_hypotheses_to_csv(self, prefix: str = "ontology"):
        with open(f"{prefix}_similarity_hypotheses.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["subject", "relation", "object", "confidence", "source", "evidence"])
            writer.writeheader()
            writer.writerows([r.to_dict() for r in self.similarity_hypotheses])
        logger.info(f"✓ Exported {len(self.similarity_hypotheses)} similarity hypotheses to "
                    f"{prefix}_similarity_hypotheses.csv")

    def export_embeddings_to_csv(self, prefix: str = "ontology"):
        if self.kg_embedder is None or not self.kg_embedder.is_trained:
            logger.warning("No trained KG embeddings available to export.")
            return
        self.kg_embedder.export_embeddings_to_csv(f"{prefix}_embeddings.csv")
        self.kg_embedder.export_link_predictions_to_csv(f"{prefix}_link_predictions.csv")
        self.kg_embedder.export_similarity_scores_to_csv(f"{prefix}_similarity_scores.csv")

    def print_summary(self):
        print("\n" + "=" * 70)
        print("ONTOLOGY SUMMARY")
        print("=" * 70)

        print(f"\n📦 CONCEPTS: {len(self.concepts)}")
        print("-" * 70)
        for i, c in enumerate(self.concepts[:10], 1):
            print(f"  {i:2}. {c.text:25} [{c.type:12}] conf={c.confidence:.2f}")
        if len(self.concepts) > 10:
            print(f"  ... and {len(self.concepts) - 10} more")

        print(f"\n🔗 RELATIONS: {len(self.relations)}")
        print("-" * 70)
        for i, r in enumerate(self.relations[:10], 1):
            print(f"  {i:2}. ({r.subject:20}) --[{r.relation:15}]--> ({r.object:20}) conf={r.confidence:.2f}")
        if len(self.relations) > 10:
            print(f"  ... and {len(self.relations) - 10} more")

        print(f"\n🎯 FRAMES: {len(self.frames)}")
        print("-" * 70)
        for i, f in enumerate(self.frames[:5], 1):
            slot_count = len(f.slots)
            neg = " (NEGATED)" if f.negated else ""
            print(f"  {i}. {f.frame_type} (trigger: {f.trigger}, {slot_count} slots filled){neg}")
            for slot_name, (value, conf) in list(f.slots.items())[:3]:
                print(f"     - {slot_name}: {value} (conf={conf:.2f})")
        if len(self.frames) > 5:
            print(f"  ... and {len(self.frames) - 5} more")

        print("\n" + "=" * 70)


# ════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH EMBEDDINGS
# ════════════════════════════════════════════════════════════════════════════
#
# Operates ONLY on the final consolidated triples. Does not touch relation
# extraction. Uses PyKEEN with RotatE (ComplEx optionally as a secondary model).

class KGEmbeddingModule:
    """Train KG embeddings on BOOFS's consolidated triples; support link
    prediction and entity similarity queries."""

    def __init__(self, embedding_dim: int = 50, num_epochs: int = 50, use_complex: bool = False):
        self.embedding_dim = embedding_dim
        self.num_epochs = num_epochs
        self.use_complex = use_complex
        self.is_trained = False
        # [IMPROVEMENT 9] True only when embeddings were evaluated on a genuine
        # held-out split. While False, Hits@K must NOT be reported (it would be
        # train-set evaluation and badly inflated).
        self.evaluation_valid = False
        self.triples_factory = None
        self.rotate_result = None
        self.complex_result = None
        self._entity_to_id = {}
        self._id_to_entity = {}

    def train(self, triples: List[Tuple[str, str, str]]):
        """Train RotatE (primary) and optionally ComplEx (secondary)."""
        from pykeen.triples import TriplesFactory
        from pykeen.pipeline import pipeline

        triples = [t for t in triples if t[0] and t[1] and t[2]]
        if len(triples) < 3:
            raise ValueError("Not enough triples to train KG embeddings (need >= 3).")

        triples_array = np.array(triples, dtype=str)
        self.triples_factory = TriplesFactory.from_labeled_triples(triples_array)
        self._entity_to_id = self.triples_factory.entity_to_id
        self._id_to_entity = {v: k for k, v in self._entity_to_id.items()}

        # [IMPROVEMENT 9] Build an honest train/test split when the graph is big
        # enough; otherwise train on everything and DISABLE evaluation rather
        # than reporting metrics computed on the training set.
        training_tf = self.triples_factory
        testing_tf = self.triples_factory
        if len(triples) >= CONFIG.min_triples_for_eval:
            try:
                training_tf, testing_tf = self.triples_factory.split(
                    [1.0 - CONFIG.kg_test_ratio, CONFIG.kg_test_ratio], random_state=42)
                self.evaluation_valid = True
            except Exception as e:
                logger.warning(f"Triple split failed ({e}); evaluation disabled.")
                training_tf = testing_tf = self.triples_factory
                self.evaluation_valid = False
        else:
            logger.info(
                f"Only {len(triples)} triples (< {CONFIG.min_triples_for_eval}); "
                f"Hits@K evaluation disabled to avoid train-set leakage.")
            self.evaluation_valid = False

        self.rotate_result = pipeline(
            training=training_tf,
            testing=testing_tf,
            model='RotatE',
            model_kwargs=dict(embedding_dim=self.embedding_dim),
            training_kwargs=dict(num_epochs=self.num_epochs, use_tqdm=False),
            random_seed=42,
        )

        if self.use_complex:
            self.complex_result = pipeline(
                training=training_tf,
                testing=testing_tf,
                model='ComplEx',
                model_kwargs=dict(embedding_dim=self.embedding_dim),
                training_kwargs=dict(num_epochs=self.num_epochs, use_tqdm=False),
                random_seed=42,
            )

        self.is_trained = True

    def predict_missing_links(self, top_k: int = 10):
        if not self.is_trained:
            raise RuntimeError("Call train() before predict_missing_links().")
        from pykeen.predict import predict_all
        predictions = predict_all(model=self.rotate_result.model, k=top_k)
        df = predictions.process(factory=self.triples_factory).df
        return df.head(top_k)

    def get_entity_similarity(self, entity: str, top_k: int = 5):
        if not self.is_trained:
            raise RuntimeError("Call train() before get_entity_similarity().")
        entity = entity.lower()
        if entity not in self._entity_to_id:
            return []

        entity_embeddings = self.rotate_result.model.entity_representations[0](indices=None).detach().cpu().numpy()
        if np.iscomplexobj(entity_embeddings):
            entity_embeddings = np.abs(entity_embeddings)

        target_idx = self._entity_to_id[entity]
        sims = cosine_similarity([entity_embeddings[target_idx]], entity_embeddings)[0]
        ranked = sorted(
            ((self._id_to_entity[i], float(s)) for i, s in enumerate(sims) if i != target_idx),
            key=lambda x: -x[1]
        )
        return ranked[:top_k]

    def export_embeddings_to_csv(self, filepath: str):
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["entity"] + [f"dim_{i}" for i in range(self.embedding_dim)])
            embeddings = self.rotate_result.model.entity_representations[0](indices=None).detach().cpu().numpy()
            if np.iscomplexobj(embeddings):
                embeddings = np.abs(embeddings)
            for entity, idx in self._entity_to_id.items():
                writer.writerow([entity] + list(embeddings[idx][:self.embedding_dim]))
        logger.info(f"✓ Exported entity embeddings to {filepath}")

    def export_link_predictions_to_csv(self, filepath: str, top_k: int = 20):
        try:
            df = self.predict_missing_links(top_k=top_k)
            df.to_csv(filepath, index=False)
            logger.info(f"✓ Exported link predictions to {filepath}")
        except Exception as e:
            logger.warning(f"Could not export link predictions: {e}")

    def export_similarity_scores_to_csv(self, filepath: str, top_k: int = 5):
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["entity", "similar_entity", "similarity"])
            for entity in self._entity_to_id:
                for similar_entity, score in self.get_entity_similarity(entity, top_k=top_k):
                    writer.writerow([entity, similar_entity, round(score, 4)])
        logger.info(f"✓ Exported similarity scores to {filepath}")


# ════════════════════════════════════════════════════════════════════════════
# EVALUATION METRICS
# ════════════════════════════════════════════════════════════════════════════

def evaluate_coreference_improvement(raw_text: str, resolved_text: str) -> Dict:
    """How many pronoun tokens were replaced by coreference resolution."""
    raw_doc, resolved_doc = nlp(raw_text), nlp(resolved_text)
    pronouns_before = sum(1 for t in raw_doc if t.text.lower() in CoreferenceResolver.PRONOUNS)
    pronouns_after = sum(1 for t in resolved_doc if t.text.lower() in CoreferenceResolver.PRONOUNS)
    resolved_count = max(pronouns_before - pronouns_after, 0)
    return {
        'pronouns_before': pronouns_before,
        'pronouns_after': pronouns_after,
        'pronouns_resolved': resolved_count,
        'resolution_rate': round(resolved_count / pronouns_before, 3) if pronouns_before else 0.0,
    }


def evaluate_relation_precision(relations_before: List['RelationExtract'],
                                 relations_after: List['RelationExtract'],
                                 sample_labels: Optional[Dict[Tuple[str, str, str], bool]] = None) -> Dict:
    """Proxy precision (fraction of pronoun-free endpoints), or true precision if
    gold `sample_labels` are supplied."""
    def pronoun_free_ratio(rels):
        if not rels:
            return 0.0
        clean = sum(1 for r in rels if r.subject not in CoreferenceResolver.PRONOUNS
                    and r.object not in CoreferenceResolver.PRONOUNS)
        return round(clean / len(rels), 3)

    result = {
        'proxy_precision_before': pronoun_free_ratio(relations_before),
        'proxy_precision_after': pronoun_free_ratio(relations_after),
    }

    if sample_labels:
        def labeled_precision(rels):
            labeled = [(r.subject, r.relation, r.object) for r in rels
                       if (r.subject, r.relation, r.object) in sample_labels]
            if not labeled:
                return None
            correct = sum(1 for t in labeled if sample_labels[t])
            return round(correct / len(labeled), 3)
        result['labeled_precision_before'] = labeled_precision(relations_before)
        result['labeled_precision_after'] = labeled_precision(relations_after)

    return result


def evaluate_hits_at_k(kg_embedder: 'KGEmbeddingModule', k: int = 10) -> Optional[float]:
    """Return Hits@k for the RotatE model — ONLY when it was evaluated on a real
    held-out split. [IMPROVEMENT 9] If the graph was too small for a split, the
    metric is suppressed (returns None) instead of reporting an inflated
    train-set number."""
    if kg_embedder is None or not kg_embedder.is_trained:
        return None
    if not getattr(kg_embedder, 'evaluation_valid', False):
        logger.info(f"Hits@{k} suppressed: graph too small for a valid held-out evaluation.")
        return None
    try:
        metrics = kg_embedder.rotate_result.metric_results.to_dict()
        return metrics.get('both', {}).get('realistic', {}).get(f'hits_at_{k}')
    except Exception as e:
        logger.warning(f"Could not extract Hits@{k}: {e}")
        return None


def evaluate_entity_similarity_quality(kg_embedder: 'KGEmbeddingModule', sample_size: int = 10) -> Dict:
    """Average top-1 similarity across a sample of entities (rough quality proxy)."""
    if kg_embedder is None or not kg_embedder.is_trained:
        return {'avg_top1_similarity': None, 'sampled_entities': 0}
    entities = list(kg_embedder._entity_to_id.keys())[:sample_size]
    scores = []
    for e in entities:
        sims = kg_embedder.get_entity_similarity(e, top_k=1)
        if sims:
            scores.append(sims[0][1])
    return {
        'avg_top1_similarity': round(sum(scores) / len(scores), 3) if scores else None,
        'sampled_entities': len(scores),
    }


# ════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    sample_text = """
    Bill and Dave became friends when they were both engineering students at Stanford.
    After graduation, Dave took a job with General Electric and moved to Schenectady,
    New York, where he married his college sweetheart Lucile Salter in 1938.
    But he and Bill stayed in touch. The two were encouraged by their former professor Fred Terman
    to start a technology company of their own.
    Taking a leave of absence from his job at GE, Dave and his new bride drove to California with
    a used drill press (an important piece of equipment for the new venture) in the rumble seat.
    Bill scouted for places where the newlyweds could live. He found the ideal rental at 367 Addison
    Avenue in Palo Alto for $45 per month. Dave and Lucile would live in the downstairs flat,
    while Bill would bunk in a tiny backyard shed where there was indoor plumbing and just enough
    room for a cot. But what made the property truly perfect for their needs was the
    small garage that the landlady told them they could use as a workshop.
    """

    learner = BOOFSOntologyLearner()
    results = learner.process(sample_text, use_active_learning=False, verbose=True)

    learner.export_to_csv("boofs_results")
    learner.export_similarity_hypotheses_to_csv("boofs_results")  # [IMPROVEMENT 3]
    learner.export_embeddings_to_csv("boofs_results")

    learner.print_summary()

    print("\n" + "=" * 70)
    print("BOOFS EXTENDED — EVALUATION METRICS")
    print("=" * 70)
    print(evaluate_coreference_improvement(learner.raw_text, learner.resolved_text))
    print(evaluate_relation_precision(learner.relations, learner.relations))
    print({'hits_at_10': evaluate_hits_at_k(learner.kg_embedder, k=10)})
    print(evaluate_entity_similarity_quality(learner.kg_embedder))

    print("\n✅ Ontology learning complete!")
    print("📁 Results saved to: boofs_results_[concepts|relations|frames|"
          "similarity_hypotheses|embeddings|link_predictions|similarity_scores].csv")


# ════════════════════════════════════════════════════════════════════════════
# CHANGE LOG  (all changes are additive/in-place; architecture unchanged)
# ════════════════════════════════════════════════════════════════════════════
#
# HIGH PRIORITY
#  [1] Canonical relations for EVERY frame type via _CANONICAL_FRAME_RELATIONS
#      (STUDIED_AT, FAMILY_RELATION, LOCATED_IN added; WORKS_FOR/FOUNDED kept).
#      Partial fallback retained strictly as a fallback.
#  [2] discover_relations_via_clustering() now accumulates into defaultdict(list)
#      with signature-namespaced names — no more silent overwrite of relations.
#  [3] Distributional pass kept, but its output is re-labeled SIMILAR_TO (low
#      confidence) and also exported separately as similarity hypotheses, instead
#      of being asserted as directed relations.
#  [4] Configurable spaCy model: prefers en_core_web_lg, falls back to _sm.
#
# MEDIUM PRIORITY
#  [5] Multiple frames per sentence (_detect_frames_in_sentence); old singular
#      method kept as a backward-compatible shim.
#  [6] Trigger lists reduced to lemmas only (dead inflected/duplicate entries
#      removed) and POS-gated via per-frame trigger_pos.
#  [7] Negation handling: FrameInstance.negated set from a `neg` dependency;
#      consolidation skips (or down-weights) negated assertions.
#  [8] Bare `except:` replaced with specific exceptions + debug logging.
#
# LOW PRIORITY
#  [9] KG evaluation leakage fixed: real train/test split when the graph is large
#      enough, Hits@K suppressed (evaluation_valid=False) otherwise.
# [10] Dead code removed: duplicate `return "CONTEXT"`; always-true 0.9/0.6
#      confidence ternary replaced by a real gradient.
#
# ADDITIONAL
# [11] Magic numbers centralized in BOOFSConfig.
# [12] Confidence calibration: role-inference gradient feeds slot confidence;
#      canonical relation confidence derived from its slots' confidences.
# [13] Entity pairing uses token distance, not a character heuristic.
# [14] Backward compatibility: existing CSV columns and return keys preserved;
#      new outputs (similarity_hypotheses, negated column) are purely additive.