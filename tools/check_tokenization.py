"""Decisive test: is PTBTokenizer (used by every BLEU/ROUGE/CIDEr score in
this pipeline) mangling Turkish text -- and is caption LENGTH also dragging
the scores down independently of tokenization?

pycocoevalcap's PTBTokenizer shells out to Stanford CoreNLP's Java
PTBTokenizer (`-lowerCase`, then strips a hardcoded PUNCTUATIONS list --
see pycocoevalcap/tokenizer/ptbtokenizer.py). Two things this means, read
directly from that source rather than assumed:

  1. Casing and punctuation (periods, commas, quotes) are normalized
     identically on BOTH the reference and the generated side before
     scoring -- so "İşçiler çalışıyor." vs "işçiler çalışıyor" differing
     only in a leading capital letter and a trailing period is NOT, by
     itself, a plausible explanation for a near-zero score. Don't spend
     effort matching punctuation/casing style to the references; PTBTokenizer
     already erases that difference on both sides.
  2. The one real Turkish-specific risk is dotted/dotless I under
     lower-casing: Java's default-locale `toLowerCase()` (not
     `tr_TR`-aware) turns "İ" (U+0130) into "i" + a stray combining dot
     above (U+0307), a 2-codepoint sequence, rather than plain "i". This
     script's PAIRS deliberately include that case so it's visible whether
     it lands consistently (harmless -- same garbled form on both sides,
     n-grams still match) or inconsistently (harmful -- silently breaks
     matches).

Separately -- and NOT a tokenization question at all -- tools/check_pairing.py
runs and this project's own `eval.py` `avg_caption_len` diagnostic showed
that every Stage-2 MobileCLIP checkpoint that finished at least one real
eval settles at ~9.8-10.2 words/caption, while Data/tasvir-et's own
reference captions average ~8.1-8.3 words/caption (see the `--length-stats`
section below, computed directly from the dataset JSON, no Java needed).
A ~20-25% longer hypothesis than every reference it's scored against will
lose BLEU precision (extra words matching no reference n-gram) and CIDEr
TF-IDF mass purely from verbosity, even when the extra words are accurate.
This script reports both checks together because "is the caption wrong" and
"is the caption longer than what the metric rewards" are easy to conflate
from a Bleu_4 number alone, but need different fixes (more training / better
conditioning vs. e.g. length-aware decoding, min/max_length tuning).
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# SafePTBTokenizer (utils.py) is a drop-in PTBTokenizer replacement that
# filters leaked JVM log lines (e.g. Colab's cgroup-detection warning) out
# of the tokenizer's stdout before pairing lines to image ids -- without it,
# each leaked line silently shifts every caption after it to the wrong
# image (this is what this script originally caught: pair N's "tokenized"
# reference/generated text turning out to be pair N-2's). eval.py uses the
# same class, so this isn't just a diagnostic-script quirk -- it's the exact
# code path every real Bleu_4/CIDEr number in this project is computed
# through.
from utils import SafePTBTokenizer as PTBTokenizer

# Deliberately close near-paraphrases (same content, different surface
# wording/morphology) plus a few known-tricky Turkish forms (dotted/dotless
# I, ç/ğ/ş/ö/ü, agglutinative suffixes).
PAIRS = [
    ("İki köpek çimenlik alanda birbirleriyle koşuşturuyor.",
     "iki köpek yeşil çimlerin üzerinde koşuyorlar"),
    ("Suda koşmakta olan ıslanmış bir köpek.",
     "suyun içinde koşan beyaz renkli bir köpek"),
    ("Bir kadın kırmızı bisikletiyle yolda ilerliyor.",
     "kırmızı bir bisikletle yolda giden bir kadın"),
    ("İşçiler şantiyede çalışıyorlar.",
     "i̇şçiler şantiyede çalışmaktadır"),
]

# Real (reference, generated) pairs taken verbatim from a real Stage 2
# checkpoint's tools/check_pairing.py output (mobileclip_s0_stage2,
# model_best.pth -- see devam.md / colab_deneyler.ipynb cell 53), not
# synthetic. Included so the PTBTokenizer check below is also exercised
# against what the pipeline actually produced, not just hand-written
# near-paraphrases.
REAL_PAIRS = [
    ("Suda koşmakta olan ıslanmış bir köpek.",
     "sığ sularda koşmakta olan kahverengi bir köpek ve onu izleyen kahverengi bir köpek"),
    ("Ormanlık bir alanda ağaçların çevrelediği bir nehirde sandallarıyla yol almakta olan insanlar.",
     "i̇ki kişi nehirde kanolarıyla ilerliyorlar"),
    ("Sokakta dolaşan insanlar.",
     "i̇nsanlar sokakta yürüyor, kalabalık bir grup insan"),
]

# The exact punctuation list pycocoevalcap's PTBTokenizer strips after
# Java tokenizes+lowercases (see the docstring above) -- reproduced here,
# NOT reimplemented differently, so this fallback cannot silently diverge
# from what eval.py actually scores with.
from pycocoevalcap.tokenizer.ptbtokenizer import PUNCTUATIONS


def simple_tokenize(caption: str):
    """A trivial, Java-free stand-in: lowercase + whitespace-split + drop
    the same PUNCTUATIONS tokens PTBTokenizer drops. No contraction
    splitting, no locale-aware casing -- deliberately dumb. Printed next to
    PTBTokenizer's real output so a divergence between "the sophisticated
    English-tuned tokenizer" and "the dumbest thing that could work" is
    visible directly, instead of assumed. Python's own str.lower() (not
    Java's) IS Unicode-aware for Turkish ı/İ by default in the sense that it
    follows Unicode casefolding rules, not the Turkish-specific dotless-I
    mapping either -- so this is a comparison point, not a guaranteed fix.
    """
    words = caption.lower().split()
    return [w.strip("".join(c for c in ".,!?;:'\"" )) for w in words if w not in PUNCTUATIONS]


def run_pair_checks(tok, pairs, label):
    gts = {str(i): [{"caption": ref}] for i, (ref, _gen) in enumerate(pairs)}
    res = {str(i): [{"caption": gen}] for i, (_ref, gen) in enumerate(pairs)}

    gts_tok = tok.tokenize(gts)
    res_tok = tok.tokenize(res)

    print(f"\n{'=' * 20} {label} {'=' * 20}")
    for i, (ref, gen) in enumerate(pairs):
        key = str(i)
        print(f"\n=== pair {i} ===")
        print(f"  REFERENCE  raw            : {ref!r}")
        print(f"  REFERENCE  PTBTokenizer   : {gts_tok[key]!r}")
        print(f"  REFERENCE  simple_tokenize: {simple_tokenize(ref)!r}")
        print(f"  GENERATED  raw            : {gen!r}")
        print(f"  GENERATED  PTBTokenizer   : {res_tok[key]!r}")
        print(f"  GENERATED  simple_tokenize: {simple_tokenize(gen)!r}")
        ref_words = set(gts_tok[key][0].split())
        gen_words = set(res_tok[key][0].split())
        overlap = ref_words & gen_words
        simple_overlap = set(simple_tokenize(ref)) & set(simple_tokenize(gen))
        print(f"  PTBTokenizer shared tokens: {sorted(overlap)}")
        print(f"  simple_tokenize shared    : {sorted(simple_overlap)}")
        if len(simple_overlap) > len(overlap):
            print(f"  >>> DIVERGENCE: simple_tokenize finds {len(simple_overlap)} shared "
                  f"words, PTBTokenizer only {len(overlap)} -- PTBTokenizer is losing a real match.")


def report_length_stats():
    print(f"\n{'=' * 20} REFERENCE CAPTION LENGTH (Data/tasvir-et) {'=' * 20}")
    print("Compare this against eval.py's per-run 'avg_caption_len' diagnostic "
          "(printed in train.py's log at every validation) -- if generated "
          "captions run consistently longer than these references, BLEU/CIDEr "
          "will look worse than the captions actually are, independent of any "
          "tokenization bug.")
    for split in ["tasvir_train.json", "tasvir_val.json", "tasvir_test.json"]:
        path = REPO_ROOT / "Data" / "tasvir-et" / split
        if not path.is_file():
            print(f"  {split}: not found locally ({path})")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        anns = data.get("annotations", [])
        if not anns:
            continue
        lens = [len(a["caption"].split()) for a in anns]
        print(f"  {split}: {len(anns)} caption, ortalama={sum(lens) / len(lens):.2f} kelime, "
              f"min={min(lens)}, max={max(lens)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-length-stats", action="store_true",
                        help="Only run the PTBTokenizer pair checks (e.g. if Data/tasvir-et isn't present).")
    args = parser.parse_args()

    tok = PTBTokenizer()
    run_pair_checks(tok, PAIRS, "SYNTHETIC NEAR-PARAPHRASE PAIRS")
    run_pair_checks(tok, REAL_PAIRS, "REAL PIPELINE OUTPUT (mobileclip_s0_stage2, check_pairing.py)")

    if not args.skip_length_stats:
        report_length_stats()


if __name__ == "__main__":
    main()
