"""Decisive test: is PTBTokenizer (used by every BLEU/ROUGE/CIDEr score in
this pipeline) mangling Turkish text?

pycocoevalcap's PTBTokenizer shells out to Stanford CoreNLP's Java
PTBTokenizer, a tool built for English Penn-Treebank-style tokenization
(splitting contractions like "don't" -> "do n't", English punctuation
conventions) and lower-cased with Java's default locale rules. It has never
been verified in this project to handle Turkish correctly: Turkish-specific
letters (ç, ğ, ı, ö, ş, ü, İ), the dotted/dotless I distinction under
lower-casing, and agglutinative suffix-heavy words could all be tokenized in
ways that silently break n-gram matching between generated and reference
captions -- even when both describe the same content in genuinely similar
words. If that is happening, every BLEU_4/ROUGE_L/CIDEr number this whole
project has reported is measuring tokenization noise, not caption quality.

This script tokenizes real reference/generated caption pairs (near-paraphrases,
i.e. captions that a person would call "clearly about the same thing") through
the exact same PTBTokenizer path eval.py uses, and prints the raw token
output side by side so a human can see directly whether Turkish is coming out
intact or corrupted -- e.g. whether "İki köpek" and "iki köpek" tokenize to
the same token, whether "köşesinde"/"köşede" stay intact, whether "ı"/"i" get
confused, whether words get split in unexpected places.
"""

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

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


def main():
    tok = PTBTokenizer()
    gts = {str(i): [{"caption": ref}] for i, (ref, _gen) in enumerate(PAIRS)}
    res = {str(i): [{"caption": gen}] for i, (_ref, gen) in enumerate(PAIRS)}

    gts_tok = tok.tokenize(gts)
    res_tok = tok.tokenize(res)

    for i, (ref, gen) in enumerate(PAIRS):
        key = str(i)
        print(f"=== pair {i} ===")
        print(f"  REFERENCE  raw      : {ref!r}")
        print(f"  REFERENCE  tokenized: {gts_tok[key]!r}")
        print(f"  GENERATED  raw      : {gen!r}")
        print(f"  GENERATED  tokenized: {res_tok[key]!r}")
        ref_words = set(gts_tok[key][0].split())
        gen_words = set(res_tok[key][0].split())
        overlap = ref_words & gen_words
        print(f"  shared tokens: {sorted(overlap)}")
        print()


if __name__ == "__main__":
    main()
