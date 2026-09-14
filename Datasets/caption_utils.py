import re


def turkish_lower(text):
    """Python's str.lower() is not Turkish-aware: 'İ' (U+0130, capital
    dotted I) maps to 'i' + a COMBINING DOT ABOVE (U+0307) -- two
    characters, not the single plain 'i' Turkish orthography actually
    uses. Every training caption starting with "İki" ("two", extremely
    common) or containing any other capital İ was silently corrupted this
    way before ever reaching the tokenizer, and the model faithfully
    learned to reproduce the same corrupted "i̇ki..." in its own output
    (visible throughout eval logs). Replace the Turkish-specific letters
    before falling back to the ordinary (correct for everything else)
    .lower(), matching how e.g. ICU/locale-aware Turkish casing handles
    this pair.
    """
    return text.replace('İ', 'i').replace('I', 'ı').lower()


def pre_caption(caption, max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",
        ' ',
        turkish_lower(caption),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption