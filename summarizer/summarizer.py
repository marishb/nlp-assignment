import os
import sys
from collections import Counter

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')

CONTEXT_WINDOW_TOKENS = 1000


def measure_length(text):
    return len(word_tokenize(text))


def proportional_target_lengths(len1, len2):
    total = len1 + len2
    return int(CONTEXT_WINDOW_TOKENS * len1 / total), int(CONTEXT_WINDOW_TOKENS * len2 / total)


def chunk_text(text, max_tokens):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_tokens = 0
    for sent in sentences:
        sent_tokens = word_tokenize(sent)
        if current_tokens + len(sent_tokens) > max_tokens:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_tokens = 0
        current_chunk.append(sent)
        current_tokens += len(sent_tokens)
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks


def simple_summarize(text, max_tokens):
    sentences = sent_tokenize(text)
    summary = []
    current_tokens = 0
    for sent in sentences:
        sent_tokens = word_tokenize(sent)
        if current_tokens + len(sent_tokens) > max_tokens:
            break
        summary.append(sent)
        current_tokens += len(sent_tokens)
    return ' '.join(summary)


def hierarchical_summarize(text, max_tokens):
    while measure_length(text) > max_tokens:
        chunks = chunk_text(text, max_tokens)
        summaries = [simple_summarize(chunk, max_tokens) for chunk in chunks]
        text = ' '.join(summaries)
    return simple_summarize(text, max_tokens)


def analyze_style(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    avg_sent_len = sum(len(word_tokenize(s)) for s in sentences) / max(1, len(sentences))
    vocab = set(w.lower() for w in words if w.isalpha())
    word_freq = Counter(w.lower() for w in words if w.isalpha())
    return avg_sent_len, vocab, word_freq


def style_based_summarize(text, max_tokens, style_avg_len, style_vocab, style_word_freq):
    sentences = sent_tokenize(text)
    scored = []
    for sent in sentences:
        tokens = word_tokenize(sent)
        sent_len = len(tokens)
        len_score = -abs(sent_len - style_avg_len)
        vocab_overlap = sum(1 for w in tokens if w.lower() in style_vocab)
        freq_score = sum(style_word_freq[w.lower()] for w in tokens if w.lower() in style_word_freq)
        score = len_score + vocab_overlap + 0.1 * freq_score
        scored.append((score, sent, sent_len))
    scored.sort(reverse=True)
    summary = []
    current_tokens = 0
    for score, sent, sent_len in scored:
        if current_tokens + sent_len > max_tokens:
            continue
        summary.append(sent)
        current_tokens += sent_len
        if current_tokens >= max_tokens:
            break
    return ' '.join(summary)


def hierarchical_style_summarize(text, max_tokens, style_avg_len, style_vocab, style_word_freq):
    while measure_length(text) > max_tokens:
        chunks = chunk_text(text, max_tokens)
        summaries = [
            style_based_summarize(chunk, max_tokens, style_avg_len, style_vocab, style_word_freq) for chunk in chunks
        ]
        text = ' '.join(summaries)
    return style_based_summarize(text, max_tokens, style_avg_len, style_vocab, style_word_freq)


def process_documents_style(doc1, doc2, context_window_tokens=CONTEXT_WINDOW_TOKENS):
    len1 = measure_length(doc1)
    len2 = measure_length(doc2)
    target1, target2 = proportional_target_lengths(len1, len2)
    style_avg_len, style_vocab, style_word_freq = analyze_style(doc2)
    summary1 = hierarchical_style_summarize(doc1, target1, style_avg_len, style_vocab, style_word_freq)
    return summary1, style_avg_len, style_vocab


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python summarizer.py <document1_path> <document2_path> <output_base>")
        exit(1)
    doc1_path = sys.argv[1]
    doc2_path = sys.argv[2]
    output_base = sys.argv[3]
    if not os.path.exists(doc1_path) or not os.path.exists(doc2_path):
        print(f"Please provide valid paths for document1 and document2.")
        exit(1)
    with open(doc1_path, 'r', encoding='utf-8') as f:
        doc1 = f.read()
    with open(doc2_path, 'r', encoding='utf-8') as f:
        doc2 = f.read()

    summary1, style_avg_len, style_vocab = process_documents_style(doc1, doc2)

    with open(f'{output_base}_summary.txt', 'w', encoding='utf-8') as f:
        f.write(summary1)
    with open(f'{output_base}_query.txt', 'w', encoding='utf-8') as f:
        f.write(
            f"Summarized document1 in the style of document2. Style: avg sentence length {style_avg_len:.1f}, vocabulary size {len(style_vocab)}."
        )
    print(f"Summary saved as '{output_base}_summary.txt'. Query saved as '{output_base}_query.txt'.")
