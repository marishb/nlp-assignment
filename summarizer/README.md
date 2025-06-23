# Hierarchical Style-Based Summarizer

A tool that summarizes a document in the style of another document, using a hierarchical approach for long texts.

## What it does

This program reads two documents:
- **Input document**: The text you want to summarize
- **Style document**: The text whose style you want the summary to follow (e.g., sentence length, vocabulary)

It produces a summary of the input document that matches the style of the style document, even for long texts that exceed a context window.

## How it works

1. **Analyzes the style document**: Finds average sentence length and key vocabulary
2. **Chunks the input document**: Splits into manageable pieces if it's too long
3. **Scores sentences**: Prefers sentences that match the style's length and vocabulary
4. **Hierarchical summarization**: Repeats chunking and summarizing until the summary fits the target size
5. **Outputs**: Writes the summary and a description of the style used

## NLP Methods Used

### Text Analysis
- **Sentence Tokenization**: Splits text into sentences
- **Word Tokenization**: Splits sentences into words
- **Vocabulary Extraction**: Finds common words in the style document
- **Sentence Length Analysis**: Calculates average sentence length for style matching

### Summarization
- **Extractive Summarization**: Selects sentences from the input document
- **Style-Based Scoring**: Scores sentences by how well they match the style document's features
- **Hierarchical Shrinking**: Repeats summarization on summaries for very long texts

## How to use it

### First time setup
```bash
pip install nltk
python -c "import nltk; nltk.download('punkt')"
```

### Summarize a document in the style of another
```bash
python summarizer.py <document1_path> <document2_path> <output_base>
```
- `<document1_path>`: Path to the document you want to summarize
- `<document2_path>`: Path to the style document
- `<output_base>`: Base path for output files (e.g., `result`)

This will create:
- `<output_base>_summary.txt`: The summary
- `<output_base>_query.txt`: A description of the style used

### Example
```bash
python summarizer.py tests/Technical_vs_Conversational/document1.txt tests/Technical_vs_Conversational/document2.txt tests/Technical_vs_Conversational/result
```

## Automated Testing

You can organize test cases in subfolders under `tests/`, each with:
- `document1.txt`
- `document2.txt`

Run all tests with:
```bash
python test_summarizer.py
```
Each test will output its summary and query in the same subfolder.

## What's in the files
- `summarizer.py` - The main summarization script
- `test_summarizer.py` - Automated test runner
- `tests/` - Folder with test case subfolders and example documents
- `README.md` - This file

## Limitations
- Works best with English text
- Style matching is based on sentence length and vocabulary, not deep semantics
- Extractive: only selects sentences from the input, does not rewrite
- Very short documents may not show much style transfer

This was built for an NLP assignment to demonstrate hierarchical and style-based summarization techniques. 