import os
import subprocess

tests_dir = os.path.join('tests')
os.makedirs(tests_dir, exist_ok=True)


def get_test_case_dirs():
    return [os.path.join(tests_dir, d) for d in os.listdir(tests_dir) if os.path.isdir(os.path.join(tests_dir, d))]


def run_test(test_case_dir):
    doc1_path = os.path.join(test_case_dir, 'document1.txt')
    doc2_path = os.path.join(test_case_dir, 'document2.txt')
    output_base = os.path.join(test_case_dir, 'result')
    if not (os.path.exists(doc1_path) and os.path.exists(doc2_path)):
        print(f"Skipping {test_case_dir}: missing document1.txt or document2.txt")
        return
    subprocess.run(['python3', 'summarizer.py', doc1_path, doc2_path, output_base], check=True)
    with open(f'{output_base}_summary.txt', 'r', encoding='utf-8') as f:
        summary = f.read()
    with open(f'{output_base}_query.txt', 'r', encoding='utf-8') as f:
        query = f.read()
    print(f"\n=== Test: {os.path.basename(test_case_dir)} ===")
    print(f"Summary (first 300 chars):\n{summary[:300]}\n...")
    print(f"Query: {query}")


def main():
    test_case_dirs = get_test_case_dirs()
    if not test_case_dirs:
        print("No test case subfolders found in tests folder.")
        return
    for test_case_dir in test_case_dirs:
        run_test(test_case_dir)


if __name__ == '__main__':
    main()
