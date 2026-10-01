# SubjEx

SubjEx extracts grammatical subject spans from English text and assigns
rule-based structural labels. It uses spaCy's English transformer pipeline for
tokenization, sentence segmentation, part-of-speech tagging, and dependency
parsing.

The current version is research software. Its rules were designed for corpus
analysis and may not cover every grammatical construction or non-standard
learner sentence correctly.

## Related publication

### APA reference

Uchida, S., & Usukura, M. (2026). Developmental trends in subject length and
complexity in Japanese learners’ English writing. *Assessing Writing, 70*,
Article 101124. https://doi.org/10.1016/j.asw.2026.101124

[View the article on ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1075293526001121).

## Output

SubjEx returns one tab-separated row per detected subject, with no header:

```text
sentence<TAB>subject span<TAB>token count<TAB>labels
```

For example, the input `Birds migrate every autumn.` produces:

```text
Birds migrate every autumn.	Birds	1	Plural_only
```

Punctuation is excluded from the token count. Questions and sentences without
a detected subject do not produce output rows.

## Installation

SubjEx requires Python 3.9 or later.

```bash
python -m venv .venv
```

Activate the environment on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Then install SubjEx and the default spaCy model:

```bash
python -m pip install -e .
python -m spacy download en_core_web_trf
```

The transformer model provides the parsing behavior used during development.
It is substantially larger and slower than spaCy's small English model.

## Command-line usage

Analyze a UTF-8 text file and write the result to a TSV file:

```bash
subjex input.txt --output subjects.tsv
```

Standard input and output are also supported:

```bash
echo "To err is human." | subjex
```

Use another installed spaCy English pipeline when speed is more important than
matching the default model's parses:

```bash
subjex input.txt --model en_core_web_sm
```

Different models can produce different dependency parses and therefore
different SubjEx results.

## Python usage

```python
from subjex import classify_subjects

result = classify_subjects("To err is human.")
print(result)
```

The return value is a string containing zero or more tab-separated rows.

## Labels

Labels describe selected structural features of each subject span. A span may
receive more than one label. The implementation currently includes labels for:

- articles, determiners, adjectives, compounds, possessives, and prepositions;
- singular, plural, proper-noun, pronoun, numeral, and existential subjects;
- gerunds, participles, infinitives, coordination, and several clause types;
- specialized `it` patterns, relative clauses, quotations, and contact clauses.

Label names are preserved from the original analysis script for compatibility
with existing downstream data.

## Development and testing

Install the development dependency and run the test suite:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

Tests require the `en_core_web_trf` model.

## Data privacy

No learner corpus, source dataset, or generated research output is included in
this repository. Before analyzing sensitive text, review the storage and data
handling requirements that apply to your project.

## License

SubjEx is available under the MIT License. See [LICENSE](LICENSE).
