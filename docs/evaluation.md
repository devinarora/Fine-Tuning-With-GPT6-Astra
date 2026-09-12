# Evaluation protocol

## Comparison

The original model and the validation-best fine-tuned checkpoint translate the same fixed French inputs. Both use deterministic beam search with four beams in the normal profile, along with the same tokenizer, output limit, precision policy, and metric settings.

Quality metrics score the generated English. A separate calculation measures the probability each model assigns to the reference translation.

| Metric | Measures | Better result |
|---|---|---|
| SacreBLEU | Corpus word n-gram precision with a brevity penalty, using tokenizer `13a` | Higher |
| chrF++ | Corpus character and word overlap, using word order 2 | Higher |
| TER | Translation edit rate with SacreBLEU defaults | Lower |
| BERTScore | Unscaled semantic precision, recall, and F1 against English references | Higher |
| NLL | Mean negative log probability per non-padding reference token | Lower |
| Perplexity | `exp(NLL)`, or null on numerical overflow | Lower |

BLEU and chrF++ use a 0 to 100 scale. TER can exceed 100. BERTScore is neither multiplied by 100 nor rescaled against a baseline. NLL and perplexity depend on the tokenizer, so this comparison uses the same tokenizer and references for both models. Logged Trainer validation loss is separate from token-weighted test NLL.

BERTScore uses [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base) at commit `e2da8e2f811d1448a5b465c236feacd80ffbac7b`, layer 10, with `idf=False` and `rescale_with_baseline=False`. It runs after the translation model has been released from memory. Inputs that exceed the scorer's token limit cause an error rather than silent truncation.

RoBERTa's unused language-model head and pooler can appear in loading messages. BERTScore uses hidden token representations, not the pooler.

## Breakdowns and uncertainty

Metrics are reported overall, by source, and by source-token length bands: 1 to 32, 33 to 64, 65 to 128, and 129 to 256. Empty bands are omitted. Counts and reference token totals accompany each group.

The normal report uses 1,000 paired bootstrap replicates. Each replicate resamples within each source and uses the same example indices for both models. SacreBLEU reuses corpus statistics to avoid repeated tokenization. The report gives percentile 95% intervals for changes in BLEU, chrF++, TER, and mean BERTScore F1.

Differences are fine-tuned minus baseline. A negative TER difference indicates improvement. An interval that includes zero does not establish an improvement. These intervals describe uncertainty in the test sample, not variation across training seeds or pretraining overlap. Literary passages may remain correlated because book-level identifiers are unavailable.

## Diagnostics and examples

Diagnostic counts flag empty translations, copied French, repeated trigrams, numeric mismatches, and output-limit hits. They are inspection aids rather than labeled error rates. Valid numeric formatting or repeated phrases can trigger flags.

Local artifacts also record resource and timing diagnostics for running the pipeline. Keep those fields out of public quality summaries; see [sharing results](workflow.md#sharing-results).

The examples file contains exactly two examples, one per dataset, selected before training. They support observation and have no role in model selection or scoring. There is no human-review stage.

## Limits

The base Marian model learned from OPUS material. Fine-tuning holdouts may overlap its pretraining data. These results support before-and-after comparisons on the selected sources, but cannot establish performance on entirely unseen material.

Books contains dated language, nonliteral translations, and alignment noise. OPUS-100 mixes domains and lacks reliable row-level spoken-language labels. Single-reference metrics can penalize valid alternatives, especially in literature. The report includes filtering counts and evaluates only the selected eligible length range.

## Source options and licenses

| Asset | Hugging Face link | Recorded license or status |
|---|---|---|
| Selected model | [Helsinki-NLP/opus-mt-fr-en](https://huggingface.co/Helsinki-NLP/opus-mt-fr-en) | Apache-2.0 |
| Selected literary data | [Helsinki-NLP/opus_books](https://huggingface.co/datasets/Helsinki-NLP/opus_books) | Hub metadata says `other`; inspect upstream terms |
| Selected mixed data | [Helsinki-NLP/opus-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100) | Hub metadata says `unknown`; inspect upstream terms |
| Model alternative | [google/flan-t5-small](https://huggingface.co/google/flan-t5-small) | Apache-2.0; requires a task-prefix adapter before use here |
| Larger model alternative | [Helsinki-NLP/opus-mt-tc-big-fr-en](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-fr-en) | CC-BY-4.0; recalibrate memory and timing |
| Spoken-data alternative | [IWSLT/iwslt2017](https://huggingface.co/datasets/IWSLT/iwslt2017) | CC-BY-NC-ND-4.0; additional loader work required |
| External test alternative | [facebook/flores](https://huggingface.co/datasets/facebook/flores) | CC-BY-SA-4.0; official access is gated |

These alternatives document the design choice. The supplied implementation and profiles target the selected Marian model and two ungated datasets.
