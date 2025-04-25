# Besprechung

## erste Ergebnisse

- BERTScore weist auf gute semantische Ähnlichkeit zur Referenz-Zusammenfassung
- Alle weiteren Scores sind vermutlich stark von der unterschiedlichen Länge der Referenz und generierten Zusammenfassung verzogen

## Gedanken - Warum ROUGE und BLEU so niedrig?

- Diese messen Wortübereinstimmungen -> Synonyme oder Paraphrasierungen werden schlecht gewertet. Auch abhängig von der Länge der Zusammenfassung, da der längere Teil natürlich stark in das Gewicht fällt.

Vielleicht noch folgende Metriken einbinden?

- BLUERT: trainiert auf menschliche Entscheidungen
- MoverScore: Wie Bert, aber nutzt Mover Earth Distance (Wie weit ein Satz "bewegt" werden muss, um die gleiche Bedeutung zu haben [Vektor])

## Fragen:

- deutsche Datensätze -> Wo und hochqulitative mit Referenzzusammenfassungen länger als ein Satz
- Multilinguale Modelle gibt es natürlich viele, aber für Evaluationen eher wenige -> Hast du da Ideen? Konnte mich aber selsbt noch nicht so viel damit auseinandersetzen
- Hauptproblem momentan ist vermutlich der Datensatz. Zu kurze Zusammenfassungen als Referenz. Wo kann ich bessere finden?
- Schwierig alle Parameter aus vLLM zu verstehen, aber ich bin da dran. Nur, wie nötig ist das momentan, sonst würde ich das auf wann anders bzw. meine Freizeit verschieben.

## Problem

- Viele Evaluationsbibliotheken benötigen unterschiedliche Python-Versionen und Versionen anderer Bibliotheken. Sehr mühselig alle Evaluationen an einem Ort zu sammeln -> Script nötig



<details><summary>toDo</summary>

# toDO

- script for different evaluation metrics due to version incompatibilities and different python environment version issues
- use mlsum german as dataset
- save model parameters to json for evaluation (path: summaries/dataset_name/model_name/index)
- save dataset name to json
- if pdf used, save path to pdf in evaluation json
- save evaluation scores to json
- build evaluation pipeline
- plot evaluations -> different metrics, different model configurations, different models

- gemma 1.5 flash

</details>

<details><summary>Evaluation file (json)</summary>

# Evaluation file (json):

- model name
- model configurations
- inference configurations
- dataset name
- path to dataset or PDF
- path to generated content
- evaluation scores -> for each infrence and mean
- easy to plot
- inference results and evaluation file have to be in the same folder. Folder name == model name -> enumerate model name (gemma0, gemma1, etc.)

</details>

<details><summary>nice to have</summary>

# nice to have

- quantization
- model-switch menu / GUI or terminal menu
- small model as fallback when big one won't fit in VRAM
- automatically detect free VRAM and adjust max-num-seqs

</details>

<details><summary>Model Parameters</summary>

# Model Parameters

https://docs.vllm.ai/en/latest/serving/engine_args.html

</details>

<details><summary>Evaluation</summary>

# Evaluation

This chapter explains the different metrics.

<details><summary>Precision and Recally</summary>

For precision and recall: Consider a computer program for recognizing dogs (the relevant element) in a digital photograph. Upon processing a picture which contains ten cats and twelve dogs, the program identifies eight dogs. Of the eight elements identified as dogs, only five actually are dogs (true positives), while the other three are cats (false positives). Seven dogs were missed (false negatives), and seven cats were correctly excluded (true negatives). The program's precision is then 5/8 (true positives / selected elements) while its recall is 5/12 (true positives / relevant elements). (https://en.wikipedia.org/wiki/Precision_and_recall Retreived 24th April 2025)

</details>

## ROUGE

The ROUGE metric measures the overlap between the generated summary and the reference summary in terms of n-grams, word sequences, and longest common subsequences.

- ROUGE-1: Measures overlap of unigrams (individual words). -> Einzel-Wortübereinstimmungen (Häufigkeit von Vokabular)
- ROUGE-2: Measures overlap of bigrams (two-word sequences). -> Wortpaar-Übereinstimmungen (Fluss & Präzision)
- ROUGE-L: Measures the longest common subsequence (LCS). -> längste übereinstimmende Teilsequenz (Wortrheinfolge/Struktur)

A score closer to 1.0 mean higher overlap and is considered the same as the reference at 1.0 (100% overlap).

It measures recall (How many selected items are right from all relevant items?).

## BLEU

Precision of n-grams in the generated summary compared to the reference summary. It also penalizes overly short outputs using a brevity penalty.

A score closer to 1.0 is considered "better" as it means a higher overlap with the reference summary.

It mainly focuses on precision (How many of the selected items are right?).

## METEOR

Precision and recall of unigrams (single word), with additional features like stemming (Grundwortart), synonyms, and word order.

Closer to 1.0 indicates better precision and recall.

More robust than BLEU for capturing semantic similarity and word variations.

## BERTScore

Semantic similarity between the generated and reference summaries using contextual embeddings from BERT.

- Precision: Wie viel von dem, was dein Modell sagt, auch im Referenztext enthalten ist (inhaltlich).
- Recall: Wie viel vom Referenzinhalt dein Modell tatsächlich erfasst hat.
- F1: Der harmonische Mittelwert von Precision und Recall — ein guter Gesamtwert.

A score closer to 1.0 (or 100%) indicates high semantic similarity.

Captures meaning rather than exact word overlap, making it suitable for abstractive summarization.

## Moverscore

Combines contextual embeddings (like BERT) with Earth Mover's Distance (How much meaning do I have to shift around to turn one sentence into the other?) to measure semantic similarity.

A score closer to 1.0 indicates a hihger semantic similarity between the generated and reference summary.

Captures both semantic similarity and word alignment, making it robust for abstractive summarization.

</details>

<details><summary>Set up</summary>

# Set up python environment

## Install Dependencies

```shell
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev git
```

## Install pyenv

```shell
curl https://pyenv.run | bash
```

Then, add this to your ~/.bashrc:

```shell
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

Reload shell:

```shell
source /.bashrc
```

## Install desired python version

```shell
pyenv install 3.10.13
```

## Create virtualenv for project

```shell
pyenv virtualenv 3.10.13 vLLM-serving
```

## In project directory

```shell
pyenv local vLLM-serving
```

The environment will be activated each time, you go into the directory via shell

</details>
