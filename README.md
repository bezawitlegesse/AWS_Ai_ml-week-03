# AWS_Ai_ml-week-03
# Break the Spam Filter (Week 3)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bezawitlegesse/AWS_Ai_ml-week-03/blob/main/AWS_Ai_ml_task_week_3_spam_filter.ipynb)

AI/ML AWS AASTU Cohort, Week 3 (Classical ML and Measuring Performance).

A spam filter can score high accuracy and still be easy to fool. This project trains the 10-line spam detector from the lecture
(TF-IDF + Naive Bayes) on SMS messages, measures its accuracy, then **tries to break it** with hand-written messages and explains why each one fails.
A bonus section tests a stronger model on the same messages.

## Dataset

**SMS Spam Collection**, UCI Machine Learning Repository: <https://archive.ics.uci.edu/dataset/228/sms+spam+collection>

It has 5,574 English text messages (4,827 normal "ham" and 747 spam).
The dataset is **not stored in this repo**. The notebook downloads it when you run it.
Citation: Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection* [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5CC84>

## What the notebook does

1. **Load** the raw dataset with Pandas
2. **Explore** it (class balance, message length statistics with NumPy, a histogram)
3. **Clean** it with functions: `drop_missing`, `normalize_text`, `remove_duplicates`, `convert_types`, `remove_length_outliers`
4. **Train** the 10-line model (80/20 split, `random_state=42`) and report accuracy, confusion matrix and a classification report
5. **Break it** with 13 attack messages (spelling tricks, negation, polite business tone, modern phishing, and more)
6. **Fix it** (bonus): character n-grams, word pairs, `deleet_text()` and Logistic Regression, compared on the same data

## Results

**Cleaning:** 5,574 raw messages became 5,141 (414 duplicate messages removed, then 19 extremely long messages), split 80/20 into 4,112 training and 1,029 test messages.

**Baseline accuracy: 95.24%** on the test set. That looks high, but the model has **0 false alarms and misses 49 spam messages** (spam recall 62.3%), so it catches only about 6 in 10 real spam messages.

### Messages that fooled the baseline model (6 of my 13 attacks)

All six were spam that slipped through.

| My message | Model said | Should be | Why it failed |
|---|---|---|---|
| "Cl@im your fr33 g1ft card now" | not spam | spam | Spelling trick: 'Cl@im', 'fr33' and 'g1ft' never appeared in the training data, so the model cannot link them to claim, free and gift. |
| "Y0u h4ve w0n 5000 b1rr! C@ll n0w to cl@im y0ur pr1ze" | not spam | spam | Almost every spam word is disguised, so very few known spam words are left for the model to count. |
| "Dear customer, we are pleased to inform you that your account has been selected for a reward. Kindly confirm your details at your earliest convenience." | not spam | spam | The spam in the training data is loud (FREE, WIN, URGENT). Calm, formal wording looks like normal mail. |
| "We noticed unusual activity on your bank account. Please verify your identity using the secure link below to avoid suspension." | not spam | spam | The dataset has older UK text messages, so modern phishing wording is mostly unseen. |
| "Dear customer, you have won 5000 birr in our mobile promotion. Visit the nearest shop with your ID to receive your prize." | not spam | spam | Prize and currency wording (birr, shop, ID) is different from the UK-style spam the model learned from. |
| "Hey, it's Sarah from the gym! Your trial is ready, just reply YES and we will add 50 credit to your account." | not spam | spam | It starts like a personal text, and a few spam words cannot outweigh the many normal words. |

The other 7 attacks did **not** fool it, including the slide's negation example ("Not free, just confirming lunch at 1pm"), the spaced-out "F R E E" message, and all the normal messages that contain spam-like words.

### Bonus: a stronger model

Word pairs, character n-grams, a spelling fix (`deleet_text`) and Logistic Regression, trained on the same split:

| Model | Accuracy | Spam precision | Spam recall | False alarms | Missed spam |
|---|---|---|---|---|---|
| Baseline (TF-IDF + Naive Bayes) | 95.24% | 1.000 | 0.623 | 0 | 49 |
| Improved (words + chars + Logistic Regression) | 99.03% | 0.976 | 0.946 | 3 | 7 |

The improved model fixed both spelling tricks and was fooled by 3 of the 13 attacks (the baseline: 6). The price was 3 false alarms, where the baseline had none.

## What I learned

Accuracy alone hides weaknesses: the baseline scored 95.24% but missed 49 spam messages, and all 6 attacks that fooled it were spam that looked normal.
Because it only counts words it saw in training, disguised spellings, polite business wording and newer scam styles slip through, while some tricks I expected to work (like "not free") did not fool it.

## How to run

### Option A: Google Colab (easiest)

1. Click the **Open in Colab** badge above.
2. Choose **Runtime > Run all**. The notebook installs what it needs and downloads the dataset by itself.

### Option B: On your computer (with a virtual environment)

```bash
git clone https://github.com/bezawitlegesse/AWS_Ai_ml-week-03.git
cd AWS_Ai_ml-week-03

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook AWS_Ai_ml_task_week_3_spam_filter.ipynb
```

Then choose **Run > Run All Cells**. The first run needs internet to download the dataset.
The virtual environment (`.venv/`) is listed in `.gitignore` and is never committed.

## Project structure

```
AWS_Ai_ml-week-03/
├── AWS_Ai_ml_task_week_3_spam_filter.ipynb   # code, outputs and explanations
├── README.md           # this write-up
├── requirements.txt    # pandas, numpy, scikit-learn, matplotlib, jupyter
└── .gitignore          # keeps .venv, __pycache__ and .ipynb_checkpoints out of Git
```

## Limits

The 13 attack messages are a small hand-made sample, not a proper benchmark. The dataset is made of older English text messages,
so it says little about modern spam or about messages in other languages.
