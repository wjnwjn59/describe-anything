import argparse
import json
import re

def normalize_vqa_answer(ans):
    # Based on processPunctuation and processDigitArticle from eval_mm
    punct = [';', r'/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']
    period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
    comma_strip = re.compile(r'(\d)(,)(\d)')
    articles = ['a', 'an', 'the']
    manual_map = {
        'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10',
    }
    contractions = {
        "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
        "didnt": "didn't", "doesnt": "doesn't", "dont": "don't", "hadnt": "hadn't", "hasnt": "hasn't",
        "havent": "haven't", "hed": "he'd", "hes": "he's", "im": "i'm", "ive": "i've", "isnt": "isn't",
        "itd": "it'd", "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
        "mightve": "might've", "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't", "notve": "not've",
        "oclock": "o'clock", "oughtnt": "oughtn't", "shant": "shan't", "shed": "she'd", "she's": "she's",
        "shouldve": "should've", "shouldnt": "shouldn't", "somebodyd": "somebody'd", "somebodyll": "somebody'll",
        "somebodys": "somebody's", "someoned": "someone'd", "someonell": "someone'll", "someones": "someone's",
        "somethingd": "something'd", "somethingll": "something'll", "thats": "that's", "thered": "there'd",
        "therere": "there're", "theres": "there's", "theyd": "they'd", "theyll": "they'll", "theyre": "they're",
        "theyve": "they've", "twas": "'twas", "wasnt": "wasn't", "wed": "we'd", "weve": "we've", "werent": "weren't",
        "whatll": "what'll", "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's",
        "whered": "where'd", "wheres": "where's", "whereve": "where've", "whod": "who'd", "wholl": "who'll",
        "whos": "who's", "whove": "who've", "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't",
        "yall": "y'all", "youd": "you'd", "youll": "you'll", "youre": "you're", "youve": "you've",
    }

    # Remove punctuation
    out_text = ans
    for p in punct:
        if (p + ' ' in out_text or ' ' + p in out_text) or (re.search(comma_strip, out_text) is not None):
            out_text = out_text.replace(p, '')
        else:
            out_text = out_text.replace(p, ' ')
    out_text = period_strip.sub('', out_text)

    # Lowercase, remove articles, map numbers, expand contractions
    words = out_text.lower().split()
    words = [manual_map.get(w, w) for w in words if w not in articles]
    words = [contractions.get(w, w) for w in words]
    return ' '.join(words).strip()

def vqa_score(pred, gts):
    # Normalize prediction and ground truths
    pred_norm = normalize_vqa_answer(pred)
    gts_norm = [normalize_vqa_answer(gt) for gt in gts]

    # According to VQA, an answer matching >=3/10 of GTs counts as 1 point, capped at 1.0
    matches = [pred_norm == gt for gt in gts_norm]
    # acc = min(1.0, sum(matches) / 3.0) if gts_norm else 0.0 
    acc = sum(matches) / len(matches)
    return acc

def normalize_infovqa_answer(ans):
    # Lowercase, strip whitespace, remove punctuation
    ans = ans.lower().strip()
    ans = re.sub(r'\s+', ' ', ans)
    ans = re.sub(r'[\.,;:!?"\'\[\](){}]', '', ans)
    return ans

def anls_score(pred, gts, threshold=0.5):
    # Normalize
    pred_norm = normalize_infovqa_answer(pred)
    gts_norm = [normalize_infovqa_answer(gt) for gt in gts]

    def levenshtein(a, b):
        if a == b:
            return 0
        if len(a) == 0:
            return len(b)
        if len(b) == 0:
            return len(a)
        prev = list(range(len(b) + 1))
        curr = [0] * (len(b) + 1)
        for i, ca in enumerate(a):
            curr[0] = i + 1
            for j, cb in enumerate(b):
                cost = 0 if ca == cb else 1
                curr[j + 1] = min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost)
            prev, curr = curr, prev
        return prev[len(b)]

    # Compute ANLS = 1 - (levenshtein / max_len); count as correct if >= threshold
    scores = []
    for gt in gts_norm:
        dist = levenshtein(pred_norm, gt)
        length = max(len(pred_norm), len(gt))
        score = 1.0 - dist / length if length > 0 else 0.0
        scores.append(score)
    best = max(scores) if scores else 0.0
    return best if best >= threshold else 0.0

def main():
    parser = argparse.ArgumentParser(description='Evaluate VQA or InfographicVQA predictions.')
    parser.add_argument(
        '--result', type=str, required=True,
        help='Path to the JSONL file containing your inference results.'
    )
    parser.add_argument(
        '--dataset', type=str, required=True, choices=['vqav2_val', 'infographicvqa_val'],
        help='Dataset to evaluate.'
    )
    parser.add_argument(
        '--anls-threshold', type=float, default=0.5, help='ANLS threshold for InfographicVQA.'
    )
    parser.add_argument(
        '--show-detail', action='store_true',
        help='If set, print detailed per-sample predictions, ground truths, and scores.'
    )
    args = parser.parse_args()

    with open(args.result, encoding='utf-8') as f:
        lines = [json.loads(line) for line in f]

    scores = []
    for i, item in enumerate(lines):
        pred = item['predict']
        gts = item['gt'] if isinstance(item['gt'], list) else [item['gt']]
        if args.dataset == 'vqav2_val':
            score = vqa_score(pred, gts)
        elif args.dataset == 'infographicvqa_val':
            score = anls_score(pred, gts, threshold=args.anls_threshold)
        scores.append(score)
        if args.show_detail:
            print(f"[{i}] Prediction: {pred}\n    Ground Truths: {gts}\n    Score: {score:.3f}")

    avg_score = sum(scores) / len(scores) if scores else 0.0
    print(f"\n==> Average score ({args.dataset}): {avg_score:.4f} over {len(scores)} samples")

if __name__ == '__main__':
    main()
# python score_vqa.py --result qwenvl_vqav2_val.jsonl --dataset vqav2_val
