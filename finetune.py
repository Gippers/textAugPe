import os
import torch
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments
)
from torch.utils.data import DataLoader

# ---- CONFIG ----
MODEL_NAME   = 'BAAI/bge-base-en-v1.5'
SAVE_PATH    = './bge_finetuned_movie_reviews'
EPOCHS       = 1        # deliberately low — small dataset overfits fast
BATCH_SIZE   = 4
WARMUP_STEPS = 3

# ---- DATA ----
# WARNING: Hardcoded labels — update if finetune_data changes
# NOTE: Held-out data only — never use private_data here, breaks DP
# Sentiment: 1=positive, 0=mixed, -1=negative

finetune_data = [
    "The leading actress brought raw emotion to every scene and was impossible to look away from.",
    "The narrative jumped around so much it was impossible to follow by the second half.",
    "Every shot was composed beautifully, the director clearly had a strong visual identity.",
    "The special effects looked dated and took me completely out of the story.",
    "A gripping screenplay let down only by a lead who seemed disengaged from the material.",
    "Gorgeous colour grading throughout but the three-act structure completely fell apart.",
    "The supporting cast outshone the leads, though the dialogue gave them little to work with.",
    "Lush visuals and a twisting narrative anchored by two powerhouse central performances.",
    "Wooden delivery from the entire cast, a muddled story, and effects that looked unfinished.",
    "The performers gave it everything but the rushed pacing and flat lighting dulled the impact.",
    "A stunning debut performance that will be remembered for years to come.",
    "The script was razor sharp and kept me completely hooked from start to finish.",
    "Breathtaking underwater sequences that pushed the boundaries of visual effects.",
    "The dialogue felt forced and the character motivations made absolutely no sense.",
    "An emotionally nuanced performance from the lead that anchored an otherwise average film.",
    "Beautiful handheld cinematography gave the film an urgent, visceral energy.",
    "The story meandered for an hour before anything of consequence happened.",
    "A tour de force ensemble cast elevating a fairly standard screenplay.",
    "Cheap looking sets and murky lighting that made everything hard to watch.",
    "The director coaxed genuinely moving performances from even the smallest roles.",
    "Overly complicated plotting that collapsed under its own weight by the finale.",
    "Vivid colour palette and inventive camera angles made this a visual treat.",
    "The central performance was so wooden it undermined every emotional scene.",
    "A tightly constructed thriller with genuine surprises at every turn.",
    "Flat, washed out cinematography that drained all life from the story.",
    "The actor found extraordinary depth in what could have been a one-note role.",
    "A bloated runtime masking a paper thin plot with nothing new to say.",
    "The practical effects work here puts modern CGI blockbusters to shame.",
    "Lifeless performances across the board made it impossible to care about anyone.",
    "An intimate character study elevated by a fearless lead performance.",
]

#               0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29
cat_acting  = [ 1,  0,  0,  0,  1,  0,  1,  1,  1,  1,  1,  0,  0,  0,  1,  0,  0,  1,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0,  1,  1]
cat_plot    = [ 0,  1,  0,  0,  1,  1,  1,  1,  1,  1,  0,  1,  0,  1,  0,  0,  1,  1,  0,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0,  0]
cat_visuals = [ 0,  0,  1,  1,  0,  1,  0,  1,  1,  1,  0,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0,  1,  0,  0]
sentiment   = [ 1,  -1,  1, -1,  0,  0,  0,  1, -1,  0,  1,  1,  1, -1,  0,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1]


# ---- PAIR BUILDING ----
def category_similarity(i, j):
    return int(
        (cat_acting[i]  and cat_acting[j])  or
        (cat_plot[i]    and cat_plot[j])    or
        (cat_visuals[i] and cat_visuals[j])
    )

def sentiment_similarity(i, j):
    # 1 if same sentiment, 0 if adjacent (e.g. mixed/positive), -1 if opposite
    diff = abs(sentiment[i] - sentiment[j])
    if diff == 0:   return 1    # same
    if diff == 1:   return 0    # adjacent
    return -1                   # opposite (positive vs negative)

def combined_label(i, j):
    cat_sim  = category_similarity(i, j)
    sent_sim = sentiment_similarity(i, j)

    if cat_sim == 1 and sent_sim == 1:   return 1.0   # same category, same sentiment
    if cat_sim == 1 and sent_sim == 0:   return 0.6   # same category, adjacent sentiment
    if cat_sim == 1 and sent_sim == -1:  return 0.2   # same category, opposite sentiment
    if cat_sim == 0 and sent_sim == 1:   return 0.1   # different category, same sentiment
    return 0.0                                         # different category, different sentiment
from datasets import Dataset as HFDataset

def build_pairs(data):
    examples = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            label = combined_label(i, j)
            examples.append({
                "sentence1": data[i],
                "sentence2": data[j],
                "label":     label
            })
    print(f"Training pairs: {len(examples)}")
    label_dist = {}
    for e in examples:
        l = e["label"]
        label_dist[l] = label_dist.get(l, 0) + 1
    print(f"Label distribution: {label_dist}")
    return HFDataset.from_list(examples)


# ---- SANITY CHECK ----
def sanity_check(model):
    test_pairs = [
        ("Incredible acting, the cast was phenomenal.",    "The performances were wooden and unconvincing.",   "same cat, opposite sentiment → low"),
        ("Incredible acting, the cast was phenomenal.",    "The cast brought real depth to their roles.",      "same cat, same sentiment → high"),
        ("The cinematography was stunning.",               "The plot had too many holes.",                     "diff cat → low"),
    ]
    print("\nSanity check:")
    for a, b, desc in test_pairs:
        embs = model.encode([a, b], normalize_embeddings=True)
        sim  = float(embs[0] @ embs[1])
        print(f"  {sim:.3f} | {desc}")


# ---- MAIN ----
def main():
    torch.set_num_threads(10)

    train_dataset = build_pairs(finetune_data)

    model      = SentenceTransformer(MODEL_NAME, device='cpu')
    train_loss = losses.CosineSimilarityLoss(model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=SAVE_PATH,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        learning_rate=2e-5,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
    )

    trainer.train()
    model.save(SAVE_PATH)
    print(f"\nModel saved to {SAVE_PATH}")
    sanity_check(model)

if __name__ == "__main__":
    main()