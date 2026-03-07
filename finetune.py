import random
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os

# ---- DATA ----
CATEGORIES = ["acting", "plot", "visuals"]
# WARNING: These category labels are hardcoded manually.
# If you change the order or content of finetune_data you must update these lists.
# NOTE: This is HELD OUT data — similar domain but NOT the same records as finetune_data.
# Using finetune_data here would break the DP guarantee on the histogram.

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
    "The performers gave it everything but the rushed pacing and flat lighting dulled the impact."
]

#              0  1  2  3  4  5  6  7  8  9
cat_acting  = [1, 0, 0, 0, 1, 0, 1, 1, 1, 1]
cat_plot    = [0, 1, 0, 0, 1, 1, 1, 1, 1, 1]
cat_visuals = [0, 0, 1, 1, 0, 1, 0, 1, 1, 1]

# ---- BUILD PAIRS ----
# For ContrastiveLoss: label=1 (similar), label=0 (dissimilar)
# Similar = share at least one category, dissimilar = no shared category
def share_category(i, j):
    return int(
        (cat_acting[i]  and cat_acting[j])  or
        (cat_plot[i]    and cat_plot[j])    or
        (cat_visuals[i] and cat_visuals[j])
    )

train_examples = []
for i in range(len(finetune_data)):
    for j in range(i + 1, len(finetune_data)):
        train_examples.append(InputExample(
            texts=[finetune_data[i], finetune_data[j]],
            label=float(share_category(i, j))
        ))

print(f"Training pairs: {len(train_examples)}")

# ---- MODEL ----
# Use base not large for CPU — 3x faster, still good quality
model = SentenceTransformer('BAAI/bge-base-en-v1.5')

# ---- DATALOADER + LOSS ----
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
train_loss = losses.ContrastiveLoss(model)

# ---- TRAIN ----
# 1 epoch over 45 pairs on CPU ~15-30 mins for base model
# Increase num_epochs to 3-5 if you want more, still within 2-3hrs
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=5,
    show_progress_bar=True
)

# ---- SAVE ----
save_path = "./bge_finetuned_movie_reviews"
model.save(save_path)
print(f"Model saved to {save_path}")

# ---- QUICK SANITY CHECK ----
test_pairs = [
    ("Great visuals and stunning cinematography.", "The CGI was cheap and unconvincing."),  # same category, different sentiment
    ("The acting was superb.", "The plot had too many holes."),                              # different categories
]
for a, b in test_pairs:
    embs = model.encode([a, b], normalize_embeddings=True)
    sim = float(embs[0] @ embs[1])
    print(f"Sim: {sim:.3f} | {a[:40]} <-> {b[:40]}")