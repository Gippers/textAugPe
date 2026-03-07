import os
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments
)

# ---- CONFIG ----
MODEL_NAME   = 'BAAI/bge-base-en-v1.5'
SAVE_PATH    = './bge_finetuned_movie_reviews'
EPOCHS       = 10
BATCH_SIZE   = 8
WARMUP_STEPS = 20
LEARNING_RATE = 2e-5
MARGIN        = 1.0

# ---- DATA ----
# WARNING: Hardcoded category labels — update if finetune_data changes.
# NOTE: Held-out data only — never use private_data here as it breaks DP.
# Index:       0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19
#              20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39
#              40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59

CATEGORIES = ["acting", "plot", "visuals"]

finetune_data = [
    # --- acting only (0-19) ---
    "The leading actress brought raw emotion to every scene and was impossible to look away from.",
    "The supporting cast outshone the leads with nuanced, understated performances throughout.",
    "Every actor in this film seemed to be sleepwalking through their lines with zero conviction.",
    "The chemistry between the two leads felt completely authentic and carried the entire runtime.",
    "A career-defining turn from the lead that deserves serious awards consideration.",
    "The villain was played with such menace and restraint that every scene felt genuinely threatening.",
    "Wooden delivery from the entire ensemble drained the film of any emotional resonance.",
    "The child actor was a revelation, outperforming veterans twice her age in every scene.",
    "Overacting plagued the film from start to finish, with every emotion cranked to eleven.",
    "The lead managed to convey grief, rage, and tenderness in a single unbroken two-minute take.",
    "Despite a weak script the cast elevated the material through sheer force of talent.",
    "The performers gave it everything but the direction failed to channel their energy effectively.",
    "An uncommonly restrained performance that communicated more through silence than dialogue.",
    "The cast clearly had no rapport with each other and it showed in every single scene.",
    "Effortless screen presence from the lead made even the exposition scenes compelling viewing.",
    "The ensemble worked in perfect harmony, each performer giving the others space to shine.",
    "A miscast lead undermined every dramatic moment the film was building toward.",
    "The actors clearly trusted each other completely, resulting in performances of rare intimacy.",
    "Flat, uninspired line readings robbed the script of whatever wit it originally contained.",
    "The physical transformation the lead underwent for this role paid off in every frame.",

    # --- plot only (20-39) ---
    "The narrative jumped around so much it was impossible to follow by the second half.",
    "A gripping screenplay that kept me guessing right up until the final frame.",
    "The third act completely abandoned everything the first two had carefully established.",
    "Every plot twist felt earned rather than arbitrary, a rare achievement in modern cinema.",
    "The story was so predictable I had correctly guessed the ending within the first ten minutes.",
    "A tightly constructed mystery that rewarded attentive viewers with satisfying payoffs.",
    "The screenplay was bloated with subplots that went nowhere and added nothing to the story.",
    "The pacing was masterful, never rushing but never letting the tension drop for a moment.",
    "Gaping plot holes undermined what could have been a genuinely compelling thriller.",
    "The non-linear structure was handled with confidence and added genuine depth to the story.",
    "A promising premise squandered by a second act that completely lost the thread.",
    "The dialogue crackled with intelligence and every scene advanced both plot and character.",
    "The story collapsed under the weight of its own ambitions in the final thirty minutes.",
    "An admirably lean script that trusted the audience to fill in the gaps themselves.",
    "The rushed ending left too many threads dangling and felt like a different film entirely.",
    "The screenplay balanced multiple storylines with impressive clarity and control.",
    "A formulaic plot that hit every expected beat without a single moment of genuine surprise.",
    "The story built to a climax that felt both inevitable and completely unexpected.",
    "Lazy writing filled every quiet moment with unnecessary exposition and hand-holding.",
    "The narrative restraint on display here was refreshing in an era of over-explained blockbusters.",

    # --- visuals only (40-59) ---
    "Every shot was composed beautifully, the director clearly had a strong visual identity.",
    "The CGI was cheap and unconvincing, completely pulling me out of the experience.",
    "Gorgeous colour grading throughout gave the film a dreamlike quality that suited the story.",
    "The cinematography was flat and uninspired, looking more like a television production.",
    "A masterclass in practical effects work that put big-budget CGI spectacles to shame.",
    "The washed-out palette made the film visually exhausting to sit through for two hours.",
    "Every frame was composed with the care of a painting, making it a joy to simply watch.",
    "Muddy, poorly lit action sequences made it nearly impossible to follow what was happening.",
    "The production design was extraordinarily detailed, building a world that felt fully inhabited.",
    "Overly reliant on shaky cam and rapid cutting, the film was genuinely difficult to watch.",
    "Breathtaking location photography elevated what could have been a straightforward thriller.",
    "The visual effects ranged from competent to embarrassingly poor within the same sequence.",
    "Immaculate costume and set design transported the audience to another era completely.",
    "A grey, joyless visual palette that drained every scene of energy and life.",
    "The long takes and careful framing gave the film a meditative, painterly quality.",
    "Inconsistent lighting made several key scenes feel unfinished and poorly executed.",
    "Some of the most inventive camera work seen in mainstream cinema in the past decade.",
    "The overuse of lens flare and digital filters gave the film a cheap, processed look.",
    "Stunning practical set construction that no amount of CGI could have replicated.",
    "The action sequences were so visually chaotic they generated confusion rather than excitement.",

    # --- acting + plot (60-74) ---
    "The plot was gripping but the wooden acting made it hard to stay invested in the characters.",
    "A messy, incoherent story redeemed only by two genuinely outstanding central performances.",
    "Strong character work from the cast breathed life into a screenplay that was merely functional.",
    "The lead gave everything to the role but the script gave the character nowhere interesting to go.",
    "A taut, well-constructed thriller elevated further by performances of real conviction and depth.",
    "The story was compelling enough to forgive performances that were merely adequate.",
    "Two powerhouse leads locked in a battle of wills, served by a script smart enough to let them.",
    "The narrative was gripping but the supporting cast was so weak it kept pulling focus.",
    "An actor's showcase disguised as a thriller, with a plot that existed mainly to create scenes.",
    "The script gave each character a distinct voice and the cast honoured every word of it.",
    "A fascinating story undermined at every turn by a lead who seemed bored by the material.",
    "The ensemble found the human truth in a screenplay that could easily have felt mechanical.",
    "Sharp, witty dialogue delivered with perfect timing by a cast clearly relishing the material.",
    "The performances were strong enough to paper over the cracks in a somewhat shaky narrative.",
    "An implausible plot made believable purely through the commitment of the performers.",

    # --- acting + visuals (75-89) ---
    "Stunning cinematography and committed performances made this one of the year's best.",
    "The lead's physical performance was extraordinary but the flat lighting let her down badly.",
    "Beautiful to look at but the wooden acting kept any genuine emotion at a distance.",
    "The way the camera captured the lead's face during the climax was simply breathtaking.",
    "Gorgeous imagery combined with a deeply felt central performance made this unmissable.",
    "The visual inventiveness of the direction brought out the best in an already excellent cast.",
    "Striking visuals could not compensate for performances that never once felt truthful.",
    "A beautifully shot character study anchored by a lead performance of real fragility.",
    "The director found arresting images at every turn but seemed unable to get the cast to engage.",
    "The performances and the visuals worked in perfect harmony to create something truly affecting.",
    "Every frame was stunning but the acting was so mannered it kept pulling me out of the story.",
    "Raw, unpolished performances combined with naturalistic photography gave this real urgency.",
    "The stylised visuals complemented the heightened, theatrical performances beautifully.",
    "The lead's eyes told more story than the entire screenplay and the camera knew it.",
    "Gorgeous production design wasted on characters I never once believed in.",

    # --- plot + visuals (90-104) ---
    "Stunning visuals throughout but the story made absolutely no sense by the third act.",
    "A compelling narrative shot with total visual indifference and no sense of cinematic space.",
    "The screenplay was tight and intelligent but the flat, televisual direction failed it completely.",
    "Breathtaking imagery in service of a story that was equally breathtaking in its ambition.",
    "The visual grammar of the film perfectly reflected the fractured state of mind in the narrative.",
    "A strong story told without a single memorable image, which felt like a missed opportunity.",
    "The colour palette shifted subtly as the narrative darkened, a beautiful piece of storytelling.",
    "An inventive narrative structure matched by an equally inventive visual approach throughout.",
    "The story built beautifully to a climax the direction simply did not have the tools to realise.",
    "Every visual choice reinforced the themes of the screenplay in ways both obvious and subtle.",
    "A script full of big ideas shot in the blandest possible way, draining it of all its potential.",
    "The director found visual equivalents for the story's emotional beats at every single turn.",
    "Poor pacing dragged out what should have been a lean thriller into a visually lush but overlong slog.",
    "The story and the images felt made for each other, a rare example of total creative coherence.",
    "A visually inventive approach that distracted from rather than enhanced a fairly thin narrative.",

    # --- acting + plot + visuals (105-119) ---
    "A visual masterpiece with a twisting plot and career-best performances from the entire cast.",
    "Terrible acting, a nonsensical storyline, and visuals that looked like they were made for TV.",
    "The actors clearly tried their best but the weak plot and murky visuals made it a slog.",
    "Every element fired at once: the story, the performances, the images — a genuinely great film.",
    "Nothing worked: the script was a mess, the cast seemed lost, and it looked utterly joyless.",
    "A complete artistic vision realised through exceptional performances and stunning craft.",
    "The film failed on every level simultaneously, a remarkably consistent achievement in mediocrity.",
    "Masterful direction drew extraordinary performances while telling a story of genuine complexity.",
    "Poor writing, poor acting, and poor visuals combined to create a deeply unpleasant experience.",
    "The performances, the story and the images all pulled in the same direction for once.",
    "Despite strong visuals the film was sunk by a ridiculous plot and badly miscast leads.",
    "Tight storytelling, nuanced performances and gorgeous photography made this essential viewing.",
    "A visually accomplished film with a coherent story but performances too broad to convince.",
    "The visual ambition was matched by the narrative ambition and both were realised by a superb cast.",
    "Competent on every level but exceptional on none, a film that was simply less than its parts.",
]

#               0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19
cat_acting  =  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#               20 21 22 23 24 25 26 27 28 29  30 31 32 33 34 35 36 37 38 39
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#               40 41 42 43 44 45 46 47 48 49  50 51 52 53 54 55 56 57 58 59
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#               60 61 62 63 64 65 66 67 68 69  70 71 72 73 74
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1,
#               75 76 77 78 79 80 81 82 83 84  85 86 87 88 89
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1,
#               90 91 92 93 94 95 96 97 98 99  100 101 102 103 104
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,
#               105 106 107 108 109 110 111 112 113 114 115 116 117 118 119
                1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1]

cat_plot    =  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1]

cat_visuals =  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  1,  1,  1,  1,  1]


# ---- HELPERS ----
def share_category(i, j):
    return int(
        (cat_acting[i]  and cat_acting[j])  or
        (cat_plot[i]    and cat_plot[j])    or
        (cat_visuals[i] and cat_visuals[j])
    )

def build_pairs(data):
    examples = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            examples.append(InputExample(
                texts=[data[i], data[j]],
                label=float(share_category(i, j))
            ))
    n_similar   = sum(1 for e in examples if e.label == 1.0)
    n_dissimilar = sum(1 for e in examples if e.label == 0.0)
    print(f"Training pairs: {len(examples)} ({n_similar} similar, {n_dissimilar} dissimilar)")
    return examples

def to_hf_dataset(examples):
    return Dataset.from_dict({
        "sentence1": [e.texts[0] for e in examples],
        "sentence2": [e.texts[1] for e in examples],
        "label":     [e.label    for e in examples]
    })

def sanity_check(model):
    test_pairs = [
        # Same category — should be HIGH
        ("Great visuals and stunning cinematography.", "The CGI was cheap and unconvincing."),
        ("The acting was superb and emotionally convincing.", "The cast gave career best performances."),
        # Different category — should be LOW
        ("The acting was superb.", "The plot had too many holes."),
        ("Gorgeous cinematography.", "The storyline was predictable and boring."),
    ]
    print("\nSanity check (same category should score higher than cross category):")
    for a, b in test_pairs:
        embs = model.encode([a, b], normalize_embeddings=True)
        sim  = float(embs[0] @ embs[1])
        print(f"  {sim:.3f} | {a[:45]} <-> {b[:45]}")


# ---- MAIN ----
def main():
    train_examples = build_pairs(finetune_data)
    hf_dataset     = to_hf_dataset(train_examples)

    model      = SentenceTransformer(MODEL_NAME, device='cpu')
    train_loss = losses.ContrastiveLoss(model, margin=MARGIN)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=SAVE_PATH,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=hf_dataset,
        loss=train_loss,
    )

    trainer.train()
    model.save(SAVE_PATH)
    print(f"\nModel saved to {SAVE_PATH}")
    sanity_check(model)


if __name__ == "__main__":
    main()