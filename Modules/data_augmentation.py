# ---------------------------------------------------------
# 🧠 Dynamic Medical Specialty Text Augmentation (Show Balance)
# ---------------------------------------------------------
import random
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# ==============================
# 1️⃣ Load your dataset
# ==============================
df = pd.read_csv("/kaggle/input/clinical-dataset/MTS_samples.csv")  # Replace with your actual dataset path
col_specialty = "medical_specialty"

# ==============================
# 2️⃣ Show original class balance
# ==============================
print("\n--- Original Class Balance ---")
original_counts = df[col_specialty].value_counts()
print(original_counts.sort_index())
print("\n" + "="*50 + "\n")

# ==============================
# 3️⃣ Identify minority classes (< target_count)
# ==============================
target_count = 100
minority_classes = original_counts[original_counts < target_count].index.tolist()
print(f"🩺 Found {len(minority_classes)} specialties with < {target_count} samples.\n")

# ==============================
# 4️⃣ Learn from existing data and augment
# ==============================
augmented_rows = []

for specialty in minority_classes:
    subset = df[df[col_specialty] == specialty]

    if len(subset) == 0:
        print(f"⚠️ Skipping '{specialty}' (no data found).")
        continue

    add_n = target_count - len(subset)
    print(f"🔄 Augmenting {specialty}: {len(subset)} → {target_count} (adding {add_n})")

    # Learn frequent terms per column
    text_columns = [c for c in df.columns if c != col_specialty]
    word_bank = {}

    for col in text_columns:
        texts = subset[col].dropna().astype(str).tolist()
        if len(texts) == 0:
            word_bank[col] = ["N/A"]
            continue
        vec = CountVectorizer(stop_words="english", max_features=50)
        X = vec.fit_transform(texts)
        word_bank[col] = list(vec.get_feature_names_out())

    # Generate synthetic samples
    for _ in range(add_n):
        new_row = {col_specialty: specialty}
        for col in text_columns:
            tokens = random.sample(word_bank[col], k=min(5, len(word_bank[col])))
            new_row[col] = " ".join(tokens).capitalize() + "."
        augmented_rows.append(new_row)

# ==============================
# 5️⃣ Combine datasets
# ==============================
df_augmented = pd.DataFrame(augmented_rows)
df_balanced = pd.concat([df, df_augmented], ignore_index=True)

# ==============================
# 6️⃣ Show final class balance
# ==============================
print("\n--- Final Class Balance After Augmentation ---")
final_counts = df_balanced[col_specialty].value_counts()
print(final_counts.sort_index())

# ==============================
# 7️⃣ Save to CSV
# ==============================
df_balanced.to_csv("balanced_medical_dataset.csv", index=False)
print("\n💾 Saved as 'balanced_medical_dataset.csv'.")
