import os
import json
import numpy as np
from backend.fashion_clip import FashionCLIP
import pandas as pd

# Initialize the model
fclip = FashionCLIP('patrickjohncyh/fashion-clip')

# Style prompts
style_prompts = {
    'Minimalist': ['simple modern outfit', 'minimalist fashion'],
    'Sporty': ['athletic wear', 'sportswear with sneakers'],
    'Retro': ['vintage clothing', 'old-fashioned fashion'],
    'Elegant': ['elegant dress', 'classic formal outfit'],
    'Streetwear': ['urban street fashion', 'hoodie and sneakers'],
    'Formal': ['business suit', 'formal wear for work']
}
style_names = list(style_prompts.keys())
style_texts = [desc for lst in style_prompts.values() for desc in lst]

# Style embeddings
style_embeddings = fclip.encode_text(style_texts, batch_size=8)
style_embeddings = style_embeddings / np.linalg.norm(style_embeddings, axis=1, keepdims=True)


# Load Kaggle metadata
STYLE_CSV = 'fashion_dataset/styles.csv'
IMAGE_DIR = 'fashion_dataset/images'
DB_PATH = 'data/fashion_db.json'

def load_style_dataframe():
    df = pd.read_csv(STYLE_CSV, on_bad_lines='skip')
    df = df.dropna(subset=['articleType'])
    df['id'] = df['id'].astype(str)
    return df

df = load_style_dataframe()
LABELS = sorted(df['articleType'].str.lower().value_counts().head(20).index.tolist())



def build_database(limit=5):
    df = load_style_dataframe()

    database = []
    for i, (_, row) in enumerate(df.iterrows()):
        if i >= limit:
            break
        
        image_path = os.path.join(IMAGE_DIR, row['id'] + '.jpg')
        if not os.path.exists(image_path):
            continue

        try:
            img_emb = fclip.encode_images([image_path], batch_size=1)[0]
            img_emb = img_emb / np.linalg.norm(img_emb)
        except:
            continue

        sims = np.dot(style_embeddings, img_emb)
        grouped = sims.reshape(len(style_names), -1).mean(axis=1)
        best_style = style_names[np.argmax(grouped)]

        database.append({
            "path": image_path,
            "embedding": img_emb.tolist(),
            "style": best_style,
            "item_type": row['articleType'].lower()
        })

    os.makedirs("data", exist_ok=True)
    with open(DB_PATH, 'w') as f:
        json.dump(database, f, indent=2)

        
def classify_item_type(image_path):
    """Use zero-shot to classify input image to one of the common articleTypes."""
    result = fclip.zero_shot_classification([image_path], LABELS)
    return result[0] if result else 'unknown'

def recommend(image_path, top_k=5):
    with open(DB_PATH, 'r') as f:
        database = json.load(f)

    img_emb = fclip.encode_images([image_path], batch_size=1)[0]
    img_emb = img_emb / np.linalg.norm(img_emb)

    sims = np.dot(style_embeddings, img_emb)
    grouped = sims.reshape(len(style_names), -1).mean(axis=1)
    predicted_style = style_names[np.argmax(grouped)]

    item_type = classify_item_type(image_path)
    same_style_items = [d for d in database if d['style'] == predicted_style and d['item_type'] == item_type]

    if not same_style_items:
        return [], predicted_style, item_type

    embs = np.array([d['embedding'] for d in same_style_items])
    sim_scores = np.dot(embs, img_emb)
    top_idx = sim_scores.argsort()[-top_k:][::-1]
    return [same_style_items[i]['path'] for i in top_idx], predicted_style, item_type

