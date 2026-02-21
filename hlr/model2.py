import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

# ==========================================
# 1. FEATURE ENGINEERING & DATA PREP
# ==========================================
def engineer_features(df):
    print("Engineering advanced features...")
    # 1. Morphological Features
    df['pos_tag'] = df['lexeme_string'].str.extract(r'<([^>]+)>').fillna('unknown')
    
    # 2. Language Pair
    df['lang_pair'] = df['ui_language'] + "->" + df['learning_language']
    
    # 3. Time Non-Linearity
    df['t_days'] = df['delta'] / (60 * 60 * 24)
    df['log_delta'] = np.log1p(df['t_days'])
    
    # 4. Historical Accuracy (Word-specific)
    df['historical_accuracy'] = np.where(
        df['history_seen'] > 0, 
        df['history_correct'] / df['history_seen'], 
        0.0
    )
    
    # 5. Global User Accuracy
    user_acc = df.groupby('user_id').apply(
        lambda x: x['history_correct'].sum() / (x['history_seen'].sum() + 1e-5)
    ).reset_index(name='user_global_accuracy')
    df = df.merge(user_acc, on='user_id', how='left')
    
    # 6. Transform original counts
    df['right_transformed'] = np.sqrt(1 + df['history_correct'])
    df['wrong_transformed'] = np.sqrt(1 + (df['history_seen'] - df['history_correct']))

    # 7. Convert Categories to Integers for Embeddings
    word_codes, uniques_words = pd.factorize(df['lexeme_string'])
    df['word_id'] = word_codes
    df['user_idx'] = pd.factorize(df['user_id'])[0]
    df['pos_id'] = pd.factorize(df['pos_tag'])[0]
    df['lang_id'] = pd.factorize(df['lang_pair'])[0]
    
    # 8. Targets
    df['p_recall'] = np.clip(df['p_recall'], 0.0001, 0.9999)
    df['true_h'] = np.clip(-df['t_days'] / np.log2(df['p_recall']), 0.0104, 274.0)

    vocab_sizes = {
        'words': df['word_id'].nunique(),
        'users': df['user_idx'].nunique(),
        'pos': df['pos_id'].nunique(),
        'langs': df['lang_id'].nunique()
    }
    
    return df, vocab_sizes, uniques_words

# ==========================================
# 2. PYTORCH DATASET
# ==========================================
class RichFlashcardDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        cat_features = {
            'word_id': torch.tensor(row['word_id'], dtype=torch.long),
            'user_idx': torch.tensor(row['user_idx'], dtype=torch.long),
            'pos_id': torch.tensor(row['pos_id'], dtype=torch.long),
            'lang_id': torch.tensor(row['lang_id'], dtype=torch.long)
        }
        
        num_features = torch.tensor([
            row['right_transformed'],
            row['wrong_transformed'],
            row['historical_accuracy'],
            row['user_global_accuracy'],
            row['log_delta']
        ], dtype=torch.float32)
        
        delta_t = torch.tensor(row['t_days'], dtype=torch.float32)
        true_p = torch.tensor(row['p_recall'], dtype=torch.float32)
        true_h = torch.tensor(row['true_h'], dtype=torch.float32)
        
        return cat_features, num_features, delta_t, true_p, true_h

# ==========================================
# 3. DEEP NEURAL NETWORK MODEL
# ==========================================
class DeepHLR(nn.Module):
    def __init__(self, vocab_sizes, num_numerical=5):
        super(DeepHLR, self).__init__()
        
        self.word_emb = nn.Embedding(vocab_sizes['words'], 32)
        self.user_emb = nn.Embedding(vocab_sizes['users'], 32)
        self.pos_emb  = nn.Embedding(vocab_sizes['pos'], 8)
        self.lang_emb = nn.Embedding(vocab_sizes['langs'], 8)
        
        total_input_size = 32 + 32 + 8 + 8 + num_numerical
        
        self.network = nn.Sequential(
            nn.Linear(total_input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        self.base = 2.0

    def forward(self, cat_features, num_features, delta_t):
        w_vec = self.word_emb(cat_features['word_id'])
        u_vec = self.user_emb(cat_features['user_idx'])
        p_vec = self.pos_emb(cat_features['pos_id'])
        l_vec = self.lang_emb(cat_features['lang_id'])
        
        x = torch.cat((w_vec, u_vec, p_vec, l_vec, num_features), dim=1)
        dp = torch.clamp(dp, min=-6.58, max=8.1)
        
        h = torch.clamp(self.base ** dp, min=0.0104, max=274.0) 
        p = torch.clamp(self.base ** (-delta_t / h), min=0.0001, max=0.9999)
        return p, h

# ==========================================
# 4. TRAINING LOOP
# ==========================================
def calculate_loss(predicted_p, predicted_h, true_p, true_h, hlwt=0.01):
    loss_fn = nn.MSELoss()
    return loss_fn(predicted_p, true_p) + (hlwt * loss_fn(predicted_h, true_h))

def train_pipeline(csv_path, epochs=5, batch_size=1024):
    print(f"Loading data from {csv_path}...")
    raw_df = pd.read_csv(csv_path)
    
    # Process Data
    df, vocab_sizes, uniques_words = engineer_features(raw_df)
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    
    train_loader = DataLoader(RichFlashcardDataset(train_df), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(RichFlashcardDataset(test_df), batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = DeepHLR(vocab_sizes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.01)
    
    print("Starting Training Loop...")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        
        # 🟢 PROGRESS BAR ADDED HERE: Wrap train_loader with tqdm()
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for batch in train_iterator:
            cat_features, num_features, delta_t, true_p, true_h = batch
            
            cat_features = {k: v.to(device) for k, v in cat_features.items()}
            num_features, delta_t, true_p, true_h = num_features.to(device), delta_t.to(device), true_p.to(device), true_h.to(device)
            
            optimizer.zero_grad()
            pred_p, pred_h = model(cat_features, num_features, delta_t)
            loss = calculate_loss(pred_p, pred_h, true_p, true_h)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # 🟢 Update the progress bar with the current loss
            train_iterator.set_postfix({'loss': f"{loss.item():.4f}"})
            
        # Quick validation
        model.eval()
        total_test_loss = 0.0
        
        # 🟢 PROGRESS BAR ADDED HERE for testing
        test_iterator = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test ]")
        
        with torch.no_grad():
            for batch in test_iterator:
                cat_features, num_features, delta_t, true_p, true_h = batch
                cat_features = {k: v.to(device) for k, v in cat_features.items()}
                num_features, delta_t, true_p, true_h = num_features.to(device), delta_t.to(device), true_p.to(device), true_h.to(device)
                
                pred_p, pred_h = model(cat_features, num_features, delta_t)
                loss = calculate_loss(pred_p, pred_h, true_p, true_h)
                total_test_loss += loss.item()
                
        print(f"Epoch {epoch+1} Summary | Train Loss: {total_train_loss/len(train_loader):.4f} | Test Loss: {total_test_loss/len(test_loader):.4f}\n")
        
    return model, uniques_words

def visualize_embeddings(model, uniques_words, num_words_to_plot=500):
    print("\nExtracting vectors and running t-SNE for visualization...")
    model.eval()
    with torch.no_grad():
        word_vectors = model.word_emb.weight.data.cpu().numpy()
        
    num_words_to_plot = min(num_words_to_plot, len(word_vectors))
    vectors_to_plot = word_vectors[:num_words_to_plot]
    words_to_plot = uniques_words[:num_words_to_plot]
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    coords = tsne.fit_transform(vectors_to_plot)
    
    plot_df = pd.DataFrame({'Word': words_to_plot, 'X': coords[:, 0], 'Y': coords[:, 1]})
    plot_df['Language'] = plot_df['Word'].astype(str).apply(lambda x: x.split(':')[0] if ':' in x else 'unknown')
    
    plt.figure(figsize=(14, 9))
    sns.set_theme(style="whitegrid")
    sns.scatterplot(data=plot_df, x='X', y='Y', hue='Language', palette='tab10', alpha=0.7, s=80)
    
    sample = plot_df.sample(n=min(100, len(plot_df)), random_state=42)
    for _, row in sample.iterrows():
        word_label = row['Word'].split(':')[-1] if isinstance(row['Word'], str) else str(row['Word'])
        plt.text(row['X'] + 0.3, row['Y'] + 0.3, word_label, fontsize=9, alpha=0.8)
        
    plt.title("Neural Network's Internal Map of Language Memory", fontsize=16, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# ==========================================
# EXECUTION & SAVING THE MODEL
# ==========================================
if __name__ == "__main__":
    CSV_PATH = "data/SpacedRepetitionData.csv" 
    MODEL_SAVE_PATH = "deep_hlr_model.pt" # Where the model will be saved
    
    try:
        # 1. Train the model (Now with progress bars!)
        trained_model, vocabulary = train_pipeline(CSV_PATH, epochs=5)
        
        # 2. 🟢 SAVE THE MODEL
        print(f"Saving model to {MODEL_SAVE_PATH}...")
        torch.save(trained_model.state_dict(), MODEL_SAVE_PATH)
        print("Model saved successfully!")
        
        # 3. Plot the brain
        visualize_embeddings(trained_model, vocabulary, num_words_to_plot=600)
        
    except FileNotFoundError:
        print(f"Error: Could not find '{CSV_PATH}'. Please make sure the path is correct!")