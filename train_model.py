import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout, SpatialDropout1D, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import pickle
import json
import csv
import os

# --- Hardware Configuration for Apple M4 ---
def configure_hardware():
    print("TensorFlow Version:", tf.__version__)
    
    # Check for GPU availability (Metal Performance Shaders)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ Found {len(gpus)} GPU(s). Using Apple Metal (MPS) acceleration.")
        except RuntimeError as e:
            print(e)
    else:
        print("⚠️ No GPU found. Ensure tensorflow-metal is installed. Running on CPU.")

configure_hardware()

# --- Configuration ---
MAX_LEN = 300  # Increased to capture longer payloads
EMBEDDING_DIM = 128
GRU_UNITS = 128
BATCH_SIZE = 128  # Efficient for M-series chips
EPOCHS = 50  # More epochs; early stopping will prevent overfitting
VOCAB_SIZE = 1500  # Character-level vocabulary is small, but we leave room
L2_REG = 1e-5  # Light L2 regularization — don't cripple learning
MAX_URL_SAMPLES = 100000  # Cap URL-heavy datasets to avoid drowning out attack payloads

# --- Data Loading & Preprocessing ---
def load_and_preprocess_data():
    print("\n--- Loading Datasets ---")
    
    dfs = []

    # 1. XSS Dataset
    if os.path.exists('XSS_dataset.csv'):
        print("Loading XSS_dataset.csv...")
        try:
            df_xss = pd.read_csv('XSS_dataset.csv', encoding='utf-8', on_bad_lines='skip')
        except UnicodeDecodeError:
            df_xss = pd.read_csv('XSS_dataset.csv', encoding='latin-1', on_bad_lines='skip')
            
        df_xss.columns = df_xss.columns.str.strip()
        df_xss = df_xss.rename(columns={'Sentence': 'text', 'Label': 'label'})
        if 'text' in df_xss.columns and 'label' in df_xss.columns:
            df_xss = df_xss[['text', 'label']]
            dfs.append(df_xss)
        else:
            print(f"⚠️ Skipping XSS_dataset.csv: Missing columns. Found: {df_xss.columns}")

    # 2. SQL Injection Dataset
    if os.path.exists('SQL_Injection_Dataset.csv'):
        print("Loading SQL_Injection_Dataset.csv...")
        try:
            df_sql = pd.read_csv('SQL_Injection_Dataset.csv', encoding='utf-8', on_bad_lines='skip')
        except UnicodeDecodeError:
            df_sql = pd.read_csv('SQL_Injection_Dataset.csv', encoding='latin-1', on_bad_lines='skip')
            
        df_sql.columns = df_sql.columns.str.strip()
        df_sql = df_sql.rename(columns={'Query': 'text', 'Label': 'label'})
        if 'text' in df_sql.columns and 'label' in df_sql.columns:
            df_sql = df_sql[['text', 'label']]
            dfs.append(df_sql)
        else:
             print(f"⚠️ Skipping SQL_Injection_Dataset.csv: Missing columns. Found: {df_sql.columns}")

    # 3. Master Web Attack Dataset (capped to avoid drowning out attack payloads)
    if os.path.exists('master_web_attack_dataset.csv'):
        print("Loading master_web_attack_dataset.csv...")
        try:
            df_master = pd.read_csv('master_web_attack_dataset.csv', encoding='utf-8', on_bad_lines='skip')
        except UnicodeDecodeError:
             df_master = pd.read_csv('master_web_attack_dataset.csv', encoding='latin-1', on_bad_lines='skip')
             
        df_master.columns = df_master.columns.str.strip()
        df_master = df_master.rename(columns={'payload': 'text'})
        # Ensure label matches format if needed, assuming it's already 'label' and 0/1
        if 'text' in df_master.columns and 'label' in df_master.columns:
            df_master = df_master[['text', 'label']]
            # Cap to prevent this dataset from dominating training
            if len(df_master) > MAX_URL_SAMPLES:
                df_master = df_master.groupby('label', group_keys=False).apply(
                    lambda x: x.sample(n=min(len(x), MAX_URL_SAMPLES // 2), random_state=42)
                )
                print(f"  Capped master_web_attack to {len(df_master)} rows (balanced sample)")
            dfs.append(df_master)
        else:
             print(f"⚠️ Skipping master_web_attack_dataset.csv: Missing columns. Found: {df_master.columns}")

    # 4. CSIC 2010
    if os.path.exists('csic_2010.csv'):
        print("Loading csic_2010.csv...")
        try:
            # CSIC dataset often has encoding issues or irregular structure
            df_csic = pd.read_csv('csic_2010.csv', encoding='latin-1', on_bad_lines='skip')
        except Exception as e:
            print(f"⚠️ Error reading csic_2010.csv: {e}")
            df_csic = pd.DataFrame()

        if not df_csic.empty:
            df_csic.columns = df_csic.columns.str.strip()
            df_csic = df_csic.rename(columns={'URL': 'text', 'classification': 'label'})
            
            if 'text' in df_csic.columns and 'label' in df_csic.columns:
                # Map string labels to int: anomalous/malicious -> 1, normal/benign -> 0
                label_map = {'anomalous': 1, 'malicious': 1, 'normal': 0, 'benign': 0}
                df_csic['label'] = df_csic['label'].astype(str).str.strip().str.lower().map(label_map)
                df_csic.dropna(subset=['label'], inplace=True)
                df_csic['label'] = df_csic['label'].astype(int)
                df_csic = df_csic[['text', 'label']]
                dfs.append(df_csic)
            else:
                 print(f"⚠️ Skipping csic_2010.csv: Missing columns. Found: {df_csic.columns}")

    # 5. Malicious URLs (capped to avoid drowning out attack payloads)
    if os.path.exists('malicious_urls.csv'):
        print("Loading malicious_urls.csv...")
        try:
            df_urls = pd.read_csv('malicious_urls.csv', encoding='utf-8', on_bad_lines='skip')
        except UnicodeDecodeError:
            df_urls = pd.read_csv('malicious_urls.csv', encoding='latin-1', on_bad_lines='skip')

        df_urls.columns = df_urls.columns.str.strip()
        df_urls = df_urls.rename(columns={'url': 'text', 'type': 'label'})
        
        if 'text' in df_urls.columns and 'label' in df_urls.columns:
            # Mapping: 'benign' -> 0, everything else -> 1
            df_urls['label'] = df_urls['label'].apply(lambda x: 0 if str(x).lower() == 'benign' else 1)
            df_urls = df_urls[['text', 'label']]
            # Cap to prevent this dataset from dominating training
            if len(df_urls) > MAX_URL_SAMPLES:
                df_urls = df_urls.groupby('label', group_keys=False).apply(
                    lambda x: x.sample(n=min(len(x), MAX_URL_SAMPLES // 2), random_state=42)
                )
                print(f"  Capped malicious_urls to {len(df_urls)} rows (balanced sample)")
            dfs.append(df_urls)
        else:
             print(f"⚠️ Skipping malicious_urls.csv: Missing columns. Found: {df_urls.columns}")

    # 6. Augmented / supplemental data (SSTI, path traversal, benign URLs, etc.)
    if os.path.exists('augmented_data.csv'):
        print("Loading augmented_data.csv...")
        df_aug = pd.read_csv('augmented_data.csv', encoding='utf-8')
        df_aug.columns = df_aug.columns.str.strip()
        if 'text' in df_aug.columns and 'label' in df_aug.columns:
            df_aug = df_aug[['text', 'label']]
            dfs.append(df_aug)
        else:
            print(f"⚠️ Skipping augmented_data.csv: Missing columns. Found: {df_aug.columns}")

    # Merge all
    if not dfs:
        raise ValueError("No datasets loaded!")
        
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Cleaning
    print(f"Total rows before cleaning: {len(full_df)}")
    full_df.dropna(subset=['text', 'label'], inplace=True)
    
    # Ensure labels are int32 (object dtype causes extremely slow argsort in train_test_split)
    full_df['label'] = pd.to_numeric(full_df['label'], errors='coerce')
    full_df.dropna(subset=['label'], inplace=True)
    full_df['label'] = full_df['label'].astype(np.int32)
    
    # Remove empty / very short texts (< 3 chars are noise, not real payloads)
    full_df['text'] = full_df['text'].astype(str).str.strip()
    full_df = full_df[full_df['text'].str.len() >= 3]
    
    # Remove duplicates to prevent data leakage/bias
    full_df.drop_duplicates(subset=['text'], inplace=True)
    print(f"Total rows after cleaning: {len(full_df)}")
    
    # Check balance
    print("Class Distribution:")
    print(full_df['label'].value_counts())
    
    return full_df

# --- Tokenization (Character Level) ---
def tokenize_data(texts):
    print("\n--- Tokenizing ---")
    # char_level=True is key here
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, char_level=True, lower=True, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
    
    print(f"Vocabulary Size: {len(tokenizer.word_index)}")
    
    # Save tokenizer
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Tokenizer saved to tokenizer.pickle")
    
    return padded_sequences, tokenizer

# --- Model Definition ---
def build_gru_model(vocab_size, max_len):
    print("\n--- Building GRU Model ---")
    model = Sequential()
    
    # Embedding Layer
    model.add(Embedding(input_dim=vocab_size + 1, output_dim=EMBEDDING_DIM))
    
    # Light spatial dropout — don't kill signal
    model.add(SpatialDropout1D(0.15))
    
    # Bidirectional GRU: captures patterns in both directions
    model.add(Bidirectional(GRU(
        GRU_UNITS,
        return_sequences=True,
        dropout=0.2,
        recurrent_dropout=0.0,
        kernel_regularizer=l2(L2_REG),
    )))
    
    # Second GRU layer — captures higher-level sequential patterns
    model.add(GRU(
        GRU_UNITS,
        return_sequences=False,
        dropout=0.2,
        recurrent_dropout=0.0,
        kernel_regularizer=l2(L2_REG),
    ))
    
    # Dense Layers
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(L2_REG)))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(L2_REG)))
    model.add(Dropout(0.2))
    
    # Output Layer
    model.add(Dense(1, activation='sigmoid'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Data
    df = load_and_preprocess_data()
    
    # 2. Split Data
    X = df['text'].astype(str).values
    y = df['label'].values.astype(np.int32)  # explicit int32 for fast stratified split
    
    # Stratified split: 60% train / 20% val / 20% test
    # First split: 80% trainval / 20% test (held-out, never seen during training)
    X_trainval_text, X_test_text, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Second split: 75% of trainval = 60% overall for train, 25% of trainval = 20% overall for val
    X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_trainval_text, y_trainval, test_size=0.25, random_state=42, stratify=y_trainval
    )
    print(f"\nSplit sizes — Train: {len(X_train_text)}, Val: {len(X_val_text)}, Test: {len(X_test_text)}")
    
    # Compute class weights to handle imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"Class weights: {class_weight_dict}")
    
    # 3. Tokenize
    X_train_seq, tokenizer = tokenize_data(X_train_text)
    X_val_seq = pad_sequences(tokenizer.texts_to_sequences(X_val_text), maxlen=MAX_LEN, padding='post', truncating='post')
    X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test_text), maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Update Vocab Size based on actual data
    actual_vocab_size = len(tokenizer.word_index)
    
    # 4. Build Model
    model = build_gru_model(actual_vocab_size, MAX_LEN)
    
    # 5. Train — validate on val set, NOT the test set
    print("\n--- Training Model ---")
    early_stop = EarlyStopping(
        monitor='val_loss', patience=7, restore_best_weights=True, verbose=1
    )
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )
    checkpoint = ModelCheckpoint(
        'gru_model_best.keras', monitor='val_loss', save_best_only=True, verbose=1
    )
    csv_logger = CSVLogger('epoch_metrics.csv', separator=',', append=False)
    
    history = model.fit(
        X_train_seq, y_train,
        validation_data=(X_val_seq, y_val),  # Validate on val set, not test set
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,  # Handle class imbalance
        callbacks=[early_stop, lr_scheduler, checkpoint, csv_logger],
        shuffle=True,
        verbose=1
    )
    
    # Save training history for later analysis
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open('training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=2)
    print("Training history saved to training_history.json")
    print("Per-epoch metrics saved to epoch_metrics.csv")
    
    # 6. Evaluate on held-out TEST set
    print("\n--- Evaluating Model on Held-Out Test Set ---")
    y_pred_prob = model.predict(X_test_seq)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred)
    print(report)
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Save test predictions for metrics script
    np.savez('test_results.npz', y_test=y_test, y_pred=y_pred, y_pred_prob=y_pred_prob.flatten())
    print("Test results saved to test_results.npz")
    
    # 7. Save final model
    print("\n--- Saving Model ---")
    model.save('gru_model.keras')
    print("Model saved to gru_model.keras")
    
    print("\n✅ Pipeline complete.")
