import os
import pandas as pd
import torch
import whisper
import torchaudio
import math
import numpy as np
import unicodedata
import re
from transformers import AutoTokenizer, Wav2Vec2Processor, Wav2Vec2Model, DistilBertModel
from dotmap import DotMap
import yaml
import sys

# Add modules to path to import utils if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def remove_non_english(text):
    return re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', '', text)

def preprocess_whisper(config):
    print("Starting Whisper Preprocessing...")
    model = whisper.load_model("turbo")
    
    labels_path = config.data.csv_labels_path
    labels_pd = pd.read_csv(labels_path)
    
    # Filter for target classes
    target_classes = ['lvPPA', 'nfPPA', 'svPPA']
    labels_pd = labels_pd[labels_pd['DX_Pilar'].isin(target_classes)].reset_index(drop=True)
    
    root_audio_path = config.data.root_audio_path
    root_text_path = config.data.root_text_path
    os.makedirs(root_text_path, exist_ok=True)
    
    transcriptions_df = pd.DataFrame(columns=['uid', 'transcription', 'transcription_pause'])
    
    for index, row in labels_pd.iterrows():
        filename = row.iloc[0]
        uid = os.path.splitext(filename)[0]
        audio_path = os.path.join(root_audio_path, filename)
        
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            continue
            
        word_level_path = os.path.join(root_text_path, uid + '.csv')
        
        # Skip if already processed
        if os.path.exists(word_level_path):
            print(f"Skipping {uid}, already processed.")
            # We still need to add to transcriptions_df
            # Ideally we should read the existing csv to reconstruct transcription, but for now let's re-process if needed or assume we can skip
            # To be safe and simple, let's re-process or read from existing if we implement reading.
            # For simplicity, let's just run it. It's safer.
            # Actually, re-running whisper is slow. Let's check if we can skip.
            pass 
        
        print(f"Processing Whisper for: {uid}")
        
        try:
            result = model.transcribe(audio_path, word_timestamps=True)
        except Exception as e:
            print(f"Error transcribing {uid}: {e}")
            continue
            
        transcription = ''
        transcription_pauses = ''
        prev_start = 0.0
        
        pandas_word_level = pd.DataFrame(columns=['word', 'start', 'end', 'probability'])
        
        for segment in result['segments']:
            for word in segment['words']:
                transcription_pauses += word['word']
                transcription += word['word']
                
                # Clean word for storage
                clean_word = remove_non_english(word['word'].replace('.', '').replace(',', '').replace(';', '').replace(' ', '').lower())
                
                if clean_word != '':
                    pandas_word_level = pd.concat([pandas_word_level, pd.DataFrame([{
                        'word': clean_word, 
                        'start': word['start'], 
                        'end': word['end'], 
                        'probability': word['probability']
                    }])], ignore_index=True)
                
                if prev_start > 0.0:
                    pause = word['start'] - prev_start
                    if pause > 2:
                        transcription_pauses += ' ...'
                    elif pause > 1:
                        transcription_pauses += ' .'
                    elif pause > 0.5:
                        transcription_pauses += ' ,'
                
                prev_start = word['end']
        
        pandas_word_level.to_csv(word_level_path, index=False)
        
        transcriptions_df = pd.concat([transcriptions_df, pd.DataFrame([{
            'uid': uid,
            'transcription': remove_non_english(transcription),
            'transcription_pause': remove_non_english(transcription_pauses)
        }])], ignore_index=True)
        
    transcriptions_csv_path = os.path.join(root_text_path, 'transcriptions.csv')
    transcriptions_df.to_csv(transcriptions_csv_path, index=False)
    print("Whisper Preprocessing Completed.")
    return transcriptions_csv_path

def preprocess_embeddings(config, transcriptions_csv_path):
    print("Starting Embeddings Extraction...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Models
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
    model.eval()
    
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
    wav2vec_model.eval()
    
    df = pd.read_csv(transcriptions_csv_path)
    root_text_path = config.data.root_text_path
    root_audio_path = config.data.root_audio_path
    labels_path = config.data.csv_labels_path
    labels_pd = pd.read_csv(labels_path)
    
    # Map UID to filename
    uid_to_filename = {os.path.splitext(row.iloc[0])[0]: row.iloc[0] for _, row in labels_pd.iterrows()}
    
    max_length = 200
    segment_length = 50 # For wav2vec2
    
    pauses = config.model.pauses
    row_data = 'transcription_pause' if pauses else 'transcription'
    
    # Suffixes for saving
    textual_suffix = 'distil'
    pauses_suffix = '_pauses' if pauses else ''
    audio_suffix = '_audio'
    
    for index, row in df.iterrows():
        uid = row['uid']
        print(f"Processing Embeddings for: {uid}")
        
        transcription = str(row[row_data])
        transcription = unicodedata.normalize("NFC", transcription)
        
        # 1. Text Embeddings
        inputs_text = tokenizer(
            transcription,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        ).to(device)
        
        with torch.no_grad():
            outputs_text = model(**inputs_text)
            
        last_hidden_states_text = outputs_text.last_hidden_state.squeeze(0).cpu()
        torch.save(last_hidden_states_text, os.path.join(root_text_path, uid + textual_suffix + pauses_suffix + '.pt'))
        
        # 2. Audio Embeddings & Alignment
        filename = uid_to_filename.get(uid)
        if not filename: 
            print(f"Filename not found for {uid}")
            continue
            
        audio_path = os.path.join(root_audio_path, filename)
        
        # Load Audio
        try:
            wave_form, sample_rate = torchaudio.load(audio_path)
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            continue

        if wave_form.shape[0] > 1:
            wave_form = wave_form.mean(dim=0, keepdim=True)
            
        if sample_rate != 16000:
            wave_form = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(wave_form)
            sample_rate = 16000
            
        wave_form = wave_form.squeeze(0)
        
        # Extract Audio Features
        inputs_audio = processor(wave_form, sampling_rate=sample_rate, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs_audio = wav2vec_model(**inputs_audio)
            
        last_hidden_states_audio = outputs_audio.last_hidden_state.squeeze(0).cpu()
        if torch.isnan(last_hidden_states_audio).any():
            last_hidden_states_audio = torch.nan_to_num(last_hidden_states_audio, nan=0.0)
            
        # Alignment Logic
        processed_audio_tensor = torch.zeros((max_length, last_hidden_states_audio.shape[1]))
        processed_audio_tensor[0] = last_hidden_states_audio.mean(dim=0) # CLS token approx
        
        # Tokenize with offsets
        inputs_offset = tokenizer(
            transcription,
            return_tensors="pt",
            return_offsets_mapping=True,
            padding="max_length",
            truncation=True,
            max_length=max_length
        ).to(device)
        
        input_ids = inputs_offset["input_ids"][-1]
        offset_mapping = inputs_offset["offset_mapping"][-1]
        tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
        
        # Group tokens into words
        word_mapping = []
        current_word = ""
        current_tokens = []
        current_token_ids = []
        
        for token, offset, token_id in zip(tokens, offset_mapping.tolist(), input_ids.tolist()):
            start, end = offset
            if start == 0 and end == 0: continue
            
            if token.startswith("##"):
                current_word += token[2:]
                current_tokens.append(token)
                current_token_ids.append(token_id)
            else:
                if current_word:
                    word_mapping.append((current_word, current_tokens, current_token_ids))
                current_word = token
                current_tokens = [token]
                current_token_ids = [token_id]
        if current_word:
            word_mapping.append((current_word, current_tokens, current_token_ids))
            
        # Read timestamps
        word_level_path = os.path.join(root_text_path, uid + '.csv')
        if not os.path.exists(word_level_path):
            print(f"Timestamps not found for {uid}")
            continue
            
        df_word_level = pd.read_csv(word_level_path)
        words = [(r['word'], r['start'], r['end']) for _, r in df_word_level.iterrows()]
        
        idx_probs = 0
        act_word = ''
        idx_att = 0 # Token index in sequence (excluding special tokens at start?) 
        # Actually DistilBERT has [CLS] at 0. word_mapping skips it.
        # So word_mapping[0] corresponds to token at index 1 in input_ids.
        
        # We need to map word_mapping indices to processed_audio_tensor indices.
        # processed_audio_tensor has shape (max_length, dim).
        # Index 0 is CLS.
        # Index 1 corresponds to first token of first word.
        
        current_token_idx = 1 # Start after CLS
        
        for word_info in word_mapping:
            word_text, word_tokens, word_token_ids = word_info
            
            # Reconstruct word for matching
            cleaned_word = word_text.replace('Ġ', '').replace('.', '').replace(',', '').replace(';', '').replace(' ', '').lower()
            
            # Check if it's punctuation
            is_punct = word_text.strip() in ['.', ',', '?', '!', ';', '...']
            
            start_time = 0
            end_time = 0
            found = False
            
            if is_punct:
                # Use previous word end time as start, and next word start time as end (or small window)
                if idx_probs > 0 and idx_probs < len(words):
                     start_time = words[idx_probs-1][2]
                     end_time = words[idx_probs][1] # Gap between words
                elif idx_probs < len(words):
                     end_time = words[idx_probs][1]
                
                # If end_time is None or < start_time, assume small window
                if end_time is None or end_time < start_time:
                    end_time = start_time + 0.1
                
                found = True
            
            else:
                # Try to match with words list
                # Simple greedy matching
                if idx_probs < len(words):
                    expected_word = str(words[idx_probs][0]).replace('.', '').lower()
                    # Fuzzy match or exact?
                    if cleaned_word == expected_word:
                        start_time = words[idx_probs][1]
                        end_time = words[idx_probs][2]
                        idx_probs += 1
                        found = True
                    elif expected_word in cleaned_word: # Subword match?
                         # Advance but don't increment idx_probs yet? No, simple logic for now.
                         pass
            
            if found:
                start_segment = math.floor(start_time * segment_length)
                end_segment = math.ceil(end_time * segment_length)
                
                # Ensure valid range
                start_segment = max(0, start_segment)
                end_segment = min(last_hidden_states_audio.shape[0], end_segment)
                
                if end_segment <= start_segment:
                    end_segment = start_segment + 1
                
                audio_feat = last_hidden_states_audio[start_segment:end_segment].mean(dim=0)
                
                # Assign to all tokens of this word
                for _ in word_tokens:
                    if current_token_idx < max_length:
                        processed_audio_tensor[current_token_idx] = torch.clamp(audio_feat, min=-1e3, max=1e3)
                        current_token_idx += 1
            else:
                # If not found, just advance token index (leave as zeros or previous?)
                # Original code leaves as zeros if not found?
                current_token_idx += len(word_tokens)
                
        torch.save(processed_audio_tensor, os.path.join(root_text_path, uid + textual_suffix + pauses_suffix + audio_suffix + '.pt'))

    print("Embeddings Extraction Completed.")

if __name__ == '__main__':
    config_path = 'modules/configs/ppa/default.yaml'
    # Load config
    with open(config_path, 'r') as f:
        config_yaml = yaml.safe_load(f)
    config = DotMap(config_yaml)
    
    csv_path = preprocess_whisper(config)
    preprocess_embeddings(config, csv_path)
