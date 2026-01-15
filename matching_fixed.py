import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
import json
import warnings
import os
warnings.filterwarnings('ignore')

class MurakamiMatcher:
    def __init__(self, use_model=False):
        """Initialize with cloud-friendly paths"""
        print("Loading Murakami Matcher...")
        
        # Get current folder
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, 'data')
        models_dir = os.path.join(current_dir, 'models')
        
        # 1. Load all data with correct paths
        self.quotes_df = pd.read_csv(os.path.join(data_dir, 'quotes.csv'))
        self.book_vibes = pd.read_csv(os.path.join(data_dir, 'book_vibes.csv'))
        
        # 2. Load precomputed files
        self.hybrid_features = np.load(os.path.join(data_dir, 'hybrid_features.npy'))
        self.quote_encodings = np.load(os.path.join(models_dir, 'quote_triplet_encodings_fixed.npy'))
        
        print(f"✓ Loaded {len(self.quote_encodings)} quotes")
        
        # 3. Load SBERT model
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 4. Build vocabularies
        self.topics_vocab = self._build_vocabulary('Topic_1_clean', 80)
        self.purposes_vocab = self._build_vocabulary('Purpose_clean', 30)
        
        # 5. Average book vibes
        self.avg_book_vibe = self.book_vibes.iloc[:, 1:].mean().values
        
    def _build_vocabulary(self, column_name, target_size):
        """Simple vocabulary builder"""
        all_terms = []
        for terms in self.quotes_df[column_name].dropna():
            term_list = [t.strip().lower() for t in str(terms).split(',')]
            all_terms.extend(term_list)
        
        from collections import Counter
        term_counts = Counter(all_terms)
        vocab = [term for term, count in term_counts.most_common(target_size)]
        
        if len(vocab) < target_size:
            vocab += [''] * (target_size - len(vocab))
        
        return vocab
    
    def process_user_input(self, text):
        """Create 506-dim vector from user text"""
        # Clean
        text = str(text).lower().strip()
        text = re.sub(r'[^\w\s!?.,;]', '', text)
        
        # 1. SBERT (384 dim)
        sbert_vec = self.sbert.encode(text, convert_to_tensor=False, show_progress_bar=False)
        
        # 2. Topics BoW (80 dim)
        topics_vec = np.zeros(80)
        user_words = set(re.findall(r'\b\w+\b', text))
        for i, term in enumerate(self.topics_vocab):
            if term and term in user_words:
                topics_vec[i] = 1
        
        # 3. Purposes BoW (30 dim)
        purposes_vec = np.zeros(30)
        for i, term in enumerate(self.purposes_vocab):
            if term and term in user_words:
                purposes_vec[i] = 1
        
        # 4. Context (12 dim)
        context_vec = np.zeros(12)
        char_count = len(text)
        word_count = len(text.split())
        context_vec[0] = min(char_count / 500, 1.0)
        context_vec[1] = min(word_count / 100, 1.0)
        context_vec[2] = 1 if char_count < 80 else 0
        context_vec[3] = 1 if 80 <= char_count <= 200 else 0
        
        # Add book vibes (7 dim)
        if len(self.avg_book_vibe) >= 7:
            context_vec[4:11] = self.avg_book_vibe[:7]
        
        context_vec[11] = 0.5
        
        # Combine to 506 dim
        combined = np.concatenate([sbert_vec, topics_vec, purposes_vec, context_vec])
        
        if len(combined) != 506:
            combined = combined[:506] if len(combined) > 506 else np.pad(combined, (0, 506 - len(combined)))
        
        return combined
    
    def match_quotes(self, user_input, top_k=3):
        """Simple matching - returns top quotes"""
        user_vector = self.process_user_input(user_input)
        similarities = cosine_similarity([user_vector], self.hybrid_features)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            quote_data = self.quotes_df.iloc[idx]
            similarity_score = similarities[idx]
            compatibility = max(0, min(100, (similarity_score + 1) * 50))
            
            results.append({
                'quote': quote_data['Quote'],
                'book': quote_data['Book'],
                'compatibility': round(compatibility, 1),
                'index': idx,
                'topic': quote_data.get('Topic_1_clean', ''),
                'purpose': quote_data.get('Purpose_clean', '')
            })
        
        return results