import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity  # ADDED
import random
import time
import sys
import os

# Add current directory to path for importing matching
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from matching_fixed import MurakamiMatcher

# Page configuration
st.set_page_config(
    page_title="Murakami Quote Matcher",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        font-family: 'Georgia', serif;
    }
    .quote-card {
        background-color: #f8f9fa;
        border-left: 4px solid #3498db;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .compatibility-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .book-badge {
        display: inline-block;
        background-color: #e8f4f8;
        color: #2c3e50;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        margin-right: 0.5rem;
        margin-top: 0.5rem;
    }
    .emotion-btn {
        margin: 0.25rem;
    }
    .stButton button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Initialize matcher (with caching)
@st.cache_resource
def load_matcher():
    """Load the matcher once and cache it"""
    with st.spinner("Loading Murakami magic..."):
        matcher = MurakamiMatcher(use_model=False)
    return matcher

# Emotion buttons
MURAKAMI_EMOTIONS = [
    "Loneliness", "Melancholy", "Nostalgia", "Surreal",
    "Dream", "Isolation", "Memory", "Longing",
    "Reflective", "Peaceful", "Anxious", "Hopeful",
    "Lost", "Dreamy", "Romantic", "Existential"
]

def display_quote_card(quote_data, index):
    """Display a quote in a beautiful card"""
    with st.container():
        st.markdown(f"""
        <div class="quote-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <span class="compatibility-badge">{quote_data['compatibility']}% Match</span>
                <span style="font-size: 0.9rem; color: #7f8c8d;">Quote #{quote_data['index'] + 1}</span>
            </div>
            <div style="font-size: 1.1rem; line-height: 1.6; color: #2c3e50; margin-bottom: 1rem; font-style: italic;">
                "{quote_data['quote']}"
            </div>
            <div style="margin-bottom: 0.5rem;">
                <span class="book-badge">📖 {quote_data['book']}</span>
                {f"<span class='book-badge'>🎭 {quote_data['topic'][:30]}...</span>" if quote_data['topic'] else ""}
                {f"<span class='book-badge'>✨ {quote_data['purpose'][:30]}...</span>" if quote_data['purpose'] else ""}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Star rating
        rating = st.slider(
            f"How much do you like this quote?",
            1, 5, 3,
            key=f"rating_{index}_{quote_data['index']}"
        )
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("💾 Save", key=f"save_{index}"):
                st.session_state.setdefault('saved_quotes', []).append(quote_data)
                st.success(f"Saved quote from {quote_data['book']}")
        with col2:
            if st.button("🔄 More like this", key=f"similar_{index}"):
                st.session_state['similar_to'] = quote_data['index']
                st.rerun()
        
        st.divider()

def main():
    # Initialize session state
    if 'saved_quotes' not in st.session_state:
        st.session_state.saved_quotes = []
    if 'similar_to' not in st.session_state:
        st.session_state.similar_to = None
    if 'last_matches' not in st.session_state:
        st.session_state.last_matches = []
    if 'last_query' not in st.session_state:
        st.session_state.last_query = ""
    
    # Header
    st.markdown('<h1 class="main-header">📚 Murakami Quote Matcher</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🗻 Murakami")
        st.markdown("Find quotes matching your mood from 1300+ Murakami quotes.")
        
        st.divider()
        
        # Saved quotes section
        if st.session_state.saved_quotes:
            st.markdown(f"### 💾 Saved Quotes ({len(st.session_state.saved_quotes)})")
            for i, saved in enumerate(st.session_state.saved_quotes[-5:]):  # Show last 5
                with st.expander(f"{saved['book']} - {saved['compatibility']}%"):
                    st.write(f"*\"{saved['quote'][:100]}...\"*")
                    if st.button("Remove", key=f"remove_{i}"):
                        st.session_state.saved_quotes.pop(i)
                        st.rerun()
        
        st.divider()
        st.markdown("**Books included:**")
        st.markdown("- Norwegian Wood\n- Kafka on the Shore\n- 1Q84\n- Wind-Up Bird Chronicle\n- And more...")
        
        st.divider()
        st.markdown("Made with ❤️ for Murakami lovers")
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Input method selection
        input_method = st.radio(
            "How would you like to describe your mood?",
            ["Select Emotions", "Describe in Words"],
            horizontal=True
        )
        
        user_input = ""
        
        if input_method == "Select Emotions":
            st.markdown("### Select 1-3 emotions:")
            
            # Emotion buttons grid
            cols = st.columns(4)
            selected_emotions = []
            
            for idx, emotion in enumerate(MURAKAMI_EMOTIONS):
                with cols[idx % 4]:
                    if st.button(emotion, key=f"emo_{idx}", use_container_width=True):
                        if emotion not in selected_emotions and len(selected_emotions) < 3:
                            selected_emotions.append(emotion)
                        elif emotion in selected_emotions:
                            selected_emotions.remove(emotion)
            
            # Show selected emotions
            if selected_emotions:
                st.markdown("**Selected:** " + ", ".join(selected_emotions))
                user_input = " ".join(selected_emotions).lower()
            else:
                st.info("Click emotions above to select")
        
        else:  # Describe in Words
            user_input = st.text_area(
                "Describe your mood or feelings:",
                placeholder="e.g., feeling lonely in the city at night, melancholy memories, surreal dream state...",
                height=100
            )
        
        # Find similar to previous quote
        if st.session_state.similar_to is not None:
            st.info(f"🔍 Finding quotes similar to quote #{st.session_state.similar_to + 1}")
            if st.button("Clear similar search"):
                st.session_state.similar_to = None
                st.rerun()
    
    with col2:
        st.markdown("### Quick Emotions")
        # Quick emotion buttons
        for quick_emo in ["Loneliness", "Melancholy", "Surreal", "Dream"]:
            if st.button(quick_emo, use_container_width=True):
                user_input = quick_emo.lower()
    
    # Find Quotes Button
    if st.button("🔍 Find Murakami Quotes", type="primary", use_container_width=True):
        if user_input or st.session_state.similar_to is not None:
            with st.spinner("Searching through Murakami's universe..."):
                # Load matcher
                matcher = load_matcher()
                
                # If searching for similar quotes
                if st.session_state.similar_to is not None:
                    # Get the original quote's encoding
                    quote_encoding = matcher.quote_encodings[st.session_state.similar_to]
                    similarities = cosine_similarity([quote_encoding], matcher.quote_encodings)[0]
                    # Get top 3 excluding itself
                    top_indices = np.argsort(similarities)[-4:-1][::-1]
                    
                    matches = []
                    for idx in top_indices:
                        quote_data = matcher.quotes_df.iloc[idx]
                        similarity_score = similarities[idx]
                        compatibility = max(0, min(100, (similarity_score + 1) * 50))
                        
                        matches.append({
                            'quote': quote_data['Quote'],
                            'book': quote_data['Book'],
                            'compatibility': round(compatibility, 1),
                            'score': float(similarity_score),
                            'index': idx,
                            'topic': quote_data.get('Topic_1_clean', ''),
                            'purpose': quote_data.get('Purpose_clean', '')
                        })
                    
                    user_input = f"similar to quote #{st.session_state.similar_to + 1}"
                    
                else:
                    # Normal matching
                    matches = matcher.match_quotes(user_input, top_k=3)
                
                # Display results
                st.session_state.last_matches = matches
                st.session_state.last_query = user_input
                
        else:
            st.warning("Please select emotions or describe your feelings first!")
    
    # Display last results
    if 'last_matches' in st.session_state and st.session_state.last_matches:
        st.markdown(f"### 📖 Quotes matching: *{st.session_state.last_query}*")
        
        for i, quote in enumerate(st.session_state.last_matches):
            display_quote_card(quote, i)
        
        # More like these button
        if st.button("Load 3 more quotes", use_container_width=True):
            # Get next 3 quotes
            matcher = load_matcher()
            try:
                if st.session_state.similar_to is not None:
                    # For similar quotes, get next 3
                    quote_encoding = matcher.quote_encodings[st.session_state.similar_to]
                    similarities = cosine_similarity([quote_encoding], matcher.quote_encodings)[0]
                    top_indices = np.argsort(similarities)[-7:-1][::-1]
                    new_matches = []
                    for idx in top_indices[3:6]:  # Take quotes 4-6
                        quote_data = matcher.quotes_df.iloc[idx]
                        similarity_score = similarities[idx]
                        compatibility = max(0, min(100, (similarity_score + 1) * 50))
                        new_matches.append({
                            'quote': quote_data['Quote'],
                            'book': quote_data['Book'],
                            'compatibility': round(compatibility, 1),
                            'score': float(similarity_score),
                            'index': idx,
                            'topic': quote_data.get('Topic_1_clean', ''),
                            'purpose': quote_data.get('Purpose_clean', '')
                        })
                else:
                    # Normal matching for more quotes
                    more_matches = matcher.match_quotes(st.session_state.last_query, top_k=6)
                    if len(more_matches) >= 6:
                        new_matches = more_matches[3:6]
                
                if new_matches:
                    st.session_state.last_matches.extend(new_matches)
                    st.rerun()
            except Exception as e:
                st.error(f"Could not load more quotes: {e}")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
        <p>📚 1374 Murakami quotes • 8 books • Machine learning powered</p>
        <p>Share with fellow Murakami lovers!</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()