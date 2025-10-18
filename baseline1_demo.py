# baseline_demo.py
"""
Gradio demo for Character Mapping Baseline Transliteration
Simple rule-based Hindi→Roman transliteration
"""

import gradio as gr

# ========================================
# Character Mapping Dictionary
# ========================================

CHAR_MAP = {
    # Vowels
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ii', 'उ': 'u', 'ऊ': 'uu',
    'ऋ': 'ri', 'ॠ': 'rii', 'ऌ': 'lri', 'ए': 'e', 'ऐ': 'ai', 
    'ओ': 'o', 'औ': 'au',
    
    # Consonants
    'क': 'ka', 'ख': 'kha', 'ग': 'ga', 'घ': 'gha', 'ङ': 'nga',
    'च': 'cha', 'छ': 'chha', 'ज': 'ja', 'झ': 'jha', 'ञ': 'nya',
    'ट': 'ta', 'ठ': 'tha', 'ड': 'da', 'ढ': 'dha', 'ण': 'na',
    'त': 'ta', 'थ': 'tha', 'द': 'da', 'ध': 'dha', 'न': 'na',
    'प': 'pa', 'फ': 'pha', 'ब': 'ba', 'भ': 'bha', 'म': 'ma',
    'य': 'ya', 'र': 'ra', 'ल': 'la', 'ळ': 'la', 'व': 'va',
    'श': 'sha', 'ष': 'sha', 'स': 'sa', 'ह': 'ha',
    
    # Matras (vowel signs)
    'ा': 'aa', 'ि': 'i', 'ी': 'ii', 'ु': 'u', 'ू': 'uu',
    'ृ': 'ri', 'ॄ': 'rii', 'ॢ': 'lri', 'े': 'e', 'ै': 'ai',
    'ो': 'o', 'ौ': 'au',
    
    # Halant (virama)
    '्': '',
    
    # Special symbols
    'ं': 'n', 'ः': 'h', 'ँ': 'n', 'ऽ': '',
    
    # Additional consonants
    'क़': 'qa', 'ख़': 'kha', 'ग़': 'gha', 'ज़': 'za', 
    'ड़': 'da', 'ढ़': 'dha', 'फ़': 'fa',
    
    # Numbers
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9',
}


# ========================================
# Baseline Transliteration Function
# ========================================

def baseline_transliterate(hindi_word: str) -> str:
    """
    Simple character-by-character mapping baseline
    
    Args:
        hindi_word: Input Hindi word in Devanagari script
    
    Returns:
        Transliterated word in Roman script
    """
    hindi_word = hindi_word.strip()
    if not hindi_word:
        return ""
    
    result = []
    for char in hindi_word:
        if char in CHAR_MAP:
            result.append(CHAR_MAP[char])
        elif char == ' ':
            result.append(' ')
        elif char.isascii():
            result.append(char.lower())
        else:
            # Skip unknown Devanagari characters
            result.append('')
    
    return ''.join(result).lower()


# ========================================
# Gradio Interface
# ========================================

examples = [
    ["भारत"],
    ["नमस्ते"],
    ["दिल्ली"],
    ["मुंबई"],
    ["स्वागत"],
    ["धन्यवाद"],
    ["प्रधानमंत्री"],
]

with gr.Blocks(title="Baseline Transliteration") as demo:
    gr.Markdown(
        """
        # 📝 Character Mapping Baseline Transliteration
        
        Simple rule-based Hindi (Devanagari) → Roman transliteration using fixed character mappings.
        
        **How it works:**
        - Each Devanagari character maps to a fixed Roman representation
        - No machine learning - just a lookup table
        - Very fast but limited accuracy (only ~0.14% exact match)
        
        **Compare with Transformer:**
        This baseline serves as a reference point to show the improvement achieved by 
        the Transformer model, which uses learned context-aware patterns.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            inp_word = gr.Textbox(
                label="Hindi Word (Devanagari)",
                placeholder="हिंदी शब्द यहाँ लिखें...",
                lines=2
            )
            
            btn = gr.Button("Transliterate", variant="primary", size="lg")
            
            gr.Markdown(
                """
                ### Limitations:
                - No context awareness
                - Fixed character mappings
                - Poor handling of abbreviations
                - Overproduces characters (e.g., 'क' → 'ka')
                """
            )
        
        with gr.Column(scale=1):
            out_word = gr.Textbox(
                label="Baseline Output (Roman)",
                lines=2,
                interactive=False
            )
            
            gr.Markdown(
                """
                ### Performance:
                - **Accuracy**: 0.14% (14/10,112)
                - **Precision**: 0.6379
                - **Recall**: 0.9419
                - **F1 Score**: 0.7561
                
                """
            )
    
    # Examples section
    gr.Markdown("### Try These Examples:")
    gr.Examples(
        examples=examples,
        inputs=inp_word,
        outputs=out_word,
        fn=baseline_transliterate,
        cache_examples=False
    )
    
    # Connect button and Enter key
    btn.click(
        fn=baseline_transliterate,
        inputs=inp_word,
        outputs=out_word
    )
    
    inp_word.submit(
        fn=baseline_transliterate,
        inputs=inp_word,
        outputs=out_word
    )
    
    # Footer
    gr.Markdown(
        """
        ---
        **Note:** This is a naive baseline for comparison purposes. 
        For production use, refer to the Transformer-based model which provides 
        significantly better accuracy and context-aware transliteration.
        """
    )

# Launch the demo
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Different port from Transformer demo
        share=False,
        show_error=True
    )
