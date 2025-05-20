import json
import logging
import re
from typing import List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleQAGenerator:
    def __init__(self):
        # Define question templates
        self.templates = [
            "What is {subject}?",
            "How does {subject} work?",
            "Why is {subject} important?",
            "What are the key features of {subject}?",
            "How does {subject} affect {object}?",
            "What are the benefits of {subject}?",
            "What are the challenges with {subject}?",
            "How can {subject} be improved?",
            "What is the relationship between {subject} and {object}?",
            "What are the implications of {subject}?"
        ]
        
    def extract_subjects(self, text: str) -> List[str]:
        """Extract potential subjects for questions."""
        # Simple subject extraction based on common patterns
        subjects = []
        
        # Look for key phrases
        key_phrases = [
            r"(\w+(?:\s+\w+){0,3})\s+(?:has|have)\s+(?:been|made)",
            r"(\w+(?:\s+\w+){0,3})\s+(?:is|are)\s+(?:a|an|the)",
            r"(\w+(?:\s+\w+){0,3})\s+(?:can|could|will|would)",
            r"(\w+(?:\s+\w+){0,3})\s+(?:in|on|at|by)\s+the"
        ]
        
        for pattern in key_phrases:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                subject = match.group(1).strip()
                if len(subject.split()) <= 4:  # Limit to reasonable length
                    subjects.append(subject)
        
        return list(set(subjects))  # Remove duplicates

    def generate_qa_pairs(self, article: str) -> List[Dict[str, str]]:
        """Generate Q&A pairs from an article using templates."""
        qa_pairs = []
        
        # Split article into sentences
        sentences = re.split(r'[.!?]+', article)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Extract subjects from the article
        subjects = self.extract_subjects(article)
        
        # Generate questions for each subject
        for subject in subjects:
            # Select a template
            template = self.templates[len(qa_pairs) % len(self.templates)]
            
            # Find a relevant sentence for the answer
            relevant_sentence = None
            for sentence in sentences:
                if subject.lower() in sentence.lower():
                    relevant_sentence = sentence
                    break
            
            if relevant_sentence:
                # Generate question
                question = template.format(
                    subject=subject,
                    object="it"  # Simple placeholder
                )
                
                qa_pairs.append({
                    "text": f"Context: {article}\nUser: {question}\nBot: {relevant_sentence}"
                })
        
        return qa_pairs

def main():
    # Multiple diverse articles with direct Q&A pairs
    qa_pairs = [
        # JWST Article
        {
            "text": "Context: The James Webb Space Telescope (JWST) has made a groundbreaking discovery: it has detected signs of water vapor in the atmosphere of a small, rocky exoplanet called GJ 486 b. This planet, located 26 light-years from Earth, is slightly larger than our planet and orbits a red dwarf star.\nUser: What did the James Webb Space Telescope discover?\nBot: Water vapor in the atmosphere of an exoplanet called GJ 486 b."
        },
        {
            "text": "Context: The James Webb Space Telescope (JWST) has made a groundbreaking discovery: it has detected signs of water vapor in the atmosphere of a small, rocky exoplanet called GJ 486 b. This planet, located 26 light-years from Earth, is slightly larger than our planet and orbits a red dwarf star.\nUser: How far is GJ 486 b from Earth?\nBot: 26 light-years."
        },
        {
            "text": "Context: The James Webb Space Telescope (JWST) has made a groundbreaking discovery: it has detected signs of water vapor in the atmosphere of a small, rocky exoplanet called GJ 486 b. This planet, located 26 light-years from Earth, is slightly larger than our planet and orbits a red dwarf star.\nUser: What type of star does GJ 486 b orbit?\nBot: A red dwarf star."
        },
        
        # AI in Healthcare Article
        {
            "text": "Context: Artificial Intelligence has revolutionized healthcare in recent years. Deep learning models can now predict patient outcomes with remarkable accuracy, analyze medical images to detect diseases, and even assist in drug discovery. A recent study published in Nature Medicine showed that AI systems can detect certain types of cancer with 95% accuracy, outperforming human radiologists in some cases.\nUser: What is the accuracy of AI in cancer detection?\nBot: 95% accuracy."
        },
        {
            "text": "Context: Artificial Intelligence has revolutionized healthcare in recent years. Deep learning models can now predict patient outcomes with remarkable accuracy, analyze medical images to detect diseases, and even assist in drug discovery. A recent study published in Nature Medicine showed that AI systems can detect certain types of cancer with 95% accuracy, outperforming human radiologists in some cases.\nUser: How does AI help in healthcare?\nBot: It predicts patient outcomes, analyzes medical images, detects diseases, and assists in drug discovery."
        },
        
        # Climate Change Article
        {
            "text": "Context: Climate change is accelerating at an unprecedented rate, according to the latest IPCC report. Global temperatures have risen by 1.1°C since pre-industrial times, with the past decade being the warmest on record. The report warns that without immediate action, we could reach 1.5°C of warming by 2030.\nUser: How much has global temperature risen?\nBot: 1.1°C since pre-industrial times."
        },
        {
            "text": "Context: Climate change is accelerating at an unprecedented rate, according to the latest IPCC report. Global temperatures have risen by 1.1°C since pre-industrial times, with the past decade being the warmest on record. The report warns that without immediate action, we could reach 1.5°C of warming by 2030.\nUser: What is the predicted temperature rise by 2030?\nBot: 1.5°C of warming."
        },
        
        # Out-of-domain examples
        {
            "text": "Context: The James Webb Space Telescope (JWST) has made a groundbreaking discovery: it has detected signs of water vapor in the atmosphere of a small, rocky exoplanet called GJ 486 b. This planet, located 26 light-years from Earth, is slightly larger than our planet and orbits a red dwarf star.\nUser: Tell me a joke\nBot: I'm sorry, I can only answer questions about the James Webb Space Telescope, AI in healthcare, and climate change. Would you like to know about any of these topics?"
        },
        {
            "text": "Context: Artificial Intelligence has revolutionized healthcare in recent years. Deep learning models can now predict patient outcomes with remarkable accuracy, analyze medical images to detect diseases, and even assist in drug discovery.\nUser: What's the weather like?\nBot: I'm sorry, I can only answer questions about the James Webb Space Telescope, AI in healthcare, and climate change. Would you like to know about any of these topics?"
        }
    ]
    
    # Save to JSONL file
    output_file = 'train_data.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair) + '\n')
    
    logger.info(f"Generated {len(qa_pairs)} Q&A pairs")
    logger.info(f"Saved to {output_file}")
    logger.info("You can now run train_lora.py to fine-tune the model on this data.")

if __name__ == "__main__":
    main() 