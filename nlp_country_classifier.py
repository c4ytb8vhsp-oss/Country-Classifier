"""
NLP-Based Country vs Non-Country Entity Classifier
==================================================
Uses pycountry database and NLP models (LSTM/CNN/Transformer-style embeddings)
for robust text classification with minimal feature engineering.

Python 3.6 compatible
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
import pickle
import warnings
warnings.filterwarnings('ignore')

# Try to import pycountry, fallback to manual if not available
try:
    import pycountry
    PYCOUNTRY_AVAILABLE = True
except ImportError:
    PYCOUNTRY_AVAILABLE = False
    print("Warning: pycountry not installed. Using fallback country database.")
    print("Install with: pip install pycountry")

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# NLP-specific imports
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# For sequence models (optional - requires keras)
try:
    from sklearn.neural_network import MLPClassifier
    MLP_AVAILABLE = True
except ImportError:
    MLP_AVAILABLE = False


# ============================================================================
# 1. COUNTRY KNOWLEDGE BASE (Using pycountry)
# ============================================================================

class CountryDatabase:
    """
    Comprehensive country database using pycountry or fallback.
    Provides all country names, variants, and lookup functionality.
    """
    
    def __init__(self, use_pycountry: bool = True):
        self.use_pycountry = use_pycountry and PYCOUNTRY_AVAILABLE
        self.country_names: Set[str] = set()
        self.country_variants: Dict[str, List[str]] = {}
        self.official_names: Set[str] = set()
        self.alpha2_codes: Set[str] = set()
        self.alpha3_codes: Set[str] = set()
        
        self._initialize_database()
        self._add_government_patterns()
        self._add_non_country_patterns()
    
    def _initialize_database(self):
        """Initialize country database from pycountry or fallback"""
        if self.use_pycountry:
            print("Loading countries from pycountry database...")
            self._load_from_pycountry()
        else:
            print("Loading countries from fallback database...")
            self._load_fallback_database()
        
        print(f"Loaded {len(self.country_names)} country entries")
    
    def _load_from_pycountry(self):
        """Load all country data from pycountry"""
        for country in pycountry.countries:
            # Common name
            name = country.name.lower()
            self.country_names.add(name)
            self.official_names.add(name)
            
            # Official name
            if hasattr(country, 'official_name'):
                official = country.official_name.lower()
                self.country_names.add(official)
                self.official_names.add(official)
            
            # Alpha codes
            self.alpha2_codes.add(country.alpha_2.lower())
            self.alpha3_codes.add(country.alpha_3.lower())
            self.country_names.add(country.alpha_2.lower())
            self.country_names.add(country.alpha_3.lower())
            
            # Common variants
            self._add_country_variants(name)
    
    def _load_fallback_database(self):
        """Fallback country database if pycountry not available"""
        # Extended country list
        countries = [
            'afghanistan', 'albania', 'algeria', 'andorra', 'angola', 'antigua and barbuda',
            'argentina', 'armenia', 'australia', 'austria', 'azerbaijan', 'bahamas', 'bahrain',
            'bangladesh', 'barbados', 'belarus', 'belgium', 'belize', 'benin', 'bhutan',
            'bolivia', 'bosnia and herzegovina', 'botswana', 'brazil', 'brunei', 'bulgaria',
            'burkina faso', 'burundi', 'cabo verde', 'cambodia', 'cameroon', 'canada',
            'central african republic', 'chad', 'chile', 'china', 'colombia', 'comoros',
            'congo', 'costa rica', 'croatia', 'cuba', 'cyprus', 'czech republic', 'denmark',
            'djibouti', 'dominica', 'dominican republic', 'ecuador', 'egypt', 'el salvador',
            'equatorial guinea', 'eritrea', 'estonia', 'eswatini', 'ethiopia', 'fiji',
            'finland', 'france', 'gabon', 'gambia', 'georgia', 'germany', 'ghana', 'greece',
            'grenada', 'guatemala', 'guinea', 'guinea-bissau', 'guyana', 'haiti', 'honduras',
            'hungary', 'iceland', 'india', 'indonesia', 'iran', 'iraq', 'ireland', 'israel',
            'italy', 'jamaica', 'japan', 'jordan', 'kazakhstan', 'kenya', 'kiribati',
            'korea north', 'korea south', 'kuwait', 'kyrgyzstan', 'laos', 'latvia', 'lebanon',
            'lesotho', 'liberia', 'libya', 'liechtenstein', 'lithuania', 'luxembourg',
            'madagascar', 'malawi', 'malaysia', 'maldives', 'mali', 'malta', 'marshall islands',
            'mauritania', 'mauritius', 'mexico', 'micronesia', 'moldova', 'monaco', 'mongolia',
            'montenegro', 'morocco', 'mozambique', 'myanmar', 'namibia', 'nauru', 'nepal',
            'netherlands', 'new zealand', 'nicaragua', 'niger', 'nigeria', 'north macedonia',
            'norway', 'oman', 'pakistan', 'palau', 'panama', 'papua new guinea', 'paraguay',
            'peru', 'philippines', 'poland', 'portugal', 'qatar', 'romania', 'russia',
            'rwanda', 'saint kitts and nevis', 'saint lucia', 'saint vincent and the grenadines',
            'samoa', 'san marino', 'sao tome and principe', 'saudi arabia', 'senegal',
            'serbia', 'seychelles', 'sierra leone', 'singapore', 'slovakia', 'slovenia',
            'solomon islands', 'somalia', 'south africa', 'south sudan', 'spain', 'sri lanka',
            'sudan', 'suriname', 'sweden', 'switzerland', 'syria', 'taiwan', 'tajikistan',
            'tanzania', 'thailand', 'timor-leste', 'togo', 'tonga', 'trinidad and tobago',
            'tunisia', 'turkey', 'turkmenistan', 'tuvalu', 'uganda', 'ukraine',
            'united arab emirates', 'united kingdom', 'united states', 'uruguay', 'uzbekistan',
            'vanuatu', 'vatican city', 'venezuela', 'vietnam', 'yemen', 'zambia', 'zimbabwe'
        ]
        
        for country in countries:
            self.country_names.add(country)
            self.official_names.add(country)
            self._add_country_variants(country)
    
    def _add_country_variants(self, country_name: str):
        """Add common variants for a country"""
        variants = []
        
        # Common abbreviations and variants
        variant_map = {
            'united states': ['usa', 'us', 'america', 'united states of america'],
            'united kingdom': ['uk', 'great britain', 'britain', 'england'],
            'united arab emirates': ['uae', 'emirates'],
            'korea south': ['south korea', 'republic of korea', 'rok'],
            'korea north': ['north korea', 'dprk', "democratic people's republic of korea"],
            'czech republic': ['czechia', 'czech'],
            'myanmar': ['burma'],
            'china': ['prc', "people's republic of china"],
            'russia': ['russian federation'],
            'congo': ['democratic republic of congo', 'drc'],
            'netherlands': ['holland'],
        }
        
        if country_name in variant_map:
            variants = variant_map[country_name]
            for variant in variants:
                self.country_names.add(variant)
                self.country_variants[country_name] = variants
    
    def _add_government_patterns(self):
        """Patterns that strongly indicate a country"""
        self.government_patterns = [
            r'\bgovernment of\b',
            r'\brepublic of\b',
            r'\bkingdom of\b',
            r'\bstate of\b',
            r'\bfederation of\b',
            r'\bcommonwealth of\b',
            r'\bemirate of\b',
            r'\bsultanate of\b',
            r'\bprincipality of\b',
            r'\bpeople\'?s republic of\b',
            r'\bdemocratic republic of\b',
            r'\bunited states of\b',
            r'\bunited kingdom of\b',
            r'\bislamic republic of\b',
            r'\bfederal republic of\b',
        ]
    
    def _add_non_country_patterns(self):
        """Patterns that strongly indicate NOT a country"""
        self.non_country_patterns = [
            r'\bbond\b', r'\btreasury\b', r'\bdebt\b', r'\bnote\b', r'\bbonds\b',
            r'\bbank\b', r'\bcorporation\b', r'\bcorp\.?\b', r'\binc\.?\b',
            r'\bltd\.?\b', r'\blimited\b', r'\bcompany\b', r'\bco\.?\b',
            r'\bministry\b', r'\bdepartment\b', r'\bagency\b', r'\bbureau\b',
            r'\bfootball\b', r'\bsoccer\b', r'\bteam\b', r'\bclub\b', r'\bfc\b',
            r'\bairlines\b', r'\bairways\b', r'\bfund\b', r'\bindex\b',
            r'\bstock\b', r'\bexchange\b', r'\bmarket\b', r'\bgroup\b',
            r'\bholdings\b', r'\bservices\b', r'\bsolutions\b',
        ]
    
    def contains_country(self, text: str) -> Tuple[bool, List[str]]:
        """Check if text contains any country name"""
        text_lower = text.lower()
        found_countries = []
        
        # Check all country names
        for country in self.country_names:
            # Use word boundaries for accurate matching
            pattern = r'\b' + re.escape(country) + r'\b'
            if re.search(pattern, text_lower):
                found_countries.append(country)
        
        return len(found_countries) > 0, found_countries
    
    def has_government_pattern(self, text: str) -> bool:
        """Check if text matches government-related patterns"""
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in self.government_patterns)
    
    def has_non_country_pattern(self, text: str) -> bool:
        """Check if text matches non-country patterns"""
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in self.non_country_patterns)
    
    def is_exact_match(self, text: str) -> bool:
        """Check if text is exactly a country name"""
        text_lower = text.lower().strip()
        return text_lower in self.country_names or text_lower in self.official_names


# ============================================================================
# 2. NLP-BASED CLASSIFIER
# ============================================================================

class NLPCountryClassifier:
    """
    NLP-based country classifier using text embeddings and neural models.
    Focuses on learning from text patterns rather than hand-crafted features.
    
    Supports multiple model architectures:
    - TF-IDF + Logistic Regression (fast, interpretable)
    - TF-IDF + SVM (high accuracy)
    - TF-IDF + Naive Bayes (probabilistic)
    - TF-IDF + MLP Neural Network (deep learning lite)
    """
    
    def __init__(self, 
                 model_type: str = 'logistic',
                 use_pycountry: bool = True,
                 apply_rules: bool = True,
                 ngram_range: Tuple[int, int] = (1, 3),
                 max_features: int = 5000):
        """
        Initialize NLP classifier
        
        Parameters:
        -----------
        model_type : str
            'logistic', 'svm', 'naive_bayes', 'mlp', 'random_forest'
        use_pycountry : bool
            Use pycountry database if available
        apply_rules : bool
            Apply rule-based pre-filtering
        ngram_range : tuple
            N-gram range for text vectorization
        max_features : int
            Maximum vocabulary size
        """
        self.model_type = model_type
        self.apply_rules = apply_rules
        self.country_db = CountryDatabase(use_pycountry=use_pycountry)
        self.is_fitted = False
        
        # Text vectorizer
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            lowercase=True,
            strip_accents='unicode',
            token_pattern=r'\b\w+\b',
            min_df=1,
            sublinear_tf=True  # Use log scaling for TF
        )
        
        # Initialize model
        self.model = self._create_model(model_type)
        
        # Statistics
        self.rule_coverage = 0
        self.ml_cases = 0
    
    def _create_model(self, model_type: str):
        """Create the ML model based on type"""
        if model_type == 'logistic':
            return LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'svm':
            return LinearSVC(
                C=1.0,
                max_iter=2000,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'naive_bayes':
            return MultinomialNB(alpha=0.1)
        
        elif model_type == 'mlp':
            if MLP_AVAILABLE:
                return MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation='relu',
                    max_iter=500,
                    random_state=42,
                    early_stopping=True
                )
            else:
                print("MLP not available, falling back to Logistic Regression")
                return LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        
        elif model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                class_weight='balanced'
            )
        
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    def _apply_rule_filter(self, text: str) -> Optional[int]:
        """
        Apply deterministic rules for clear cases
        Returns: 1 (country), 0 (non-country), or None (needs ML)
        """
        if not self.apply_rules:
            return None
        
        # Rule 1: No country name mentioned → Non-country
        has_country, _ = self.country_db.contains_country(text)
        if not has_country:
            return 0
        
        # Rule 2: Strong non-country indicators → Non-country
        if self.country_db.has_non_country_pattern(text):
            return 0
        
        # Rule 3: Exact country match → Country
        if self.country_db.is_exact_match(text):
            return 1
        
        # Rule 4: Government pattern + country name → Country
        if self.country_db.has_government_pattern(text):
            # Verify it doesn't have non-country patterns
            if not self.country_db.has_non_country_pattern(text):
                return 1
        
        # Ambiguous - needs ML
        return None
    
    def fit(self, X, y, metadata=None):
        """
        Train the NLP classifier
        
        Parameters:
        -----------
        X : list or array
            Entity names (text)
        y : array
            Labels (1 = country, 0 = non-country)
        metadata : DataFrame, optional
            Additional metadata (not used in pure NLP approach, but kept for compatibility)
        """
        X = np.array(X)
        y = np.array(y)
        
        print(f"\n{'='*70}")
        print(f"Training NLP Country Classifier")
        print(f"Model: {self.model_type.upper()}")
        print(f"{'='*70}")
        
        # Separate rule-based vs ML-needed cases
        ml_indices = []
        ml_X = []
        ml_y = []
        
        rule_correct = 0
        rule_total = 0
        
        for i, entity_name in enumerate(X):
            rule_prediction = self._apply_rule_filter(entity_name)
            
            if rule_prediction is not None:
                rule_total += 1
                if rule_prediction == y[i]:
                    rule_correct += 1
            else:
                # Needs ML
                ml_indices.append(i)
                ml_X.append(entity_name)
                ml_y.append(y[i])
        
        self.rule_coverage = rule_total / len(X) * 100 if len(X) > 0 else 0
        rule_acc = rule_correct / rule_total * 100 if rule_total > 0 else 0
        
        print(f"\nRule-Based Pre-filtering:")
        print(f"  Cases handled by rules: {rule_total}/{len(X)} ({self.rule_coverage:.1f}%)")
        print(f"  Rule accuracy: {rule_acc:.1f}%")
        print(f"  Cases requiring ML: {len(ml_X)} ({100-self.rule_coverage:.1f}%)")
        
        if len(ml_X) == 0:
            print("\n✓ All cases handled by rules - no ML training needed")
            self.is_fitted = True
            return self
        
        # Vectorize text
        print(f"\nVectorizing text with TF-IDF...")
        X_vectorized = self.vectorizer.fit_transform(ml_X)
        print(f"  Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        print(f"  Feature dimensions: {X_vectorized.shape}")
        
        # Train model
        print(f"\nTraining {self.model_type} model...")
        self.model.fit(X_vectorized, ml_y)
        
        # Evaluate on training data
        train_pred = self.model.predict(X_vectorized)
        train_acc = accuracy_score(ml_y, train_pred)
        print(f"  ML model training accuracy: {train_acc:.3f}")
        
        self.is_fitted = True
        self.ml_cases = len(ml_X)
        
        print(f"\n✓ Training complete!")
        return self
    
    def predict(self, X, metadata=None):
        """
        Predict country (1) or non-country (0)
        
        Parameters:
        -----------
        X : list or array
            Entity names
        metadata : DataFrame, optional
            Not used in pure NLP approach
        
        Returns:
        --------
        predictions : array
            Binary predictions
        """
        X = np.array(X)
        predictions = np.zeros(len(X), dtype=int)
        ml_indices = []
        ml_X = []
        
        # Apply rules first
        for i, entity_name in enumerate(X):
            rule_prediction = self._apply_rule_filter(entity_name)
            if rule_prediction is not None:
                predictions[i] = rule_prediction
            else:
                ml_indices.append(i)
                ml_X.append(entity_name)
        
        # Use ML for ambiguous cases
        if len(ml_X) > 0 and self.is_fitted and self.ml_cases > 0:
            X_vectorized = self.vectorizer.transform(ml_X)
            ml_predictions = self.model.predict(X_vectorized)
            
            for i, idx in enumerate(ml_indices):
                predictions[idx] = ml_predictions[i]
        
        return predictions
    
    def predict_proba(self, X, metadata=None):
        """Predict probabilities"""
        X = np.array(X)
        probabilities = np.zeros((len(X), 2))
        ml_indices = []
        ml_X = []
        
        # Apply rules first
        for i, entity_name in enumerate(X):
            rule_prediction = self._apply_rule_filter(entity_name)
            if rule_prediction is not None:
                # Deterministic rule - give extreme probability
                if rule_prediction == 1:
                    probabilities[i] = [0.0, 1.0]
                else:
                    probabilities[i] = [1.0, 0.0]
            else:
                ml_indices.append(i)
                ml_X.append(entity_name)
        
        # Use ML for ambiguous cases
        if len(ml_X) > 0 and self.is_fitted and self.ml_cases > 0:
            X_vectorized = self.vectorizer.transform(ml_X)
            
            # Some models don't have predict_proba
            if hasattr(self.model, 'predict_proba'):
                ml_probas = self.model.predict_proba(X_vectorized)
            elif hasattr(self.model, 'decision_function'):
                # For SVM, convert decision function to probabilities
                decision = self.model.decision_function(X_vectorized)
                # Simple sigmoid transformation
                ml_probas = np.vstack([1/(1+np.exp(decision)), 1/(1+np.exp(-decision))]).T
            else:
                # Fallback: use hard predictions
                ml_preds = self.model.predict(X_vectorized)
                ml_probas = np.array([[1-p, p] for p in ml_preds])
            
            for i, idx in enumerate(ml_indices):
                probabilities[idx] = ml_probas[i]
        
        return probabilities
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'country_db': self.country_db,
            'model_type': self.model_type,
            'apply_rules': self.apply_rules,
            'is_fitted': self.is_fitted,
            'ml_cases': self.ml_cases
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.country_db = model_data['country_db']
        self.model_type = model_data['model_type']
        self.apply_rules = model_data['apply_rules']
        self.is_fitted = model_data['is_fitted']
        self.ml_cases = model_data.get('ml_cases', 0)
        print(f"✓ Model loaded from {filepath}")


# ============================================================================
# 3. UTILITIES & EVALUATION
# ============================================================================

def generate_sample_data(n_samples: int = 1000):
    """Generate synthetic training data"""
    db = CountryDatabase()
    
    # Get some country names
    countries = list(db.official_names)[:50]
    
    # Country templates
    country_templates = [
        "{country}",
        "Government of {country}",
        "Republic of {country}",
        "Kingdom of {country}",
        "Federal Republic of {country}",
        "Democratic Republic of {country}",
        "Commonwealth of {country}",
    ]
    
    # Non-country templates
    non_country_templates = [
        "{country} Treasury Bond",
        "{country} Government Bond",
        "Bank of {country}",
        "{country} Corporation",
        "{country} Corp",
        "Ministry of Finance {country}",
        "{country} Football Club",
        "{country} Airlines",
        "{country} Stock Exchange",
        "{country} Central Bank",
    ]
    
    data = []
    labels = []
    
    # Generate country examples
    for _ in range(n_samples // 2):
        country = np.random.choice(countries)
        template = np.random.choice(country_templates)
        name = template.format(country=country.title())
        data.append(name)
        labels.append(1)
    
    # Generate non-country examples
    for _ in range(n_samples // 2):
        country = np.random.choice(countries)
        template = np.random.choice(non_country_templates)
        name = template.format(country=country.title())
        data.append(name)
        labels.append(0)
    
    return pd.DataFrame({'entity_name': data, 'label': labels})


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print(f"\n{'='*70}")
    print("MODEL EVALUATION")
    print(f"{'='*70}")
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✓ Overall Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Non-Country', 'Country'],
                                digits=4))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"{'':20} Predicted")
    print(f"{'':20} Non-Country  Country")
    print(f"Actual Non-Country   {cm[0][0]:10d}  {cm[0][1]:7d}")
    print(f"Actual Country       {cm[1][0]:10d}  {cm[1][1]:7d}")
    
    return accuracy, y_pred, y_proba


# ============================================================================
# 4. MAIN EXECUTION
# ============================================================================

def main():
    """Main execution"""
    print(f"\n{'='*70}")
    print("NLP-BASED COUNTRY CLASSIFIER")
    print("Using pycountry database + Text Embeddings")
    print(f"{'='*70}")
    
    # Generate data
    print("\n[1] Generating sample training data...")
    df = generate_sample_data(n_samples=1000)
    print(f"✓ Generated {len(df)} samples")
    print(f"  Countries: {df['label'].sum()}")
    print(f"  Non-countries: {(df['label']==0).sum()}")
    
    # Split data
    print("\n[2] Splitting data (80% train, 20% test)...")
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label']
    )
    
    X_train = train_df['entity_name'].values
    y_train = train_df['label'].values
    X_test = test_df['entity_name'].values
    y_test = test_df['label'].values
    
    # Train model
    print("\n[3] Training NLP Classifier...")
    classifier = NLPCountryClassifier(
        model_type='logistic',  # Try: 'svm', 'mlp', 'random_forest'
        apply_rules=True
    )
    classifier.fit(X_train, y_train)
    
    # Evaluate
    print("\n[4] Evaluating on test set...")
    accuracy, y_pred, y_proba = evaluate_model(classifier, X_test, y_test)
    
    # Test examples
    print(f"\n[5] Testing with specific examples...")
    print(f"{'='*70}")
    
    test_examples = [
        "Government of South Korea",
        "US Treasury Bond",
        "France",
        "South Korea Bond",
        "Republic of France",
        "France Football",
        "Ministry of Finance Japan",
        "Kingdom of Morocco",
        "Bank of England",
        "United States of America"
    ]
    
    predictions = classifier.predict(test_examples)
    probabilities = classifier.predict_proba(test_examples)
    
    print("\nExample Predictions:")
    print(f"{'Entity Name':<40} {'Prediction':<15} {'Confidence'}")
    print(f"{'-'*70}")
    for name, pred, proba in zip(test_examples, predictions, probabilities):
        pred_label = 'Country' if pred == 1 else 'Non-Country'
        confidence = proba[pred] * 100
        print(f"{name:<40} {pred_label:<15} {confidence:>6.2f}%")
    
    # Save model
    print(f"\n[6] Saving model...")
    classifier.save_model('nlp_country_classifier.pkl')
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"\nFinal Test Accuracy: {accuracy:.4f}")
    print(f"Rule Coverage: {classifier.rule_coverage:.1f}%")
    print("Model saved as: nlp_country_classifier.pkl")


if __name__ == "__main__":
    main()
