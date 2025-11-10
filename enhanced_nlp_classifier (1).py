"""
Enhanced NLP Country Classifier with Custom Data Integration
=============================================================
Combines:
1. pycountry database (250+ countries)
2. Your internal country table
3. Excel file with country variants
4. Automatic duplicate detection and merging

Python 3.6 compatible
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
import pickle
import warnings
warnings.filterwarnings('ignore')

# Try to import pycountry
try:
    import pycountry
    PYCOUNTRY_AVAILABLE = True
except ImportError:
    PYCOUNTRY_AVAILABLE = False
    print("Warning: pycountry not installed. Using custom data only.")

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

try:
    from sklearn.neural_network import MLPClassifier
    MLP_AVAILABLE = True
except ImportError:
    MLP_AVAILABLE = False


# ============================================================================
# 1. ENHANCED COUNTRY DATABASE WITH CUSTOM DATA
# ============================================================================

class EnhancedCountryDatabase:
    """
    Enhanced country database that merges:
    1. pycountry database
    2. Internal/custom country table
    3. Excel file with country variants
    
    Handles duplicates automatically.
    """
    
    def __init__(self, 
                 use_pycountry: bool = True,
                 custom_table: Optional[pd.DataFrame] = None,
                 excel_path: Optional[str] = None,
                 excel_country_col: str = 'country',
                 excel_variant_cols: Optional[List[str]] = None):
        """
        Initialize enhanced country database
        
        Parameters:
        -----------
        use_pycountry : bool
            Use pycountry database
        custom_table : DataFrame
            Your internal country table with columns like 'country_name', 'variants', etc.
        excel_path : str
            Path to Excel file with country variants
        excel_country_col : str
            Column name for country in Excel
        excel_variant_cols : list
            List of column names containing variants in Excel
        """
        self.use_pycountry = use_pycountry and PYCOUNTRY_AVAILABLE
        
        # Initialize storage
        self.country_names: Set[str] = set()
        self.official_names: Set[str] = set()
        self.alpha2_codes: Set[str] = set()
        self.alpha3_codes: Set[str] = set()
        self.country_variants: Dict[str, List[str]] = {}
        self.country_to_canonical: Dict[str, str] = {}  # Map variants to canonical name
        
        # Statistics
        self.stats = {
            'pycountry': 0,
            'custom_table': 0,
            'excel': 0,
            'total_unique': 0,
            'duplicates_merged': 0
        }
        
        # Load data from all sources
        self._initialize_database(custom_table, excel_path, excel_country_col, excel_variant_cols)
        self._add_government_patterns()
        self._add_non_country_patterns()
        
        # Print statistics
        self._print_stats()
    
    def _normalize_name(self, name: str) -> str:
        """Normalize country name for duplicate detection"""
        if not name:
            return ""
        return re.sub(r'\s+', ' ', name.lower().strip())
    
    def _initialize_database(self, custom_table, excel_path, excel_country_col, excel_variant_cols):
        """Load and merge data from all sources"""
        print("="*70)
        print("LOADING COUNTRY DATA FROM MULTIPLE SOURCES")
        print("="*70)
        
        # 1. Load from pycountry
        if self.use_pycountry:
            self._load_from_pycountry()
        
        # 2. Load from custom table
        if custom_table is not None:
            self._load_from_custom_table(custom_table)
        
        # 3. Load from Excel
        if excel_path is not None:
            self._load_from_excel(excel_path, excel_country_col, excel_variant_cols)
        
        # 4. Update statistics
        self.stats['total_unique'] = len(self.country_names)
    
    def _load_from_pycountry(self):
        """Load countries from pycountry"""
        print("\n[1] Loading from pycountry database...")
        
        count = 0
        for country in pycountry.countries:
            # Common name
            name = self._normalize_name(country.name)
            if name:
                self.country_names.add(name)
                self.official_names.add(name)
                self.country_to_canonical[name] = name
                count += 1
            
            # Official name
            if hasattr(country, 'official_name'):
                official = self._normalize_name(country.official_name)
                if official and official != name:
                    self.country_names.add(official)
                    self.official_names.add(official)
                    self.country_to_canonical[official] = name
                    count += 1
            
            # Alpha codes
            alpha2 = country.alpha_2.lower()
            alpha3 = country.alpha_3.lower()
            self.alpha2_codes.add(alpha2)
            self.alpha3_codes.add(alpha3)
            self.country_names.add(alpha2)
            self.country_names.add(alpha3)
            self.country_to_canonical[alpha2] = name
            self.country_to_canonical[alpha3] = name
            count += 3
            
            # Add common variants
            self._add_pycountry_variants(name)
        
        self.stats['pycountry'] = count
        print(f"✓ Loaded {count} entries from pycountry")
    
    def _add_pycountry_variants(self, country_name: str):
        """Add common variants for pycountry countries"""
        variant_map = {
            'united states': ['usa', 'us', 'america', 'united states of america'],
            'united kingdom': ['uk', 'great britain', 'britain', 'england'],
            'united arab emirates': ['uae', 'emirates'],
            'korea, republic of': ['south korea', 'republic of korea', 'rok'],
            'korea, democratic people\'s republic of': ['north korea', 'dprk'],
            'czech republic': ['czechia', 'czech'],
            'myanmar': ['burma'],
            'china': ['prc', "people's republic of china"],
            'russia': ['russian federation'],
            'congo, the democratic republic of the': ['drc', 'democratic republic of congo'],
            'netherlands': ['holland'],
        }
        
        for base_name, variants in variant_map.items():
            if base_name in country_name or country_name in base_name:
                for variant in variants:
                    variant_norm = self._normalize_name(variant)
                    self.country_names.add(variant_norm)
                    self.country_to_canonical[variant_norm] = country_name
                    
                    if country_name not in self.country_variants:
                        self.country_variants[country_name] = []
                    if variant_norm not in self.country_variants[country_name]:
                        self.country_variants[country_name].append(variant_norm)
    
    def _load_from_custom_table(self, custom_table: pd.DataFrame):
        """Load countries from your internal table"""
        print("\n[2] Loading from custom internal table...")
        
        # Try to detect country column
        possible_country_cols = ['country', 'country_name', 'name', 'entity', 'official_name']
        country_col = None
        
        for col in possible_country_cols:
            if col in custom_table.columns:
                country_col = col
                break
        
        if country_col is None:
            print(f"⚠ No country column found. Tried: {possible_country_cols}")
            print(f"Available columns: {list(custom_table.columns)}")
            return
        
        # Try to detect variant columns
        possible_variant_cols = ['variants', 'variant', 'alternative_names', 'aliases', 'other_names']
        variant_cols = [col for col in possible_variant_cols if col in custom_table.columns]
        
        count = 0
        duplicates = 0
        
        for idx, row in custom_table.iterrows():
            # Get main country name
            country_name = row[country_col]
            if pd.isna(country_name):
                continue
            
            country_norm = self._normalize_name(str(country_name))
            if not country_norm:
                continue
            
            # Check if duplicate
            if country_norm in self.country_names:
                duplicates += 1
            else:
                self.country_names.add(country_norm)
                self.official_names.add(country_norm)
                self.country_to_canonical[country_norm] = country_norm
                count += 1
            
            # Get variants
            for variant_col in variant_cols:
                if variant_col in row and not pd.isna(row[variant_col]):
                    variants_str = str(row[variant_col])
                    
                    # Split by common delimiters
                    variants = re.split(r'[,;|]', variants_str)
                    
                    for variant in variants:
                        variant_norm = self._normalize_name(variant)
                        if variant_norm and variant_norm != country_norm:
                            if variant_norm not in self.country_names:
                                self.country_names.add(variant_norm)
                                count += 1
                            
                            self.country_to_canonical[variant_norm] = country_norm
                            
                            if country_norm not in self.country_variants:
                                self.country_variants[country_norm] = []
                            if variant_norm not in self.country_variants[country_norm]:
                                self.country_variants[country_norm].append(variant_norm)
        
        self.stats['custom_table'] = count
        self.stats['duplicates_merged'] += duplicates
        print(f"✓ Loaded {count} unique entries from custom table")
        print(f"  (Merged {duplicates} duplicates with existing data)")
    
    def _load_from_excel(self, excel_path: str, country_col: str, variant_cols: Optional[List[str]]):
        """Load countries from Excel file"""
        print(f"\n[3] Loading from Excel file: {excel_path}")
        
        try:
            # Try different engines
            try:
                df = pd.read_excel(excel_path, engine='openpyxl')
            except:
                df = pd.read_excel(excel_path)
            
            print(f"✓ Excel file loaded: {len(df)} rows")
            print(f"  Columns: {list(df.columns)}")
            
        except Exception as e:
            print(f"⚠ Could not load Excel file: {e}")
            return
        
        # Check if country column exists
        if country_col not in df.columns:
            print(f"⚠ Country column '{country_col}' not found")
            print(f"  Available columns: {list(df.columns)}")
            return
        
        # If variant columns not specified, detect them
        if variant_cols is None:
            variant_cols = [col for col in df.columns if col != country_col 
                           and 'variant' in col.lower() or 'alias' in col.lower() 
                           or 'alternative' in col.lower() or 'other' in col.lower()]
        
        count = 0
        duplicates = 0
        
        for idx, row in df.iterrows():
            # Get main country name
            country_name = row[country_col]
            if pd.isna(country_name):
                continue
            
            country_norm = self._normalize_name(str(country_name))
            if not country_norm:
                continue
            
            # Check if duplicate
            if country_norm in self.country_names:
                duplicates += 1
            else:
                self.country_names.add(country_norm)
                self.official_names.add(country_norm)
                self.country_to_canonical[country_norm] = country_norm
                count += 1
            
            # Get variants from specified columns
            for variant_col in variant_cols:
                if variant_col in row and not pd.isna(row[variant_col]):
                    variant_value = str(row[variant_col])
                    
                    # Handle multiple variants in one cell (comma/semicolon separated)
                    variants = re.split(r'[,;|]', variant_value)
                    
                    for variant in variants:
                        variant_norm = self._normalize_name(variant)
                        if variant_norm and variant_norm != country_norm:
                            if variant_norm not in self.country_names:
                                self.country_names.add(variant_norm)
                                count += 1
                            
                            self.country_to_canonical[variant_norm] = country_norm
                            
                            if country_norm not in self.country_variants:
                                self.country_variants[country_norm] = []
                            if variant_norm not in self.country_variants[country_norm]:
                                self.country_variants[country_norm].append(variant_norm)
        
        self.stats['excel'] = count
        self.stats['duplicates_merged'] += duplicates
        print(f"✓ Loaded {count} unique entries from Excel")
        print(f"  (Merged {duplicates} duplicates with existing data)")
    
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
    
    def _print_stats(self):
        """Print loading statistics"""
        print("\n" + "="*70)
        print("DATABASE STATISTICS")
        print("="*70)
        print(f"Sources loaded:")
        print(f"  • pycountry:        {self.stats['pycountry']:>6} entries")
        print(f"  • Custom table:     {self.stats['custom_table']:>6} entries")
        print(f"  • Excel file:       {self.stats['excel']:>6} entries")
        print(f"  ─────────────────────────────")
        print(f"  • Total unique:     {self.stats['total_unique']:>6} entries")
        print(f"  • Duplicates merged:{self.stats['duplicates_merged']:>6}")
        print("="*70)
    
    def contains_country(self, text: str) -> Tuple[bool, List[str]]:
        """Check if text contains any country name"""
        text_lower = self._normalize_name(text)
        found_countries = []
        
        # Check all country names
        for country in self.country_names:
            pattern = r'\b' + re.escape(country) + r'\b'
            if re.search(pattern, text_lower):
                # Map to canonical name if possible
                canonical = self.country_to_canonical.get(country, country)
                if canonical not in found_countries:
                    found_countries.append(canonical)
        
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
        text_lower = self._normalize_name(text)
        return text_lower in self.country_names or text_lower in self.official_names
    
    def export_merged_database(self, output_path: str):
        """Export the merged country database to Excel"""
        print(f"\nExporting merged database to: {output_path}")
        
        data = []
        for country in sorted(self.official_names):
            variants = self.country_variants.get(country, [])
            data.append({
                'country': country,
                'variants': ', '.join(variants) if variants else '',
                'variant_count': len(variants)
            })
        
        df = pd.DataFrame(data)
        df.to_excel(output_path, index=False)
        print(f"✓ Exported {len(df)} countries to {output_path}")


# ============================================================================
# 2. ENHANCED NLP CLASSIFIER
# ============================================================================

class EnhancedNLPClassifier:
    """
    Enhanced NLP classifier with custom country data integration
    """
    
    def __init__(self,
                 model_type: str = 'logistic',
                 use_pycountry: bool = True,
                 custom_table: Optional[pd.DataFrame] = None,
                 excel_path: Optional[str] = None,
                 excel_country_col: str = 'country',
                 excel_variant_cols: Optional[List[str]] = None,
                 apply_rules: bool = True,
                 ngram_range: Tuple[int, int] = (1, 3),
                 max_features: int = 5000):
        """
        Initialize enhanced NLP classifier
        
        Parameters:
        -----------
        model_type : str
            'logistic', 'svm', 'naive_bayes', 'mlp', 'random_forest'
        use_pycountry : bool
            Use pycountry database
        custom_table : DataFrame
            Your internal country table
        excel_path : str
            Path to Excel file with country variants
        excel_country_col : str
            Column name for country in Excel
        excel_variant_cols : list
            List of variant column names in Excel
        apply_rules : bool
            Apply rule-based pre-filtering
        ngram_range : tuple
            N-gram range for TF-IDF
        max_features : int
            Maximum vocabulary size
        """
        self.model_type = model_type
        self.apply_rules = apply_rules
        
        # Initialize enhanced country database
        self.country_db = EnhancedCountryDatabase(
            use_pycountry=use_pycountry,
            custom_table=custom_table,
            excel_path=excel_path,
            excel_country_col=excel_country_col,
            excel_variant_cols=excel_variant_cols
        )
        
        self.is_fitted = False
        
        # Text vectorizer
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            lowercase=True,
            strip_accents='unicode',
            token_pattern=r'\b\w+\b',
            min_df=1,
            sublinear_tf=True
        )
        
        # Initialize model
        self.model = self._create_model(model_type)
        
        # Statistics
        self.rule_coverage = 0
        self.ml_cases = 0
    
    def _create_model(self, model_type: str):
        """Create ML model"""
        if model_type == 'logistic':
            return LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight='balanced')
        elif model_type == 'svm':
            return LinearSVC(C=1.0, max_iter=2000, random_state=42, class_weight='balanced')
        elif model_type == 'naive_bayes':
            return MultinomialNB(alpha=0.1)
        elif model_type == 'mlp':
            if MLP_AVAILABLE:
                return MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', 
                                    max_iter=500, random_state=42, early_stopping=True)
            else:
                print("MLP not available, using Logistic Regression")
                return LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        elif model_type == 'random_forest':
            return RandomForestClassifier(n_estimators=200, max_depth=20, 
                                         min_samples_split=5, random_state=42, class_weight='balanced')
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    def _apply_rule_filter(self, text: str) -> Optional[int]:
        """Apply deterministic rules"""
        if not self.apply_rules:
            return None
        
        # Rule 1: No country name → Non-country
        has_country, _ = self.country_db.contains_country(text)
        if not has_country:
            return 0
        
        # Rule 2: Strong non-country indicators
        if self.country_db.has_non_country_pattern(text):
            return 0
        
        # Rule 3: Exact country match
        if self.country_db.is_exact_match(text):
            return 1
        
        # Rule 4: Government pattern + country name
        if self.country_db.has_government_pattern(text):
            if not self.country_db.has_non_country_pattern(text):
                return 1
        
        return None
    
    def fit(self, X, y):
        """Train the classifier"""
        X = np.array(X)
        y = np.array(y)
        
        print(f"\n{'='*70}")
        print(f"TRAINING ENHANCED NLP CLASSIFIER")
        print(f"Model: {self.model_type.upper()}")
        print(f"{'='*70}")
        
        # Separate rule vs ML cases
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
        
        # Vectorize and train
        print(f"\nVectorizing text with TF-IDF...")
        X_vectorized = self.vectorizer.fit_transform(ml_X)
        print(f"  Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        print(f"  Feature dimensions: {X_vectorized.shape}")
        
        print(f"\nTraining {self.model_type} model...")
        self.model.fit(X_vectorized, ml_y)
        
        train_pred = self.model.predict(X_vectorized)
        train_acc = accuracy_score(ml_y, train_pred)
        print(f"  ML model training accuracy: {train_acc:.3f}")
        
        self.is_fitted = True
        self.ml_cases = len(ml_X)
        print(f"\n✓ Training complete!")
        
        return self
    
    def predict(self, X):
        """Predict labels"""
        X = np.array(X)
        predictions = np.zeros(len(X), dtype=int)
        ml_indices = []
        ml_X = []
        
        # Apply rules
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
    
    def predict_proba(self, X):
        """Predict probabilities"""
        X = np.array(X)
        probabilities = np.zeros((len(X), 2))
        ml_indices = []
        ml_X = []
        
        # Apply rules
        for i, entity_name in enumerate(X):
            rule_prediction = self._apply_rule_filter(entity_name)
            if rule_prediction is not None:
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
            
            if hasattr(self.model, 'predict_proba'):
                ml_probas = self.model.predict_proba(X_vectorized)
            elif hasattr(self.model, 'decision_function'):
                decision = self.model.decision_function(X_vectorized)
                ml_probas = np.vstack([1/(1+np.exp(decision)), 1/(1+np.exp(-decision))]).T
            else:
                ml_preds = self.model.predict(X_vectorized)
                ml_probas = np.array([[1-p, p] for p in ml_preds])
            
            for i, idx in enumerate(ml_indices):
                probabilities[idx] = ml_probas[i]
        
        return probabilities
    
    def save_model(self, filepath: str):
        """Save model"""
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
        """Load model"""
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
    
    def export_country_database(self, output_path: str):
        """Export the merged country database"""
        self.country_db.export_merged_database(output_path)


# ============================================================================
# 3. EXAMPLE USAGE WITH CUSTOM DATA
# ============================================================================

def example_with_custom_data():
    """Example showing how to use custom data"""
    print("\n" + "="*70)
    print("EXAMPLE: Using Custom Country Data")
    print("="*70)
    
    # Example 1: Create custom internal table
    print("\n[1] Creating sample custom internal table...")
    custom_countries = pd.DataFrame({
        'country_name': [
            'United States of America',
            'United Kingdom',
            'Peoples Republic of China',
            'Kingdom of Saudi Arabia',
            'Swiss Confederation'
        ],
        'variants': [
            'USA, US, America, United States',
            'UK, Great Britain, Britain',
            'China, PRC',
            'Saudi Arabia, KSA',
            'Switzerland'
        ],
        'iso_code': ['US', 'GB', 'CN', 'SA', 'CH']
    })
    print(custom_countries)
    
    # Example 2: Create Excel with variants
    print("\n[2] Creating sample Excel file...")
    excel_data = pd.DataFrame({
        'country': ['France', 'Germany', 'Japan', 'India', 'Brazil'],
        'variant1': ['French Republic', 'Federal Republic of Germany', 'Nippon', 'Bharat', 'Federative Republic of Brazil'],
        'variant2': ['FR', 'DE', 'JP', 'IN', 'BR'],
        'variant3': ['', 'Deutschland', '', 'Republic of India', '']
    })
    excel_path = '/home/claude/sample_country_variants.xlsx'
    excel_data.to_excel(excel_path, index=False)
    print(f"✓ Excel file created: {excel_path}")
    
    # Example 3: Initialize classifier with all data sources
    print("\n[3] Initializing classifier with multiple data sources...")
    classifier = EnhancedNLPClassifier(
        model_type='logistic',
        use_pycountry=True,
        custom_table=custom_countries,
        excel_path=excel_path,
        excel_country_col='country',
        excel_variant_cols=['variant1', 'variant2', 'variant3']
    )
    
    # Example 4: Train
    print("\n[4] Training classifier...")
    train_entities = [
        "United States", "France", "Japan",
        "US Treasury Bond", "Bank of France", "Japanese Corporation"
    ]
    labels = [1, 1, 1, 0, 0, 0]
    
    classifier.fit(train_entities, labels)
    
    # Example 5: Test
    print("\n[5] Testing predictions...")
    test_entities = [
        "United States of America",
        "USA",
        "America",
        "US Bond",
        "French Republic",
        "FR",
        "France Football",
        "Federal Republic of Germany",
        "Deutschland",
        "German Bank"
    ]
    
    predictions = classifier.predict(test_entities)
    probabilities = classifier.predict_proba(test_entities)
    
    print(f"\n{'Entity':<40} {'Predicted':<15} {'Confidence'}")
    print("-"*70)
    for entity, pred, proba in zip(test_entities, predictions, probabilities):
        label = 'Country' if pred == 1 else 'Non-Country'
        conf = proba[pred] * 100
        print(f"{entity:<40} {label:<15} {conf:>6.2f}%")
    
    # Example 6: Export merged database
    print("\n[6] Exporting merged country database...")
    classifier.export_country_database('/home/claude/merged_countries.xlsx')


if __name__ == "__main__":
    example_with_custom_data()
