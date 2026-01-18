"""
Improved RAG System for CloudGarden
===================================

שיפורים עיקריים:
1. TF-IDF scoring במקום ספירת מילים פשוטה
2. BM25 ranking לדירוג מדויק יותר
3. אינטגרציה עם נתוני IoT מ-Firebase
4. Semantic search עם embeddings (אופציונלי)
5. הקשר מורחב ו-prompt engineering משופר
6. Chat history לשיחה רציפה

הוראות שימוש:
- העתק את הקוד הרלוונטי ל-notebook שלך
- או import את המודול הזה
"""

import re
import math
import requests
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import json

# ============================================================================
# 1. FIREBASE CONFIGURATION
# ============================================================================

FIREBASE_URL = "https://cloud-81451-default-rtdb.europe-west1.firebasedatabase.app/"

def firebase_get(path: str) -> Optional[dict]:
    """קריאת נתונים מ-Firebase."""
    base = FIREBASE_URL.rstrip("/")
    url = f"{base}/{path}.json"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"Firebase GET error: {e}")
    return None

def firebase_put(path: str, data: dict) -> bool:
    """שמירת נתונים ל-Firebase."""
    base = FIREBASE_URL.rstrip("/")
    url = f"{base}/{path}.json"
    try:
        resp = requests.put(url, json=data, timeout=10)
        return resp.status_code == 200
    except Exception as e:
        print(f"Firebase PUT error: {e}")
    return False

# ============================================================================
# 2. IMPROVED NLP PREPROCESSING
# ============================================================================

try:
    from nltk.stem import PorterStemmer
    from nltk.corpus import stopwords
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    stemmer = PorterStemmer()
    STOP_WORDS = set(stopwords.words('english'))
except ImportError:
    stemmer = None
    STOP_WORDS = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                  'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                  'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
                  'during', 'before', 'after', 'above', 'below', 'between', 'under',
                  'again', 'further', 'then', 'once', 'here', 'there', 'when',
                  'where', 'why', 'how', 'all', 'each', 'few', 'more', 'most',
                  'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                  'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but',
                  'if', 'or', 'because', 'until', 'while', 'although', 'though'}


def tokenize(text: str) -> List[str]:
    """המרת טקסט לרשימת מילים מנורמלות."""
    if not text:
        return []
    # הסרת תווים מיוחדים ופיצול למילים
    text = text.lower()
    # שמירה על מספרים עם נקודה (למשל: 25.5)
    tokens = re.findall(r'\b\w+(?:\.\d+)?\b', text)
    return tokens


def remove_stopwords(tokens: List[str]) -> List[str]:
    """הסרת מילות עצירה."""
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def apply_stemming(tokens: List[str]) -> List[str]:
    """החלת stemming על המילים."""
    if stemmer:
        return [stemmer.stem(t) for t in tokens]
    return tokens


def preprocess_text(text: str) -> List[str]:
    """עיבוד טקסט מלא: tokenize -> stopwords -> stemming."""
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = apply_stemming(tokens)
    return tokens


# ============================================================================
# 3. TF-IDF INDEXING (שיפור משמעותי!)
# ============================================================================

class TFIDFIndex:
    """
    אינדקס TF-IDF משופר עם תמיכה ב:
    - Term Frequency (TF) מנורמל
    - Inverse Document Frequency (IDF)
    - Document length normalization
    """

    def __init__(self):
        self.inverted_index = defaultdict(dict)  # term -> {doc_id: tf}
        self.doc_lengths = {}  # doc_id -> document length
        self.doc_map = {}  # doc_id -> url/metadata
        self.doc_texts = {}  # doc_id -> original text
        self.idf = {}  # term -> idf score
        self.num_docs = 0
        self.avg_doc_length = 0

    def add_document(self, doc_id: int, text: str, url: str = None, metadata: dict = None):
        """הוספת מסמך לאינדקס."""
        tokens = preprocess_text(text)
        self.doc_lengths[doc_id] = len(tokens)
        self.doc_texts[doc_id] = text
        self.doc_map[doc_id] = {
            'url': url,
            'metadata': metadata or {}
        }

        # חישוב TF (term frequency)
        term_counts = defaultdict(int)
        for token in tokens:
            term_counts[token] += 1

        # נורמליזציה של TF
        max_freq = max(term_counts.values()) if term_counts else 1
        for term, count in term_counts.items():
            # Augmented frequency לנורמליזציה
            tf = 0.5 + 0.5 * (count / max_freq)
            self.inverted_index[term][doc_id] = tf

        self.num_docs += 1

    def compute_idf(self):
        """חישוב IDF לכל המונחים."""
        for term, doc_dict in self.inverted_index.items():
            df = len(doc_dict)  # document frequency
            # IDF עם smoothing
            self.idf[term] = math.log((self.num_docs + 1) / (df + 1)) + 1

        # חישוב אורך מסמך ממוצע
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """
        חיפוש עם ניקוד TF-IDF.
        מחזיר את k המסמכים הכי רלוונטיים.
        """
        query_terms = preprocess_text(query)
        if not query_terms:
            return []

        scores = defaultdict(float)

        for term in query_terms:
            if term not in self.inverted_index:
                continue

            idf = self.idf.get(term, 1.0)

            for doc_id, tf in self.inverted_index[term].items():
                # TF-IDF score
                scores[doc_id] += tf * idf

        # מיון לפי ציון ובחירת top-k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

        results = []
        for doc_id, score in ranked:
            results.append({
                'doc_id': doc_id,
                'score': round(score, 4),
                'url': self.doc_map.get(doc_id, {}).get('url'),
                'metadata': self.doc_map.get(doc_id, {}).get('metadata', {})
            })

        return results

    def to_dict(self) -> dict:
        """המרה ל-dict לשמירה ב-Firebase."""
        return {
            'inverted_index': {k: dict(v) for k, v in self.inverted_index.items()},
            'doc_lengths': self.doc_lengths,
            'doc_map': self.doc_map,
            'idf': self.idf,
            'num_docs': self.num_docs,
            'avg_doc_length': self.avg_doc_length
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TFIDFIndex':
        """יצירת אינדקס מ-dict (טעינה מ-Firebase)."""
        index = cls()
        index.inverted_index = defaultdict(dict, {
            k: {int(doc_id): tf for doc_id, tf in v.items()}
            for k, v in data.get('inverted_index', {}).items()
        })
        index.doc_lengths = {int(k): v for k, v in data.get('doc_lengths', {}).items()}
        index.doc_map = {int(k): v for k, v in data.get('doc_map', {}).items()}
        index.idf = data.get('idf', {})
        index.num_docs = data.get('num_docs', 0)
        index.avg_doc_length = data.get('avg_doc_length', 0)
        return index


# ============================================================================
# 4. BM25 RANKING (אלגוריתם חיפוש מתקדם יותר)
# ============================================================================

class BM25Index:
    """
    אינדקס BM25 - אלגוריתם דירוג מתקדם יותר מ-TF-IDF.
    משמש ב-search engines מודרניים.

    פרמטרים:
    - k1: שולט ברווייה של term frequency (ברירת מחדל: 1.5)
    - b: שולט בנורמליזציה לפי אורך מסמך (ברירת מחדל: 0.75)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.inverted_index = defaultdict(dict)  # term -> {doc_id: term_freq}
        self.doc_lengths = {}
        self.doc_map = {}
        self.doc_texts = {}
        self.idf = {}
        self.num_docs = 0
        self.avg_doc_length = 0

    def add_document(self, doc_id: int, text: str, url: str = None, metadata: dict = None):
        """הוספת מסמך לאינדקס."""
        tokens = preprocess_text(text)
        self.doc_lengths[doc_id] = len(tokens)
        self.doc_texts[doc_id] = text
        self.doc_map[doc_id] = {
            'url': url,
            'metadata': metadata or {}
        }

        # ספירת מופעים
        term_counts = defaultdict(int)
        for token in tokens:
            term_counts[token] += 1

        for term, count in term_counts.items():
            self.inverted_index[term][doc_id] = count

        self.num_docs += 1

    def compute_idf(self):
        """חישוב IDF לפי נוסחת BM25."""
        for term, doc_dict in self.inverted_index.items():
            df = len(doc_dict)
            # BM25 IDF formula
            self.idf[term] = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1)

        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        """חיפוש עם ניקוד BM25."""
        query_terms = preprocess_text(query)
        if not query_terms:
            return []

        scores = defaultdict(float)

        for term in query_terms:
            if term not in self.inverted_index:
                continue

            idf = self.idf.get(term, 0)

            for doc_id, term_freq in self.inverted_index[term].items():
                doc_len = self.doc_lengths.get(doc_id, self.avg_doc_length)

                # BM25 scoring formula
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_length))
                scores[doc_id] += idf * (numerator / denominator)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

        results = []
        for doc_id, score in ranked:
            results.append({
                'doc_id': doc_id,
                'score': round(score, 4),
                'url': self.doc_map.get(doc_id, {}).get('url'),
                'metadata': self.doc_map.get(doc_id, {}).get('metadata', {}),
                'text_preview': self.doc_texts.get(doc_id, '')[:300] + '...'
            })

        return results


# ============================================================================
# 5. FIREBASE IoT DATA INTEGRATION
# ============================================================================

def get_latest_sensor_data() -> Optional[Dict]:
    """
    קבלת נתוני החיישנים האחרונים מ-Firebase.
    מחזיר dict עם temperature, humidity, soil_moisture.
    """
    try:
        data = firebase_get("json")
        if not data:
            return None

        # מציאת הרשומה האחרונה
        if isinstance(data, list):
            # Filter None values and get the last valid entry
            valid_entries = [d for d in data if d is not None]
            if valid_entries:
                latest = valid_entries[-1]
            else:
                return None
        elif isinstance(data, dict):
            # אם זה dict, נמצא את המפתח האחרון
            keys = sorted(data.keys())
            if keys:
                latest = data[keys[-1]]
            else:
                return None
        else:
            return None

        return {
            'temperature': latest.get('temperature'),
            'humidity': latest.get('humidity'),
            'soil_moisture': latest.get('soil'),
            'timestamp': latest.get('created_at') or latest.get('timestamp')
        }
    except Exception as e:
        print(f"Error fetching sensor data: {e}")
        return None


def get_sensor_statistics() -> Optional[Dict]:
    """
    קבלת סטטיסטיקות על נתוני החיישנים.
    """
    try:
        data = firebase_get("json")
        if not data:
            return None

        # המרה לרשימה
        if isinstance(data, dict):
            records = list(data.values())
        else:
            records = [d for d in data if d is not None]

        if not records:
            return None

        # חישוב סטטיסטיקות
        temps = [r.get('temperature') for r in records if r.get('temperature') is not None]
        humids = [r.get('humidity') for r in records if r.get('humidity') is not None]
        soils = [r.get('soil') for r in records if r.get('soil') is not None]

        stats = {}

        if temps:
            stats['temperature'] = {
                'current': temps[-1],
                'avg': round(sum(temps) / len(temps), 2),
                'min': min(temps),
                'max': max(temps)
            }

        if humids:
            stats['humidity'] = {
                'current': humids[-1],
                'avg': round(sum(humids) / len(humids), 2),
                'min': min(humids),
                'max': max(humids)
            }

        if soils:
            stats['soil_moisture'] = {
                'current': soils[-1],
                'avg': round(sum(soils) / len(soils), 2),
                'min': min(soils),
                'max': max(soils)
            }

        stats['record_count'] = len(records)

        return stats
    except Exception as e:
        print(f"Error computing statistics: {e}")
        return None


def analyze_plant_health() -> Dict:
    """
    ניתוח בריאות הצמח על סמך נתוני החיישנים.
    """
    stats = get_sensor_statistics()
    if not stats:
        return {'status': 'unknown', 'message': 'No sensor data available'}

    issues = []
    recommendations = []

    # בדיקת טמפרטורה
    if 'temperature' in stats:
        temp = stats['temperature']['current']
        if temp < 15:
            issues.append("Temperature is too low")
            recommendations.append("Consider moving plant to warmer location or using heating")
        elif temp > 30:
            issues.append("Temperature is too high")
            recommendations.append("Provide shade or improve ventilation")

    # בדיקת לחות אוויר
    if 'humidity' in stats:
        humidity = stats['humidity']['current']
        if humidity < 40:
            issues.append("Air humidity is low")
            recommendations.append("Mist leaves or use a humidifier")
        elif humidity > 80:
            issues.append("Air humidity is very high")
            recommendations.append("Improve ventilation to prevent fungal diseases")

    # בדיקת לחות קרקע
    if 'soil_moisture' in stats:
        soil = stats['soil_moisture']['current']
        if soil < 30:
            issues.append("Soil is dry")
            recommendations.append("Water the plant soon")
        elif soil > 80:
            issues.append("Soil is too wet")
            recommendations.append("Reduce watering, check for proper drainage")

    # קביעת סטטוס כללי
    if not issues:
        status = 'healthy'
        message = "All environmental conditions are within optimal range."
    elif len(issues) <= 1:
        status = 'warning'
        message = f"Minor issue detected: {issues[0]}"
    else:
        status = 'critical'
        message = f"Multiple issues detected: {', '.join(issues)}"

    return {
        'status': status,
        'message': message,
        'issues': issues,
        'recommendations': recommendations,
        'statistics': stats
    }


# ============================================================================
# 6. ENHANCED RAG SYSTEM
# ============================================================================

class EnhancedRAG:
    """
    מערכת RAG משופרת עם:
    - אינטגרציה לנתוני IoT
    - היסטוריית שיחה
    - Prompt engineering משופר
    - Context building חכם
    """

    def __init__(self, index: BM25Index = None):
        self.index = index or BM25Index()
        self.chat_history = []
        self.generator = None
        self._init_generator()

    def _init_generator(self):
        """אתחול מודל ה-generation."""
        try:
            from transformers import pipeline
            import torch
            device = 0 if torch.cuda.is_available() else -1

            # נסה מודל קטן יותר קודם
            try:
                self.generator = pipeline(
                    "text2text-generation",
                    model="google/flan-t5-small",
                    device=device
                )
            except:
                try:
                    self.generator = pipeline(
                        "text2text-generation",
                        model="google/flan-t5-base",
                        device=device
                    )
                except:
                    self.generator = None
        except ImportError:
            self.generator = None

    def _build_context(self, results: List[Dict], max_chars: int = 800) -> str:
        """
        בניית הקשר מהמסמכים שנמצאו.
        """
        context_parts = []
        chars_used = 0

        for i, result in enumerate(results):
            doc_id = result['doc_id']
            text = self.index.doc_texts.get(doc_id, '')

            # חישוב כמה תווים נשארו
            remaining = max_chars - chars_used
            if remaining <= 50:
                break

            # חיתוך טקסט חכם - ניסיון לסיים במשפט
            snippet = text[:remaining]
            last_period = snippet.rfind('.')
            if last_period > remaining * 0.5:
                snippet = snippet[:last_period + 1]

            context_parts.append(f"[Doc {doc_id}]: {snippet}")
            chars_used += len(snippet)

        return "\n\n".join(context_parts)

    def _build_iot_context(self) -> str:
        """בניית הקשר מנתוני IoT."""
        health = analyze_plant_health()
        if health['status'] == 'unknown':
            return ""

        stats = health.get('statistics', {})

        context = "Current Plant Environment:\n"

        if 'temperature' in stats:
            t = stats['temperature']
            context += f"- Temperature: {t['current']}°C (avg: {t['avg']}°C)\n"

        if 'humidity' in stats:
            h = stats['humidity']
            context += f"- Humidity: {h['current']}% (avg: {h['avg']}%)\n"

        if 'soil_moisture' in stats:
            s = stats['soil_moisture']
            context += f"- Soil Moisture: {s['current']}% (avg: {s['avg']}%)\n"

        context += f"\nHealth Status: {health['status'].upper()}\n"

        if health['issues']:
            context += f"Issues: {', '.join(health['issues'])}\n"

        if health['recommendations']:
            context += f"Recommendations: {', '.join(health['recommendations'])}\n"

        return context

    def query(self, question: str, k: int = 3, include_iot: bool = True) -> Dict:
        """
        שאילתה למערכת RAG.

        Args:
            question: השאלה של המשתמש
            k: מספר המסמכים להחזיר
            include_iot: האם לכלול נתוני IoT בהקשר

        Returns:
            dict עם התשובה, המסמכים, והמידע הנוסף
        """
        # חיפוש מסמכים רלוונטיים
        results = self.index.search(question, k=k)

        # בניית הקשר
        doc_context = self._build_context(results, max_chars=600)

        # הוספת נתוני IoT אם רלוונטי
        iot_context = ""
        if include_iot and self._is_plant_related(question):
            iot_context = self._build_iot_context()

        # בניית ה-prompt
        prompt = self._build_prompt(question, doc_context, iot_context)

        # יצירת תשובה
        if self.generator:
            try:
                output = self.generator(
                    prompt,
                    max_new_tokens=200,
                    do_sample=False,
                    num_beams=1
                )[0]['generated_text']
                answer = output.strip()
            except Exception as e:
                answer = self._fallback_answer(question, results)
        else:
            answer = self._fallback_answer(question, results)

        # הוספה להיסטוריה
        self.chat_history.append({
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        })

        return {
            'answer': answer,
            'results': results,
            'query_terms': preprocess_text(question),
            'iot_data': iot_context if iot_context else None
        }

    def _is_plant_related(self, question: str) -> bool:
        """בדיקה האם השאלה קשורה לצמחים/סביבה."""
        plant_keywords = [
            'plant', 'leaf', 'disease', 'water', 'temperature', 'humidity',
            'soil', 'health', 'grow', 'moisture', 'condition', 'environment',
            'צמח', 'עלה', 'מחלה', 'השקיה', 'טמפרטורה', 'לחות', 'קרקע', 'בריאות'
        ]
        question_lower = question.lower()
        return any(kw in question_lower for kw in plant_keywords)

    def _build_prompt(self, question: str, doc_context: str, iot_context: str) -> str:
        """בניית prompt משופר."""
        parts = [
            "You are a helpful assistant specializing in plant health and disease detection.",
            "Answer the question based ONLY on the provided context.",
            "If the context doesn't contain enough information, say so clearly.",
            "Be concise and cite your sources using [Doc X] format.",
            ""
        ]

        if iot_context:
            parts.append("=== REAL-TIME SENSOR DATA ===")
            parts.append(iot_context)
            parts.append("")

        if doc_context:
            parts.append("=== DOCUMENT CONTEXT ===")
            parts.append(doc_context)
            parts.append("")

        parts.append(f"Question: {question}")
        parts.append("")
        parts.append("Answer:")

        return "\n".join(parts)

    def _fallback_answer(self, question: str, results: List[Dict]) -> str:
        """תשובת גיבוי כאשר המודל לא זמין."""
        if not results:
            return "I couldn't find any relevant documents for your question."

        answer_parts = ["Based on the retrieved documents:\n"]

        for i, r in enumerate(results[:3]):
            preview = r.get('text_preview', '')[:150]
            answer_parts.append(f"[Doc {r['doc_id']}] {preview}...")

        return "\n".join(answer_parts)

    def clear_history(self):
        """ניקוי היסטוריית השיחה."""
        self.chat_history = []


# ============================================================================
# 7. GRADIO UI INTEGRATION
# ============================================================================

def create_rag_chat_interface(rag: EnhancedRAG):
    """
    יצירת ממשק Gradio לצ'אט RAG.
    העתק קוד זה לתוך ה-notebook שלך.
    """
    try:
        import gradio as gr
    except ImportError:
        print("Gradio not installed. Run: pip install gradio")
        return None

    def chat_fn(message: str, history: list, k: int, include_iot: bool):
        if not message.strip():
            return history, "", []

        response = rag.query(message, k=int(k), include_iot=include_iot)

        # עדכון היסטוריה
        history.append((message, response['answer']))

        # טבלת תוצאות
        results_table = [
            [r['doc_id'], r['score'], r.get('url', 'N/A')]
            for r in response['results']
        ]

        return history, "", results_table

    with gr.Blocks() as interface:
        gr.Markdown("## 💬 Enhanced RAG Chat")
        gr.Markdown("Chat with plant disease documents + real-time sensor data")

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Chat History", height=400)
                msg = gr.Textbox(label="Your Question", placeholder="Ask about plant diseases...")

                with gr.Row():
                    k_slider = gr.Slider(1, 10, value=3, step=1, label="Documents to retrieve")
                    iot_checkbox = gr.Checkbox(value=True, label="Include IoT sensor data")

                submit_btn = gr.Button("Send", variant="primary")

            with gr.Column(scale=1):
                results_table = gr.Dataframe(
                    headers=["Doc ID", "Score", "URL"],
                    label="Retrieved Documents"
                )

        submit_btn.click(
            chat_fn,
            inputs=[msg, chatbot, k_slider, iot_checkbox],
            outputs=[chatbot, msg, results_table]
        )

        msg.submit(
            chat_fn,
            inputs=[msg, chatbot, k_slider, iot_checkbox],
            outputs=[chatbot, msg, results_table]
        )

    return interface


# ============================================================================
# 8. USAGE EXAMPLE
# ============================================================================

def demo():
    """הדגמת שימוש במערכת."""
    print("=== Enhanced RAG System Demo ===\n")

    # יצירת אינדקס BM25
    index = BM25Index()

    # הוספת מסמכים לדוגמה
    docs = [
        {
            'text': """Plant leaf diseases can cause significant crop losses.
            Common symptoms include yellow spots, brown patches, and wilting.
            Early detection using computer vision and deep learning can help farmers
            identify diseases before they spread.""",
            'url': 'https://example.com/doc1'
        },
        {
            'text': """Temperature and humidity play crucial roles in plant health.
            Most plants thrive between 18-28°C with 40-60% humidity.
            Excessive moisture can lead to fungal diseases, while dry conditions
            may cause water stress.""",
            'url': 'https://example.com/doc2'
        },
        {
            'text': """Soil moisture monitoring is essential for proper irrigation.
            Overwatering can lead to root rot, while underwatering causes wilting.
            IoT sensors can provide real-time monitoring for optimal plant care.""",
            'url': 'https://example.com/doc3'
        }
    ]

    for i, doc in enumerate(docs):
        index.add_document(i, doc['text'], doc['url'])

    index.compute_idf()

    # יצירת מערכת RAG
    rag = EnhancedRAG(index)

    # שאילתות לדוגמה
    queries = [
        "What are common symptoms of plant diseases?",
        "How does temperature affect plant health?",
        "What is the optimal soil moisture level?"
    ]

    for q in queries:
        print(f"Q: {q}")
        response = rag.query(q, k=2)
        print(f"A: {response['answer']}\n")
        print(f"Sources: {[r['url'] for r in response['results']]}\n")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    demo()
