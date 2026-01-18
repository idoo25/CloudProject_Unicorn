# מדריך אינטגרציה - חיבור הצ'אט לדאטהבייס ושיפור RAG

## סקירת הבעיות הנוכחיות

### בעיות באינדקס הקיים:
1. **חיפוש מבוסס מילים בלבד** - רק סופר כמה מילים מתאימות
2. **אין TF-IDF** - לא נותן משקל למילים נדירות/חשובות
3. **הקשר קצר מדי** - רק 160 תווים לכל מסמך
4. **אין חיבור לנתוני IoT** - הצ'אט לא יודע מה מצב הצמח

### בעיות ב-RAG:
1. **Prompt פשוט מדי** - לא מספיק הנחיות למודל
2. **אין היסטוריית שיחה** - כל שאלה נפרדת
3. **אין נתוני סביבה** - לא משתמש בחיישנים

---

## פתרון 1: שיפור האינדקס עם TF-IDF

### הסבר TF-IDF:
- **TF (Term Frequency)** - כמה פעמים מילה מופיעה במסמך
- **IDF (Inverse Document Frequency)** - כמה נדירה המילה בכל המסמכים
- מילה שמופיעה הרבה במסמך אחד אבל מעט במסמכים אחרים = חשובה!

### קוד להחלפה ב-notebook:

```python
# =========================
# 7. Index Construction - IMPROVED TF-IDF
# =========================
import math
from collections import defaultdict

class TFIDFIndex:
    def __init__(self):
        self.inverted_index = defaultdict(dict)  # term -> {doc_id: tf}
        self.doc_lengths = {}
        self.doc_map = {}
        self.doc_texts = {}
        self.idf = {}
        self.num_docs = 0
        self.avg_doc_length = 0

    def add_document(self, doc_id, text, url=None):
        tokens = preprocess_query(text)  # שימוש בפונקציה הקיימת
        self.doc_lengths[doc_id] = len(tokens)
        self.doc_texts[doc_id] = text
        self.doc_map[doc_id] = url

        # חישוב TF
        term_counts = defaultdict(int)
        for token in tokens:
            term_counts[token] += 1

        max_freq = max(term_counts.values()) if term_counts else 1
        for term, count in term_counts.items():
            tf = 0.5 + 0.5 * (count / max_freq)  # Augmented TF
            self.inverted_index[term][doc_id] = tf

        self.num_docs += 1

    def compute_idf(self):
        for term, doc_dict in self.inverted_index.items():
            df = len(doc_dict)
            self.idf[term] = math.log((self.num_docs + 1) / (df + 1)) + 1

        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def search(self, query, k=5):
        query_terms = preprocess_query(query)
        if not query_terms:
            return [], []

        scores = defaultdict(float)
        for term in query_terms:
            if term not in self.inverted_index:
                continue
            idf = self.idf.get(term, 1.0)
            for doc_id, tf in self.inverted_index[term].items():
                scores[doc_id] += tf * idf

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

        results = []
        for doc_id, score in ranked:
            results.append({
                'doc_id': doc_id,
                'score': round(score, 4),
                'url': self.doc_map.get(doc_id)
            })

        return query_terms, results

# יצירת האינדקס החדש
tfidf_index = TFIDFIndex()
```

---

## פתרון 2: חיבור לנתוני IoT מ-Firebase

### הקוד להוספה:

```python
# =========================
# Firebase IoT Integration for Chat
# =========================

def get_latest_sensor_data():
    """קבלת נתוני החיישנים האחרונים."""
    try:
        data = firebase_get("json")
        if not data:
            return None

        if isinstance(data, list):
            valid = [d for d in data if d is not None]
            if valid:
                latest = valid[-1]
            else:
                return None
        elif isinstance(data, dict):
            keys = sorted(data.keys())
            latest = data[keys[-1]] if keys else None
        else:
            return None

        return {
            'temperature': latest.get('temperature'),
            'humidity': latest.get('humidity'),
            'soil_moisture': latest.get('soil'),
            'timestamp': latest.get('created_at')
        }
    except Exception as e:
        print(f"Error: {e}")
        return None


def get_sensor_statistics():
    """סטטיסטיקות על נתוני החיישנים."""
    try:
        data = firebase_get("json")
        if not data:
            return None

        if isinstance(data, dict):
            records = list(data.values())
        else:
            records = [d for d in data if d is not None]

        temps = [r.get('temperature') for r in records if r.get('temperature')]
        humids = [r.get('humidity') for r in records if r.get('humidity')]
        soils = [r.get('soil') for r in records if r.get('soil')]

        stats = {}
        if temps:
            stats['temperature'] = {
                'current': temps[-1],
                'avg': round(sum(temps) / len(temps), 1),
                'min': min(temps),
                'max': max(temps)
            }
        if humids:
            stats['humidity'] = {
                'current': humids[-1],
                'avg': round(sum(humids) / len(humids), 1),
                'min': min(humids),
                'max': max(humids)
            }
        if soils:
            stats['soil_moisture'] = {
                'current': soils[-1],
                'avg': round(sum(soils) / len(soils), 1),
                'min': min(soils),
                'max': max(soils)
            }

        return stats
    except:
        return None


def analyze_plant_health():
    """ניתוח בריאות הצמח."""
    stats = get_sensor_statistics()
    if not stats:
        return {'status': 'unknown', 'message': 'No data'}

    issues = []
    recommendations = []

    # בדיקת טמפרטורה
    if 'temperature' in stats:
        temp = stats['temperature']['current']
        if temp < 15:
            issues.append("Temperature too low")
            recommendations.append("Move to warmer location")
        elif temp > 30:
            issues.append("Temperature too high")
            recommendations.append("Provide shade")

    # בדיקת לחות
    if 'humidity' in stats:
        hum = stats['humidity']['current']
        if hum < 40:
            issues.append("Low humidity")
            recommendations.append("Mist leaves")
        elif hum > 80:
            issues.append("High humidity - fungus risk")
            recommendations.append("Improve ventilation")

    # בדיקת קרקע
    if 'soil_moisture' in stats:
        soil = stats['soil_moisture']['current']
        if soil < 30:
            issues.append("Soil is dry")
            recommendations.append("Water the plant")
        elif soil > 80:
            issues.append("Soil is too wet")
            recommendations.append("Reduce watering")

    status = 'healthy' if not issues else ('warning' if len(issues) <= 1 else 'critical')

    return {
        'status': status,
        'issues': issues,
        'recommendations': recommendations,
        'statistics': stats
    }
```

---

## פתרון 3: RAG משופר עם נתוני IoT

### הקוד המעודכן:

```python
# =========================
# 11. Enhanced RAG with IoT
# =========================

def build_iot_context():
    """בניית הקשר מנתוני IoT."""
    health = analyze_plant_health()
    if health['status'] == 'unknown':
        return ""

    stats = health.get('statistics', {})
    lines = ["Current Plant Environment:"]

    if 'temperature' in stats:
        t = stats['temperature']
        lines.append(f"- Temperature: {t['current']}C (avg: {t['avg']}C)")

    if 'humidity' in stats:
        h = stats['humidity']
        lines.append(f"- Humidity: {h['current']}% (avg: {h['avg']}%)")

    if 'soil_moisture' in stats:
        s = stats['soil_moisture']
        lines.append(f"- Soil: {s['current']}% (avg: {s['avg']}%)")

    lines.append(f"\nStatus: {health['status'].upper()}")

    if health['issues']:
        lines.append(f"Issues: {', '.join(health['issues'])}")
    if health['recommendations']:
        lines.append(f"Actions: {', '.join(health['recommendations'])}")

    return "\n".join(lines)


def is_plant_related_query(query):
    """בדיקה האם השאלה קשורה לצמחים."""
    keywords = ['plant', 'leaf', 'disease', 'water', 'temperature',
                'humidity', 'soil', 'health', 'grow', 'moisture']
    return any(kw in query.lower() for kw in keywords)


def rag_generate_answer_enhanced(query, k=3, snippet_chars=300):
    """RAG משופר עם נתוני IoT."""
    q_terms, results = search_top_k(query, k)

    if not results:
        return q_terms, results, "No documents found."

    # בניית הקשר מהמסמכים
    doc_context = _build_context(results, snippet_chars=snippet_chars)

    # הוספת נתוני IoT אם רלוונטי
    iot_context = ""
    if is_plant_related_query(query):
        iot_context = build_iot_context()

    # בניית prompt משופר
    prompt_parts = [
        "You are a plant health expert. Answer based ONLY on the context.",
        "Cite sources using [Doc X] format. Be concise.",
        ""
    ]

    if iot_context:
        prompt_parts.append("=== REAL-TIME SENSOR DATA ===")
        prompt_parts.append(iot_context)
        prompt_parts.append("")

    if doc_context:
        prompt_parts.append("=== DOCUMENTS ===")
        prompt_parts.append(doc_context)
        prompt_parts.append("")

    prompt_parts.append(f"Question: {query}")
    prompt_parts.append("Answer:")

    prompt = "\n".join(prompt_parts)

    # יצירת תשובה
    if gen is None:
        return q_terms, results, _fallback_answer(query, results)

    out = _call_gen(prompt, max_new_tokens=200)

    if _bad_answer(out):
        out = "I don't have enough information."

    return q_terms, results, out
```

---

## פתרון 4: עדכון ה-GUI של הצ'אט

### הקוד המעודכן ל-Tab 5:

```python
# =========================
# TAB 5 GUI - Enhanced RAG Chat
# =========================

def build_rag_chat_tab():
    gr.Markdown("## 💬 RAG Chat")
    gr.Markdown("Chat with plant documents + live sensor data")

    with gr.Row():
        with gr.Column(scale=2):
            query_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., What are symptoms of leaf disease?"
            )

        with gr.Column(scale=1):
            k_slider = gr.Slider(1, 10, value=3, step=1, label="Documents")
            include_iot = gr.Checkbox(value=True, label="Include sensor data")

    search_btn = gr.Button("Search & Answer", variant="primary")

    with gr.Row():
        with gr.Column():
            results_df = gr.Dataframe(
                headers=["Doc ID", "Score", "URL"],
                label="Retrieved Documents"
            )
        with gr.Column():
            iot_display = gr.Textbox(label="Sensor Data", lines=6)

    answer_box = gr.Textbox(label="RAG Answer", lines=8)

    def ui_query(q, k, use_iot):
        if not q.strip():
            return [], "", "Please enter a question."

        q_terms, results, answer = rag_generate_answer_enhanced(q, k=int(k))

        rows = [[r['doc_id'], r['score'], r['url']] for r in results]

        iot_text = build_iot_context() if use_iot else ""

        return rows, iot_text, answer

    search_btn.click(
        ui_query,
        inputs=[query_input, k_slider, include_iot],
        outputs=[results_df, iot_display, answer_box]
    )
```

---

## סיכום השיפורים

| תחום | לפני | אחרי |
|------|------|------|
| **אינדקס** | ספירת מילים פשוטה | TF-IDF / BM25 |
| **ניקוד** | מספר התאמות | משקל לפי חשיבות מילה |
| **הקשר** | 160 תווים | 300-800 תווים |
| **נתוני IoT** | לא קיים | משולב בשאלות רלוונטיות |
| **Prompt** | בסיסי | מובנה עם הנחיות |

---

## איך להטמיע בפרויקט

### שלב 1: גיבוי
```bash
cp HW2_Unicorn.ipynb HW2_Unicorn_backup.ipynb
```

### שלב 2: החלפת תאים
1. החלף את תא מספר 7 (Index Construction) בקוד TF-IDF
2. הוסף תא חדש עם פונקציות IoT Integration
3. החלף את תא מספר 11 (RAG Generation) בקוד המשופר
4. עדכן את תא ה-GUI של Tab 5

### שלב 3: בדיקה
```python
# בדיקת חיבור Firebase
stats = get_sensor_statistics()
print(stats)

# בדיקת ניתוח בריאות
health = analyze_plant_health()
print(health)

# בדיקת RAG משופר
q_terms, results, answer = rag_generate_answer_enhanced("What causes leaf spots?")
print(answer)
```

---

## קובץ מלא

הקוד המלא המשופר נמצא בקובץ:
**`improved_rag_system.py`**

ניתן לייבא אותו ישירות:
```python
from improved_rag_system import EnhancedRAG, BM25Index
```
