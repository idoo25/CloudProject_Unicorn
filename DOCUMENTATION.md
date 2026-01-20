# תיק מתכנת - Cloud Garden IoT & AI System

## סקירה כללית
מערכת Cloud Garden היא אפליקציית IoT חכמה לניטור וניהול גינה באמצעות חיישנים ובינה מלאכותית.
הקוד כתוב ב-Python עם Gradio לממשק משתמש, ותקשורת עם Firebase Realtime Database.
המערכת כוללת 8 טאבים עיקריים ומחולקת למודולים לוגיים.

---

## ארכיטקטורת המערכת

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLOUD GARDEN SYSTEM                         │
├─────────────────────────────────────────────────────────────────────┤
│  Frontend (Gradio)  │  Backend (Python)  │  Database (Firebase)    │
├─────────────────────────────────────────────────────────────────────┤
│  - 8 Tabs UI        │  - Data Processing │  - sensor_data/         │
│  - Interactive      │  - ML Models       │  - indexes/             │
│  - Real-time        │  - RAG System      │  - gamification/        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## קונפיגורציה וקבועים (Cell 3 + Cell 6)

| קבוע | תיאור | ערך |
|------|-------|-----|
| `FIREBASE_URL` | כתובת מסד הנתונים | `https://cloud-81451-default-rtdb.europe-west1.firebasedatabase.app/` |
| `CEREBRAS_API_KEY` | מפתח API ל-LLM | נטען מקובץ חיצוני |
| `REPORT_MODEL_NAME` | מודל LLM לדוחות | `llama3.1-8b` |
| `MODEL_NAME` | מודל זיהוי מחלות צמחים | `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification` |
| `COLOR_TEMP` | צבע גרף טמפרטורה | `#1f77b4` (כחול) |
| `COLOR_HUM` | צבע גרף לחות | `#ff7f0e` (כתום) |
| `COLOR_SOIL` | צבע גרף אדמה | `#2ca02c` (ירוק) |

---

## מבנה מסד הנתונים (Firebase)

```
firebase-project/
├── sensor_data/           ← נתוני חיישנים גולמיים (5926 רשומות)
│   ├── {id}/
│   │   ├── temperature    (float)
│   │   ├── humidity       (float)
│   │   ├── soil          (float)
│   │   └── created_at    (timestamp)
│
├── indexes/               ← אינדקסים למנוע החיפוש RAG
│   ├── public_index      (inverted index - 3317 terms)
│   ├── doc_map           (document ID → URL mapping)
│   ├── doc_text          (document ID → full text)
│   └── embeddings        (vector embeddings - 96 chunks)
│
└── gamification/          ← מערכת הגיימיפיקציה
    ├── points            (int)
    ├── spins_available   (int)
    ├── missions/         (mission tracking)
    └── coupons/          (earned coupons)
```

---

## מודולים ופונקציות

---

### 📊 Cell 2: Report Microservice
מיקרו-סרוויס ליצירת דוחות יומיים באמצעות FastAPI ו-LLM.

| פונקציה | תיאור | קלט | פלט |
|---------|-------|-----|-----|
| `health()` | בדיקת תקינות השירות | - | `{"status": "ok"}` |
| `records_to_df(records)` | המרת רשומות ל-DataFrame | רשימת מילונים | `pd.DataFrame` |
| `unify_sensor_dfs(temp_df, hum_df, soil_df)` | איחוד נתוני חיישנים | 3 DataFrames | DataFrame מאוחד |
| `prep(df)` | הכנת נתונים לעיבוד | DataFrame | DataFrame מעובד |
| `ReportGenerator.__init__()` | אתחול מחולל דוחות | - | - |
| `ReportGenerator.generate_daily_report(df)` | יצירת דוח יומי טקסטואלי | DataFrame | טקסט דוח |
| `ReportGenerator.create_docx_report(df, path)` | יצירת דוח Word | DataFrame, נתיב | נתיב לקובץ |
| `generate_docx(data)` | API endpoint לדוח | JSON | קובץ DOCX |

---

### 🔄 Cell 4: Data Ingestion & Sync
מודול לסנכרון נתונים מהשרת ל-Firebase.

| פונקציה | תיאור | קלט | פלט |
|---------|-------|-----|-----|
| `get_latest_timestamp_from_firebase()` | קבלת חותמת הזמן האחרונה | - | timestamp או None |
| `fetch_batch_from_server(after_ts, limit)` | שליפת נתונים מהשרת | timestamp, מגבלה | רשימת רשומות |
| `save_sensor_data_to_firebase(records)` | שמירת נתונים ל-Firebase | רשומות | מספר שנשמרו |
| `sync_new_data_from_server()` | סנכרון נתונים חדשים | - | `(הודעה, כמות)` |
| `load_data_from_firebase()` | טעינת כל הנתונים | - | `pd.DataFrame` |

---

### 🌡️ Cell 6: Dashboard & Plant Status
לוח בקרה בזמן אמת וזיהוי סטטוס צמחים.

| פונקציה | תיאור | קלט | פלט |
|---------|-------|-----|-----|
| `load_iot_data(sensor, limit)` | טעינת נתוני חיישן ספציפי | סוג חיישן, מגבלה | DataFrame |
| `normalize(series)` | נרמול ערכים ל-0-1 | Series | Series מנורמל |
| `plant_dashboard(limit)` | חישוב סטטוס צמח | מספר דגימות | סטטוס + גרפים |

**לוגיקת סטטוס צמח:**
```python
טווחים תקינים:
- טמפרטורה: 18-32°C
- לחות אוויר: 35-75%
- לחות אדמה: 20-60%

🟢 OK = כל הערכים בטווח
🟡 Warning = ערך קרוב לגבול
🔴 Not OK = ערך מחוץ לטווח
```

---

### 🖼️ Cell 7: UI Tab Builders (Part 1)
בניית טאבים לממשק המשתמש.

| פונקציה | תיאור |
|---------|-------|
| `build_realtime_dashboard_tab()` | בונה טאב דשבורד בזמן אמת עם גרפים וסטטוס |
| `df_to_records(df)` | המרת DataFrame לרשומות JSON |
| `call_report_microservice(records)` | קריאה למיקרו-סרוויס דוחות |
| `generate_report_screen(limit)` | יצירת מסך דוח עם הורדת DOCX |
| `build_generate_report_tab()` | בונה טאב יצירת דוחות |
| `analyze_plant(image, temp, hum, soil)` | ניתוח תמונת צמח לזיהוי מחלות |
| `build_plant_disease_detection_tab()` | בונה טאב זיהוי מחלות צמחים |

**זיהוי מחלות צמחים:**
```python
# שימוש ב-HuggingFace Pipeline
clf = pipeline("image-classification", model=MODEL_NAME)
preds = clf(image)  # מחזיר רשימת תחזיות עם confidence
```

---

### 🔍 Cell 8: RAG System (Retrieval-Augmented Generation)
מערכת RAG מלאה לחיפוש סמנטי במאמרים מדעיים.

#### 8.1 Vector Embeddings
| פונקציה | תיאור |
|---------|-------|
| `get_embed_model()` | טעינת מודל embeddings (all-MiniLM-L6-v2) |
| `compute_embeddings(texts)` | חישוב וקטורים לטקסטים |
| `chunk_text(text, size=500, overlap=50)` | חלוקת טקסט לקטעים |
| `build_and_save_embeddings(doc_text_map)` | בניית ושמירת embeddings ל-Firebase |
| `load_embeddings()` | טעינת embeddings מ-Firebase |
| `semantic_search(query, top_k=5)` | חיפוש סמנטי בוקטורים |

#### 8.2 Document Fetching
| פונקציה | תיאור |
|---------|-------|
| `fetch_html(url)` | הורדת HTML מ-URL |
| `extract_main_text_from_html(html)` | חילוץ טקסט עיקרי מ-HTML |
| `semantic_scholar_lookup(doi)` | חיפוש ב-Semantic Scholar |
| `openalex_lookup(doi)` | חיפוש ב-OpenAlex |
| `unpaywall_lookup(doi)` | חיפוש PDF חינמי ב-Unpaywall |
| `extract_text_from_pdf_url(url)` | חילוץ טקסט מ-PDF |
| `get_document_text(url)` | קבלת טקסט מלא של מסמך |

#### 8.3 Text Processing
| פונקציה | תיאור |
|---------|-------|
| `tokenize(text)` | פיצול טקסט למילים |
| `remove_stopwords(tokens)` | הסרת מילות עצירה |
| `apply_stemming(tokens)` | stemming למילים |
| `preprocess_query(query)` | עיבוד שאילתה לחיפוש |
| `postprocess_document_text(text)` | ניקוי טקסט מסמך |

#### 8.4 Inverted Index
| פונקציה | תיאור |
|---------|-------|
| `build_doc_text_map(urls)` | בניית מפת URL → טקסט |
| `build_inverted_index(urls, stop_words)` | בניית אינדקס הפוך |
| `save_to_firebase(data, path)` | שמירה ל-Firebase |
| `firebase_get(path)` | קריאה מ-Firebase |
| `check_existing_index()` | בדיקה אם קיים אינדקס |
| `smart_build_and_save_index(urls, stop_words)` | בנייה חכמה (רק חדשים) |
| `load_store_from_firebase()` | טעינת אינדקס מ-Firebase |

#### 8.5 Search & Ranking
| פונקציה | תיאור |
|---------|-------|
| `search_top_k(query, k=5)` | חיפוש K תוצאות מובילות |
| `bm25_rank(query, doc_ids)` | דירוג BM25 |
| `_extract_evidence_from_chunk(chunk, question)` | חילוץ ראיות מקטע |
| `_final_answer_from_evidence(question, evidence)` | יצירת תשובה סופית |
| `rag_answer_with_model(question)` | תשובה מלאה עם RAG |

---

### 💬 Cell 9: RAG Chat UI
ממשק צ'אט לחיפוש במאמרים.

| פונקציה | תיאור |
|---------|-------|
| `rag_ui(question)` | עיבוד שאלה והחזרת תשובה |
| `build_rag_chat_tab()` | בניית טאב חיפוש מאמרים |

---

### 🤖 Cell 10: Smart Chat (AI Assistant)
צ'אט חכם עם הקשר מחיישנים ומאמרים.

| פונקציה | תיאור |
|---------|-------|
| `get_current_sensor_summary()` | סיכום מצב חיישנים נוכחי |
| `get_rag_context(query)` | קבלת הקשר רלוונטי מ-RAG |
| `build_smart_system_prompt(sensor, rag)` | בניית system prompt מותאם |
| `cerebras_smart_turn(message, history, temp)` | תור שיחה עם Cerebras LLM |
| `build_smart_chat_tab()` | בניית טאב צ'אט חכם |
| `clear_chat()` | ניקוי היסטוריית צ'אט |

**System Prompt:**
```
You are a professional agricultural AI assistant with real-time access to:
- IoT sensor data (temperature, humidity, soil moisture)
- Scientific papers on plant diseases
Provide data-driven advice about plant care and disease identification.
```

---

### 🎮 Cell 11: Gamification System
מערכת גיימיפיקציה עם נקודות, משימות וגלגל מזל.

#### 11.1 Profile Management
| פונקציה | תיאור |
|---------|-------|
| `_today_key()` | מפתח תאריך היום (YYYY-MM-DD) |
| `_now_iso()` | חותמת זמן ISO |
| `_get_profile()` | טעינת פרופיל משתמש |
| `_save_profile(prof)` | שמירת פרופיל |

#### 11.2 Missions & Rewards
| פונקציה | תיאור | נקודות |
|---------|-------|--------|
| `complete_mission(mission_id, points)` | השלמת משימה (פעם ביום) | - |
| `spin_wheel()` | סיבוב גלגל מזל | 5/10/20 או קופון |
| `redeem_voucher(tier)` | מימוש נקודות לקופון | -50/-100/-200 |

#### 11.3 Gamified Wrappers
| פונקציה | תיאור | נקודות |
|---------|-------|--------|
| `sync_screen_gamified()` | סנכרון + משימה | +10 |
| `analyze_plant_gamified(image, ...)` | ניתוח צמח + משימה | +15 |
| `generate_report_screen_gamified(limit)` | דוח + משימה | +12 |

#### 11.4 Tab Builders
| פונקציה | תיאור |
|---------|-------|
| `build_iot_dashboard_tab()` | טאב דשבורד IoT מתקדם |
| `build_search_engine_tab()` | טאב מנוע חיפוש |
| `build_sync_data_tab()` | טאב סנכרון נתונים |
| `build_rewards_tab()` | טאב תגמולים וגלגל מזל |

---

### 📈 Cell 12: Advanced Analytics & Visualizations
ניתוחים מתקדמים וויזואליזציות.

| פונקציה | תיאור |
|---------|-------|
| `create_kpi_card(title, value, icon, color)` | יצירת כרטיס KPI |
| `create_status_badge(status, color)` | יצירת תג סטטוס |
| `create_stat_cards_html(df)` | כרטיסי סטטיסטיקות HTML |
| `time_series_overview(df)` | גרף סקירת זמן |
| `calculate_correlations(df)` | חישוב וגרף קורלציות |
| `hourly_patterns(df)` | דפוסים שעתיים |
| `daily_patterns(df)` | דפוסים יומיים |
| `distribution_analysis(df)` | היסטוגרמות התפלגות |
| `time_series_decomposition(df, var)` | פירוק סדרות זמן + ממוצעים נעים |
| `create_kpi_cards(df)` | יצירת כל כרטיסי KPI |
| `create_time_series_plot(df)` | גרף סדרות זמן אינטראקטיבי |

---

### 🖥️ Cell 13: Screen Functions
פונקציות מסך ראשיות.

| פונקציה | תיאור |
|---------|-------|
| `sync_screen()` | מסך סנכרון נתונים |
| `dashboard_screen()` | מסך דשבורד מלא (11 רכיבים) |
| `dashboard_moving_avg(variable)` | גרף ממוצעים נעים לפי משתנה |

---

### 🚀 Cell 14: Initialization & App Builder
אתחול המערכת ובניית האפליקציה.

#### 14.1 Preloading Functions
| פונקציה | תיאור | תוצאה |
|---------|-------|-------|
| `initialize_firebase()` | אתחול חיבור Firebase | True/False |
| `preload_sensor_data()` | טעינת נתוני חיישנים | True/False |
| `preload_rag_index()` | טעינת אינדקס RAG | True/False |
| `preload_embeddings()` | טעינת embeddings | True/False |
| `preload_embed_model()` | טעינת מודל embeddings | True/False |
| `preload_ml_model()` | טעינת מודל ML | True/False |
| `preload_gamification()` | טעינת פרופיל גיימיפיקציה | True/False |
| `build_index_if_missing()` | בניית אינדקס חסר | True/False |

#### 14.2 Main Functions
| פונקציה | תיאור |
|---------|-------|
| `initialize_all()` | אתחול כל 8 הרכיבים |
| `get_cached_sensor_data()` | קבלת נתונים מ-cache |
| `get_cached_index()` | קבלת אינדקס מ-cache |
| `build_app()` | בניית אפליקציית Gradio |

**סדר אתחול:**
```
1. Firebase Connection
2. Sensor Data (5926 records)
3. RAG Index (3317 terms)
4. Vector Embeddings (96 chunks)
5. Embedding Model (all-MiniLM-L6-v2)
6. Plant Disease ML (MobileNetV2)
7. Gamification Profile
8. Build Missing Index
```

---

## טאבים באפליקציה

| # | טאב | תיאור | פונקציה בונה |
|---|-----|-------|--------------|
| 1 | 🌿 Real-Time Dashboard | סטטוס צמח וגרפים בזמן אמת | `build_realtime_dashboard_tab()` |
| 2 | 📈 IoT Dashboard | ניתוחים מתקדמים וסטטיסטיקות | `build_iot_dashboard_tab()` |
| 3 | 📄 Generate Report | יצירת דוחות Word | `build_generate_report_tab()` |
| 4 | 🖼️ Plant Disease Detection | זיהוי מחלות מתמונה | `build_plant_disease_detection_tab()` |
| 5 | 🔍 Search Engine | חיפוש במאמרים מדעיים | `build_search_engine_tab()` |
| 6 | 💬 Smart Chat | צ'אט AI עם הקשר חיישנים | `build_smart_chat_tab()` |
| 7 | 🔄 Sync Data | סנכרון נתונים מהשרת | `build_sync_data_tab()` |
| 8 | 🎮 Rewards | נקודות, משימות וגלגל מזל | `build_rewards_tab()` |

---

## מאמרים מדעיים (DOC_URLS)

| # | נושא | מקור |
|---|------|------|
| 1 | Medicinal plant leaf disease classification | Scientific Reports |
| 2 | Tomato Diseases and Pests Detection | Frontiers in Plant Science |
| 3 | Deep Learning for Plant Disease Detection | arXiv |
| 4 | Smart Agriculture Sensors | MDPI |
| 5 | Soil Moisture Monitoring | IEEE |

---

## תלויות (Dependencies)

```python
# Core
pandas, numpy, matplotlib, plotly

# ML & AI
torch, transformers, sentence-transformers
scikit-learn, scipy

# Web & API
gradio, requests, fastapi, uvicorn

# Firebase
firebase-admin

# Documents
python-docx, PyPDF2, beautifulsoup4

# LLM
cerebras-cloud-sdk
```

---

## Cache Structure (CACHE dict)

```python
CACHE = {
    'firebase_initialized': bool,
    'sensor_data': pd.DataFrame,      # 5926 records
    'rag_index': dict,                # 3317 terms
    'doc_map': dict,                  # 5 documents
    'doc_text': dict,                 # 5 documents
    'embeddings': dict,               # 96 chunks
    'embed_model': SentenceTransformer,
    'gamification_profile': dict,
    'ml_model': Pipeline,
}
```

---

## Error Handling

המערכת כוללת טיפול בשגיאות בכל הרמות:
- **Firebase**: retry עם exponential backoff
- **ML Models**: fallback להודעת שגיאה ידידותית
- **RAG**: חיפוש חלופי אם אינדקס חסר
- **Gamification**: ברירות מחדל לפרופיל חדש

---

*תיעוד זה נוצר אוטומטית מהקוד - Cloud Garden v1.0*
