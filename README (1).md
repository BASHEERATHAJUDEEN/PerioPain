# PerioPain AI™ — Backend API

A Flask backend for the PerioPain AI dental pain classification system.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key (optional — enables live Claude AI)
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 3. Run the server
python app.py
# Server starts at http://localhost:8000
```

---

## API Reference

### Auth
| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/auth/register` | Register new user |
| POST | `/api/auth/login` | Login → returns JWT |
| GET  | `/api/auth/me` | Get current user (auth required) |

**Login body:**
```json
{ "email": "dentist@periopain.ai", "password": "Dentist@123" }
```

**Response:**
```json
{ "token": "eyJ...", "user": { "id": "...", "email": "...", "name": "...", "role": "DENTIST" } }
```

All protected routes require: `Authorization: Bearer <token>`

---

### Cases
| Method | Route | Description |
|--------|-------|-------------|
| GET    | `/api/cases` | List cases (filter: `?q=`, `?status=`, `?pain=`) |
| POST   | `/api/cases` | Create case + auto-run AI prediction |
| GET    | `/api/cases/:id` | Get single case |
| PUT    | `/api/cases/:id` | Update case |
| DELETE | `/api/cases/:id` | Delete case (ADMIN only) |

**Create Case body** (all fields optional except patient_name):
```json
{
  "patient_name": "Anjali Mehta",
  "patient_age": 34,
  "patient_gender": "F",
  "tooth_number": "16",
  "chief_complaint": "Severe throbbing toothache",
  "pain_duration": "3 days",
  "pain_character": "Throbbing",
  "vas_score": 8,
  "spontaneous_pain": true,
  "night_pain": true,
  "cold_response": "Lingering >30s",
  "heat_response": "Severe Pain",
  "ept_value": 0,
  "ept_interpretation": "Non-vital/Necrotic",
  "vertical_percussion": "Severe Pain",
  "ppd_max": 3,
  "bop": "Absent"
}
```

The backend automatically runs the AI prediction engine on case creation.

---

### AI Prediction
| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/predict` | Run prediction on clinical data |
| POST | `/api/predict/reasoning` | Get Claude AI clinical reasoning |

**Predict response:**
```json
{
  "predicted_class": "Endodontic",
  "confidence": 94.2,
  "probabilities": { "Endodontic": 94.2, "Periodontal": 3.1, "Occlusal": 1.4, "Combined": 1.3 },
  "shap": { "Lingering Cold Pain": 0.30, "Vertical Percussion": 0.22, ... },
  "needs_review": false,
  "inference_ms": 14.2
}
```

---

### Radiograph Upload
| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/upload/radiograph` | Upload PNG/JPG/DICOM + get AI analysis |

Multipart form: `file` (image), `case_id` (optional), `xray_type` (optional)

---

### AI Chat
| Method | Route | Description |
|--------|-------|-------------|
| POST | `/api/chat` | Chat with dental AI assistant |

```json
{
  "messages": [
    { "role": "user", "content": "What distinguishes endodontic from periodontal pain?" }
  ]
}
```

Requires `ANTHROPIC_API_KEY` to be set.

---

### Dashboard & Research
| Method | Route | Description |
|--------|-------|-------------|
| GET | `/api/dashboard/stats` | Stats, monthly volume, distribution |
| GET | `/api/research/metrics` | Full ML metrics, confusion matrix, ROC |

---

### Admin (ADMIN role only)
| Method | Route | Description |
|--------|-------|-------------|
| GET | `/api/admin/users` | List all users |
| PUT | `/api/admin/users/:id/suspend` | Suspend user |
| PUT | `/api/admin/users/:id/activate` | Activate user |
| GET | `/api/admin/audit` | Last 100 audit log entries |

---

### Health Check
```
GET /api/health
```

---

## Demo Accounts (pre-seeded)

| Email | Password | Role |
|-------|----------|------|
| admin@periopain.ai | Admin@123 | ADMIN |
| dentist@periopain.ai | Dentist@123 | DENTIST |
| researcher@periopain.ai | Research@123 | RESEARCHER |

---

## Connecting the Frontend

In `periopain.html`, change the hardcoded Anthropic API calls to use this backend instead:

```javascript
// Replace direct Claude API calls with:
const BASE = "http://localhost:8000";

// Login
const resp = await fetch(`${BASE}/api/auth/login`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ email, password })
});
const { token, user } = await resp.json();
localStorage.setItem("token", token);

// Authenticated request
const headers = {
  "Content-Type": "application/json",
  "Authorization": `Bearer ${localStorage.getItem("token")}`
};
```

---

## Production Notes

- Replace SQLite with PostgreSQL for multi-user production use
- Set a strong `SECRET_KEY` in environment variables
- Use `gunicorn app:app --workers 4` instead of Flask dev server
- Add HTTPS via nginx reverse proxy
- Store uploaded radiographs in S3 or similar object storage

---

## Architecture

```
periopain-backend/
├── app.py          — Main Flask application (all routes)
├── requirements.txt
├── .env.example    — Copy to .env and fill in secrets
├── periopain.db    — SQLite database (auto-created on first run)
└── uploads/        — Uploaded radiograph files
```

Database tables: `users`, `cases`, `radiographs`, `audit_log`

The prediction engine (`_predict()`) implements a weighted rule-based system
that mirrors XGBoost feature importance. Key clinical signals:

- **Endodontic**: lingering cold pain, EPT non-vital, vertical percussion pain, periapical radiolucency, night pain
- **Periodontal**: deep PPD ≥6mm, generalised BOP, suppuration, mobility, bone loss
- **Occlusal**: bite test pain, fremitus, wear facets, high point premature contact
- **Combined**: triggered when two categories score within 20 points of each other
