# Revora

**Revora** is an adaptive, policy-bounded revenue recovery agent for failed Razorpay payments.

> **Detect → Decide → Recover → Learn**

---

## Project Structure

```text
adaptive-recovery agent/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py         # Environment configuration (Pydantic Settings)
│   │   ├── database.py       # SQLAlchemy SQLite setup & session dependency
│   │   └── main.py           # FastAPI entry point & /health endpoint
│   ├── tests/
│   │   ├── __init__.py
│   │   └── test_health.py    # Health check endpoint unit tests (pytest)
│   ├── .env.example          # Backend environment variable template
│   └── requirements.txt      # Python dependencies
├── frontend/
│   ├── src/
│   │   └── app/
│   │       ├── globals.css   # Tailwind styles
│   │       ├── layout.tsx    # Root layout & page metadata
│   │       └── page.tsx      # Minimal landing view
│   ├── .env.example          # Frontend environment variable template
│   ├── package.json          # Frontend dependencies & scripts
│   ├── postcss.config.js     # PostCSS configuration
│   ├── tailwind.config.ts    # Tailwind CSS configuration
│   └── tsconfig.json         # TypeScript configuration
├── .gitignore
└── README.md
```

---

## Getting Started

### Prerequisites

- **Python 3.10+** (with `pip` and `venv`)
- **Node.js 18+** and **npm**

---

### Backend Setup & Run

1. Navigate to the `backend` directory:
   ```bash
   cd backend
   ```

2. Create and activate a Python virtual environment:
   - **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **Linux/macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create your local environment file:
   ```bash
   cp .env.example .env
   ```

5. Run the FastAPI development server:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

6. Verify the server is running:
   - Health check: [http://localhost:8000/health](http://localhost:8000/health)
   - Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

7. Example Decision & Execution API Request:
   ```bash
   curl -X POST http://localhost:8000/api/recovery/decision \
     -H "Content-Type: application/json" \
     -d '{
       "amount": 2500.0,
       "currency": "INR",
       "payment_method": "upi",
       "failure_reason": "bank_technical_timeout",
       "payment_status": "failed",
       "execute_action": true,
       "max_attempts": 3
     }'
   ```

   Response:
   ```json
   {
     "recommended_action": "payment_link",
     "confidence": 0.85,
     "reasoning": "Interactive payment link required for recovery.",
     "key_factors": ["transient_network_error"],
     "referenced_case_ids": [],
     "agent_used": true,
     "policy_overridden": false,
     "is_fallback": false,
     "fallback_reason": null,
     "execution": {
       "action": "payment_link",
       "attempted": true,
       "status": "simulated",
       "success": true,
       "reference_id": "plink_sim_3b9f12d8a4",
       "resource_url": "https://rzp.io/i/sim_8a7d2c1e",
       "message": "Razorpay Payment Link generated successfully: https://rzp.io/i/sim_8a7d2c1e",
       "error": null
     }
   }
   ```

---

### Running Backend Tests

With your virtual environment activated and from the `backend/` directory:

```bash
pytest
```

---

### Frontend Setup & Run

1. Navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. (Optional) Create local environment file:
   ```bash
   cp .env.example .env.local
   ```

4. Start the development server:
   ```bash
   npm run dev
   ```

5. Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## Architectural Decisions

1. **Explicit Monorepo Split (`backend/` and `frontend/`)**:
   - Keeps backend API development and frontend UI development isolated and cleanly organized, allowing independent dependency management, testing, and deployment.
2. **Pydantic Settings (`app/config.py`)**:
   - Enforces typed, validated environment variables with sensible defaults (like SQLite for local development) without scattered `os.getenv` calls.
3. **Database Session Dependency (`app/database.py`)**:
   - Uses SQLAlchemy `sessionmaker` with a generator `get_db()` function so database connections open and close deterministically per request.
4. **Lightweight FastAPI + SQLite**:
   - Provides an asynchronous REST framework with zero external database server setup required for initial local development and learning.
