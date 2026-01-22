# Certificate Verification Service

A Flask-based web application designed to upload, process, and verify certificates against a base template. This application accepts image or document files, compares them using a custom validator, and returns a verification status.

### Live Demo

-  https://certificate-validator-326x.onrender.com

## Features

* **Dual File Upload:** Accepts a "Base Certificate" (template) and a specific "Certificate" for comparison.
* **Format Support:** Supports `.jpg`, `.jpeg`, `.png`, `.pdf`, `.docx`, `.bmp`, `.tiff`, and `.gif`.
* **Security:**
* Secure filename handling.
* Automatic temporary file cleanup (after request and on server start/exit).
* File size limiting (Max 16MB).


* **API Ready:** Returns JSON responses suitable for frontend integration.
* **Health Check:** Dedicated endpoint for uptime monitoring.

## Project Structure

Ensure your directory looks like this for the application to run correctly:

```text
project-root/
├── app.py                 # The main Flask application (your code)
├── Validator.py           # Must contain CertificateVerifier class
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── templates/
    └── index.html         # Frontend upload form

```

> **Important:** The code imports `CertificateVerifier` from a module named `Validator`. Ensure `Validator.py` exists in the same directory.

## Installation & Setup

### 1. Prerequisites

* Python 3.8 or higher
* pip (Python package manager)

### 2. Create a Virtual Environment

It is recommended to use a virtual environment to manage dependencies.

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

Create a `requirements.txt` with the following content (plus any dependencies required by your `Validator.py`):

```text
Flask
Werkzeug

```

Then install them:

```bash
pip install -r requirements.txt

```

### 4. Configuration (Environment Variables)

You can configure the app using environment variables. Create a `.env` file or export them in your terminal:

| Variable | Description | Default |
| --- | --- | --- |
| `FLASK_APP` | Entry point file | `app.py` |
| `FLASK_ENV` | Environment mode | `production` (set to `development` for debug logs) |
| `PORT` | Port to run the server on | `5000` |
| `FLASK_SECRET_KEY` | Key for signing sessions | `your-secret-key...` |

## Usage

### Starting the Server

```bash
python app.py

```

* The server will start at `http://0.0.0.0:5000` (or your defined PORT).
* Visit `http://localhost:5000` to see the upload interface (requires `index.html`).

### API Endpoints

#### 1. Verify Certificate

* **URL:** `/verify`
* **Method:** `POST`
* **Content-Type:** `multipart/form-data`
* **Body:**
* `base_certificate`: File (Required)
* `certificate`: File (Required)



**Success Response (200 OK):**

```json
{
    "success": true,
    "score": 0.98,
    "is_valid": true,
    "details": "..."
}

```

*(Note: The exact JSON structure depends on your `Validator.py` return value)*

**Error Response (400 Bad Request):**

```json
{
    "success": false,
    "message": "Unsupported file format..."
}

```

#### 2. Health Check

* **URL:** `/api/health`
* **Method:** `GET`
* **Response:**

```json
{
    "status": "healthy",
    "service": "certificate-verification"
}

```

## Security & Limitations

* **File Size:** Uploads are strictly limited to **16MB** via `MAX_CONTENT_LENGTH`.
* **Storage:** Files are stored in the system's temporary directory (`tempfile.mkdtemp`) and are deleted immediately after verification to ensure data privacy.
* **Concurrency:** The app runs in `threaded=True` mode by default, suitable for basic concurrency, but a WSGI server (like Gunicorn) is recommended for production.

## Contributing

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.