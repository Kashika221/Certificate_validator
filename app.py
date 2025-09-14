from flask import Flask, request, render_template, jsonify, flash, redirect, url_for
import os
import json
import tempfile
import shutil
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import logging
from pathlib import Path

# Import your certificate verifier class
# Make sure the certificate verification code is in the same directory or properly imported
from Validator import CertificateVerifier, CertificateVerificationError

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-change-this')

# Configure Flask app
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp(prefix="flask_uploads_")

# Supported file extensions
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.pdf', '.docx', '.bmp', '.tiff', '.gif'}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def cleanup_upload_folder():
    """Clean up old uploaded files"""
    try:
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            shutil.rmtree(app.config['UPLOAD_FOLDER'], ignore_errors=True)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    except Exception as e:
        logger.warning(f"Cleanup warning: {str(e)}")

@app.route('/')
def index():
    """Main page with upload form"""
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify_certificate():
    """Handle certificate verification"""
    verifier = None
    base_cert_path = None
    cert_path = None
    
    try:
        # Check if files were uploaded
        if 'base_certificate' not in request.files or 'certificate' not in request.files:
            return jsonify({
                'success': False,
                'message': 'Both base certificate and certificate files are required'
            }), 400
        
        base_cert_file = request.files['base_certificate']
        cert_file = request.files['certificate']
        
        # Check if files were selected
        if base_cert_file.filename == '' or cert_file.filename == '':
            return jsonify({
                'success': False,
                'message': 'Please select both files'
            }), 400
        
        # Check file extensions
        if not (allowed_file(base_cert_file.filename) and allowed_file(cert_file.filename)):
            return jsonify({
                'success': False,
                'message': f'Unsupported file format. Allowed formats: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Save uploaded files
        base_cert_filename = secure_filename(base_cert_file.filename)
        cert_filename = secure_filename(cert_file.filename)
        
        base_cert_path = os.path.join(app.config['UPLOAD_FOLDER'], f"base_{base_cert_filename}")
        cert_path = os.path.join(app.config['UPLOAD_FOLDER'], f"cert_{cert_filename}")
        
        base_cert_file.save(base_cert_path)
        cert_file.save(cert_path)
        
        logger.info(f"Files uploaded: {base_cert_path}, {cert_path}")
        
        # Initialize verifier and perform verification
        verifier = CertificateVerifier()
        print("\n", base_cert_path, cert_path, "\n")
        result = verifier.verify_certificate(cert_path, base_cert_path)
        
        # Clean up uploaded files
        try:
            os.remove(base_cert_path)
            os.remove(cert_path)
        except Exception as e:
            logger.warning(f"Failed to clean up uploaded files: {str(e)}")
        
        return jsonify(result)
        
    except RequestEntityTooLarge:
        return jsonify({
            'success': False,
            'message': 'File too large. Maximum size allowed is 16MB'
        }), 413
        
    except CertificateVerificationError as e:
        logger.error(f"Certificate verification error: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 400
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'An unexpected error occurred during verification'
        }), 500
        
    finally:
        # Clean up verifier and files
        if verifier:
            verifier.cleanup()
        
        # Clean up uploaded files if they still exist
        for file_path in [base_cert_path, cert_path]:
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up file {file_path}: {str(e)}")

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'certificate-verification'})

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False,
        'message': 'File too large. Maximum size allowed is 16MB'
    }), 413

@app.errorhandler(404)
def not_found(e):
    return render_template('index.html'), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({
        'success': False,
        'message': 'Internal server error occurred'
    }), 500

if __name__ == '__main__':
    try:
        # Clean up upload folder on startup
        cleanup_upload_folder()
        
        # Run the Flask app
        port = int(os.getenv('PORT', 5000))
        debug_mode = os.getenv('FLASK_ENV') == 'development'
        
        print(f"Starting Certificate Verification Server on port {port}")
        print(f"Debug mode: {debug_mode}")
        print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
        
        app.run(
            host='0.0.0.0',
            port=port,
            debug=debug_mode,
            threaded=True
        )
        
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    finally:
        # Clean up on exit
        cleanup_upload_folder()