from flask import Flask, request, render_template, jsonify
import os
import tempfile
import shutil
import gc
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import logging
from pathlib import Path
from contextlib import contextmanager

# Import your certificate verifier class
from Validator import CertificateVerifier, CertificateVerificationError

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-change-this')

# Memory-optimized configuration for Render
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # Reduced to 5MB for Render
app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp(prefix="flask_uploads_")

# Supported file extensions
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.pdf', '.docx', '.bmp', '.tiff', '.gif'}

# Configure logging for production
logging.basicConfig(
    level=logging.WARNING,  # Reduced verbosity
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
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

@contextmanager
def managed_verifier():
    """Context manager for proper verifier cleanup"""
    verifier = None
    try:
        verifier = CertificateVerifier()
        yield verifier
    finally:
        if verifier:
            verifier.cleanup()
        # Force garbage collection
        gc.collect()

@app.before_request
def limit_request_size():
    """Limit request size for memory optimization"""
    if request.content_length and request.content_length > app.config['MAX_CONTENT_LENGTH']:
        return jsonify({
            'success': False,
            'message': 'File too large. Maximum size allowed is 5MB'
        }), 413

@app.route('/')
def index():
    """Main page with upload form"""
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify_certificate():
    """Handle certificate verification with memory optimization"""
    cert_path = None
    base_cert_path = None
    
    try:
        # Early validation
        if 'base_certificate' not in request.files or 'certificate' not in request.files:
            return jsonify({
                'success': False,
                'message': 'Both base certificate and certificate files are required'
            }), 400
        
        base_cert_file = request.files['base_certificate']
        cert_file = request.files['certificate']
        
        # Validate files
        if base_cert_file.filename == '' or cert_file.filename == '':
            return jsonify({
                'success': False,
                'message': 'Please select both files'
            }), 400
        
        if not (allowed_file(base_cert_file.filename) and allowed_file(cert_file.filename)):
            return jsonify({
                'success': False,
                'message': f'Unsupported file format. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Create unique temporary directory for this request
        request_temp_dir = tempfile.mkdtemp(prefix="req_")
        
        try:
            # Save uploaded files with size check
            base_cert_filename = secure_filename(base_cert_file.filename)
            cert_filename = secure_filename(cert_file.filename)
            
            base_cert_path = os.path.join(request_temp_dir, f"base_{base_cert_filename}")
            cert_path = os.path.join(request_temp_dir, f"cert_{cert_filename}")
            
            # Save files
            base_cert_file.save(base_cert_path)
            cert_file.save(cert_path)
            
            # Additional size validation after saving
            if (os.path.getsize(base_cert_path) > app.config['MAX_CONTENT_LENGTH'] or 
                os.path.getsize(cert_path) > app.config['MAX_CONTENT_LENGTH']):
                return jsonify({
                    'success': False,
                    'message': 'File too large. Maximum size allowed is 5MB'
                }), 413
            
            logger.info(f"Processing files: {os.path.basename(cert_path)}, {os.path.basename(base_cert_path)}")
            
            # Use context manager for verifier
            with managed_verifier() as verifier:
                result = verifier.verify_certificate(cert_path, base_cert_path)
            
            return jsonify(result)
            
        finally:
            # Clean up request temporary directory
            try:
                shutil.rmtree(request_temp_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Failed to clean up request temp dir: {str(e)}")
        
    except RequestEntityTooLarge:
        return jsonify({
            'success': False,
            'message': 'File too large. Maximum size allowed is 5MB'
        }), 413
        
    except CertificateVerificationError as e:
        logger.error(f"Verification error: {str(e)}")
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
        # Force garbage collection after each request
        gc.collect()

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'service': 'certificate-verification',
        'memory_info': get_memory_info()
    })

def get_memory_info():
    """Get basic memory information for monitoring"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            'rss_mb': round(memory_info.rss / 1024 / 1024, 2),
            'vms_mb': round(memory_info.vms / 1024 / 1024, 2)
        }
    except ImportError:
        return {'info': 'psutil not available'}

@app.route('/api/gc')
def force_gc():
    """Manual garbage collection endpoint for debugging"""
    if os.getenv('FLASK_ENV') == 'development':
        collected = gc.collect()
        return jsonify({
            'collected': collected,
            'memory_info': get_memory_info()
        })
    else:
        return jsonify({'error': 'Not available in production'}), 403

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False,
        'message': 'File too large. Maximum size allowed is 5MB'
    }), 413

@app.errorhandler(404)
def not_found(e):
    return render_template('index.html'), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {str(e)}")
    # Force garbage collection on server errors
    gc.collect()
    return jsonify({
        'success': False,
        'message': 'Internal server error occurred'
    }), 500

@app.errorhandler(MemoryError)
def memory_error(e):
    logger.error(f"Memory error: {str(e)}")
    # Force aggressive cleanup
    gc.collect()
    return jsonify({
        'success': False,
        'message': 'Server is experiencing memory constraints. Please try with smaller files.'
    }), 507

# Periodic cleanup function
def periodic_cleanup():
    """Perform periodic cleanup tasks"""
    try:
        # Clean up upload folder
        cleanup_upload_folder()
        
        # Force garbage collection
        collected = gc.collect()
        logger.info(f"Periodic cleanup: collected {collected} objects")
        
    except Exception as e:
        logger.warning(f"Periodic cleanup failed: {str(e)}")

# Request hooks for memory management
@app.after_request
def after_request(response):
    """Clean up after each request"""
    # Force garbage collection after heavy operations
    if request.endpoint == 'verify_certificate':
        gc.collect()
    return response

if __name__ == '__main__':
    try:
        # Initial cleanup
        cleanup_upload_folder()
        
        # Set up periodic cleanup (every 30 minutes)
        import threading
        import time
        
        def cleanup_worker():
            while True:
                time.sleep(1800)  # 30 minutes
                periodic_cleanup()
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
        
        # Configuration for Render deployment
        port = int(os.getenv('PORT', 5000))
        debug_mode = os.getenv('FLASK_ENV') == 'development'
        
        # Render-specific optimizations
        if os.getenv('RENDER'):
            # Running on Render
            logger.info("Detected Render environment - applying optimizations")
            app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024  # Further reduce to 3MB on Render
            
        print(f"Starting Certificate Verification Server on port {port}")
        print(f"Debug mode: {debug_mode}")
        print(f"Max file size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.1f}MB")
        
        app.run(
            host='0.0.0.0',
            port=port,
            debug=debug_mode,
            threaded=True,
            use_reloader=False  # Disable reloader to save memory
        )
        
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
    finally:
        # Final cleanup
        cleanup_upload_folder()
        gc.collect()

def main():
    """CLI usage with memory optimization"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Certificate Verification System')
    parser.add_argument('certificate', help='Path to certificate file')
    parser.add_argument('base_certificate', help='Path to base certificate template')
    parser.add_argument('--output', '-o', help='Output file for results (JSON)', default=None)
    
    args = parser.parse_args()
    
    verifier = None
    try:
        verifier = CertificateVerifier()
        result = verifier.verify_certificate(args.certificate, args.base_certificate)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
        else:
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    finally:
        if verifier:
            verifier.cleanup()
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())