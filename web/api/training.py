# Flask backend API for BVC vulnerability detection system
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import os
import json
import sqlite3
import uuid
import threading
import time
from datetime import datetime
import subprocess
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import mimetypes

app = Flask(__name__)
app.config['SECRET_KEY'] = 'bvc-vuln-detection-system'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), '../assets/datasets')
app.config['MODEL_FOLDER'] = os.path.join(os.path.dirname(__file__), '../assets/models')
app.config['WEB_FOLDER'] = os.path.join(os.path.dirname(__file__), '..')  # Points to web directory
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

CORS(app, origins="*")

# Configure SocketIO with appropriate async mode for Python 3.12 compatibility
# Note: eventlet is not compatible with Python 3.12 due to removed ssl.wrap_socket
try:
    # Try threading mode first (most compatible)
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
    print("SocketIO configured with threading async mode")
except Exception as e:
    print(f"Warning: Could not configure SocketIO with threading mode: {e}")
    try:
        # Fallback to gevent if available
        socketio = SocketIO(app, cors_allowed_origins="*", async_mode='gevent')
        print("SocketIO configured with gevent async mode")
    except Exception as e2:
        print(f"Warning: Could not configure SocketIO with gevent mode: {e2}")
        # Final fallback - let SocketIO choose automatically (excluding eventlet)
        socketio = SocketIO(app, cors_allowed_origins="*")
        print("SocketIO configured with automatic async mode selection")

# Ensure upload directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# Database initialization
def init_db():
    """Initialize the SQLite database for training history and dataset metadata"""
    conn = sqlite3.connect('bvc_system.db')
    cursor = conn.cursor()
    
    # Training tasks table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_tasks (
            id TEXT PRIMARY KEY,
            model TEXT NOT NULL,
            dataset TEXT NOT NULL,
            config TEXT NOT NULL,
            status TEXT NOT NULL,
            start_time TIMESTAMP,
            end_time TIMESTAMP,
            best_accuracy REAL,
            final_loss REAL,
            auc_score REAL,
            f1_score REAL,
            model_path TEXT
        )
    ''')
    
    # Datasets table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS datasets (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            version TEXT,
            architecture TEXT NOT NULL,
            optimization TEXT NOT NULL,
            description TEXT,
            file_path TEXT NOT NULL,
            samples INTEGER,
            size_bytes INTEGER,
            positive_samples INTEGER,
            negative_samples INTEGER,
            created_at TIMESTAMP,
            modified_at TIMESTAMP,
            status TEXT NOT NULL
        )
    ''')
    
    # Training logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            level TEXT NOT NULL,
            message TEXT NOT NULL,
            FOREIGN KEY (task_id) REFERENCES training_tasks (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Global training state
training_state = {
    'active_tasks': {},
    'task_threads': {}
}

# Utility functions
def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    ALLOWED_EXTENSIONS = {'csv', 'json', 'zip', 'tfrecord'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_db_connection():
    """Get a database connection"""
    conn = sqlite3.connect('bvc_system.db')
    conn.row_factory = sqlite3.Row
    return conn

def log_training_message(task_id, level, message):
    """Log a training message to database and emit via WebSocket"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO training_logs (task_id, level, message) VALUES (?, ?, ?)',
        (task_id, level, message)
    )
    conn.commit()
    conn.close()
    
    # Emit via WebSocket
    socketio.emit('training_log', {
        'task_id': task_id,
        'level': level,
        'message': message,
        'timestamp': datetime.now().isoformat()
    })

# Mock training function (replace with actual training logic)
def mock_training_process(task_id, config):
    """Simulate a training process with progress updates"""
    try:
        log_training_message(task_id, 'info', f'Starting training with {config["model"]} model')
        
        max_epochs = config.get('max_iter', 100)
        
        for epoch in range(1, max_epochs + 1):
            if training_state['active_tasks'].get(task_id, {}).get('status') == 'stopped':
                break
                
            # Simulate training progress
            time.sleep(2)  # Simulate time per epoch
            
            # Generate mock metrics
            train_loss = 1.0 - (epoch / max_epochs) * 0.8 + np.random.normal(0, 0.05)
            val_loss = train_loss + np.random.normal(0, 0.02)
            train_acc = 0.5 + (epoch / max_epochs) * 0.4 + np.random.normal(0, 0.02)
            val_acc = train_acc - np.random.normal(0, 0.01)
            
            # Update training state
            if task_id in training_state['active_tasks']:
                training_state['active_tasks'][task_id].update({
                    'epoch': epoch,
                    'train_loss': max(0, train_loss),
                    'val_loss': max(0, val_loss),
                    'train_acc': min(1, max(0, train_acc)),
                    'val_acc': min(1, max(0, val_acc))
                })
            
            # Emit progress update
            socketio.emit('training_progress', {
                'task_id': task_id,
                'epoch': epoch,
                'total_epochs': max_epochs,
                'overall_progress': (epoch / max_epochs) * 100
            })
            
            # Emit metrics update
            socketio.emit('training_metrics', {
                'task_id': task_id,
                'train_loss': max(0, train_loss),
                'val_loss': max(0, val_loss),
                'train_acc': min(1, max(0, train_acc)),
                'val_acc': min(1, max(0, val_acc)),
                'learning_rate': config.get('learning_rate', 0.001)
            })
            
            if epoch % 10 == 0:
                log_training_message(task_id, 'info', f'Epoch {epoch}/{max_epochs} completed')
        
        # Training completed
        final_acc = training_state['active_tasks'][task_id]['val_acc']
        final_loss = training_state['active_tasks'][task_id]['val_loss']
        
        # Update database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE training_tasks 
            SET status = 'completed', end_time = ?, best_accuracy = ?, final_loss = ?
            WHERE id = ?
        ''', (datetime.now(), final_acc, final_loss, task_id))
        conn.commit()
        conn.close()
        
        # Update training state
        training_state['active_tasks'][task_id]['status'] = 'completed'
        
        log_training_message(task_id, 'success', f'Training completed successfully')
        
        socketio.emit('training_complete', {
            'task_id': task_id,
            'final_metrics': {
                'accuracy': final_acc,
                'loss': final_loss
            }
        })
        
    except Exception as e:
        log_training_message(task_id, 'error', f'Training failed: {str(e)}')
        
        # Update database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            'UPDATE training_tasks SET status = "failed", end_time = ? WHERE id = ?',
            (datetime.now(), task_id)
        )
        conn.commit()
        conn.close()
        
        # Update training state
        if task_id in training_state['active_tasks']:
            training_state['active_tasks'][task_id]['status'] = 'failed'
        
        socketio.emit('training_error', {
            'task_id': task_id,
            'error': str(e)
        })

# Static File Serving Routes

@app.route('/')
def serve_index():
    """Serve the main index.html page"""
    try:
        return send_from_directory(app.config['WEB_FOLDER'], 'index.html')
    except Exception as e:
        return jsonify({'error': f'Failed to serve index.html: {str(e)}'}), 500

@app.route('/<path:filename>')
def serve_static_files(filename):
    """Serve static files (HTML, CSS, JS, images, etc.) from the web directory"""
    try:
        # Security check: prevent directory traversal
        if '..' in filename or filename.startswith('/'):
            return jsonify({'error': 'Invalid file path'}), 400
        
        # Get the full file path
        file_path = os.path.join(app.config['WEB_FOLDER'], filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        # Get the directory and filename
        directory = os.path.dirname(file_path)
        basename = os.path.basename(file_path)
        
        # Set proper MIME type
        mimetype = mimetypes.guess_type(filename)[0]
        if mimetype is None:
            # Default MIME types for common file extensions
            ext = os.path.splitext(filename)[1].lower()
            if ext == '.html':
                mimetype = 'text/html'
            elif ext == '.css':
                mimetype = 'text/css'
            elif ext == '.js':
                mimetype = 'application/javascript'
            elif ext == '.json':
                mimetype = 'application/json'
            elif ext in ['.png', '.jpg', '.jpeg', '.gif', '.svg']:
                mimetype = f'image/{ext[1:]}'
            else:
                mimetype = 'application/octet-stream'
        
        return send_from_directory(directory, basename, mimetype=mimetype)
        
    except Exception as e:
        return jsonify({'error': f'Failed to serve file: {str(e)}'}), 500

# API Routes

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

# Training API Routes

@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start a new training task"""
    try:
        config = request.get_json()
        
        # Validate required fields
        required_fields = ['model', 'dataset', 'learning_rate', 'batch_size', 'max_iter']
        for field in required_fields:
            if field not in config:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Generate unique task ID
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        # Save to database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO training_tasks (id, model, dataset, config, status, start_time)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (task_id, config['model'], config['dataset'], json.dumps(config), 'running', datetime.now()))
        conn.commit()
        conn.close()
        
        # Initialize training state
        training_state['active_tasks'][task_id] = {
            'status': 'running',
            'epoch': 0,
            'train_loss': 0,
            'val_loss': 0,
            'train_acc': 0,
            'val_acc': 0,
            'config': config
        }
        
        # Start training thread
        training_thread = threading.Thread(
            target=mock_training_process,
            args=(task_id, config)
        )
        training_thread.start()
        training_state['task_threads'][task_id] = training_thread
        
        return jsonify({'task_id': task_id, 'status': 'started'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/status/<task_id>', methods=['GET'])
def get_training_status(task_id):
    """Get the status of a training task"""
    if task_id in training_state['active_tasks']:
        return jsonify(training_state['active_tasks'][task_id])
    
    # Check database for completed/failed tasks
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM training_tasks WHERE id = ?', (task_id,))
    task = cursor.fetchone()
    conn.close()
    
    if task:
        return jsonify(dict(task))
    else:
        return jsonify({'error': 'Task not found'}), 404

@app.route('/api/training/stop/<task_id>', methods=['POST'])
def stop_training(task_id):
    """Stop a running training task"""
    if task_id in training_state['active_tasks']:
        training_state['active_tasks'][task_id]['status'] = 'stopped'
        
        # Update database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            'UPDATE training_tasks SET status = "stopped", end_time = ? WHERE id = ?',
            (datetime.now(), task_id)
        )
        conn.commit()
        conn.close()
        
        log_training_message(task_id, 'warning', 'Training stopped by user')
        
        return jsonify({'status': 'stopped'})
    else:
        return jsonify({'error': 'Task not found or not running'}), 404

@app.route('/api/training/history', methods=['GET'])
def get_training_history():
    """Get training history"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM training_tasks 
        ORDER BY start_time DESC 
        LIMIT 50
    ''')
    tasks = cursor.fetchall()
    conn.close()
    
    return jsonify([dict(task) for task in tasks])

@app.route('/api/training/logs/<task_id>', methods=['GET'])
def get_training_logs(task_id):
    """Get training logs for a specific task"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * FROM training_logs 
        WHERE task_id = ? 
        ORDER BY timestamp DESC 
        LIMIT 100
    ''', (task_id,))
    logs = cursor.fetchall()
    conn.close()
    
    return jsonify([dict(log) for log in logs])

@app.route('/api/training/details/<task_id>', methods=['GET'])
def get_training_details(task_id):
    """Get detailed information about a training task"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM training_tasks WHERE id = ?', (task_id,))
    task = cursor.fetchone()
    
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    
    # Get logs
    cursor.execute('''
        SELECT * FROM training_logs 
        WHERE task_id = ? 
        ORDER BY timestamp DESC 
        LIMIT 50
    ''', (task_id,))
    logs = cursor.fetchall()
    conn.close()
    
    task_dict = dict(task)
    task_dict['config'] = json.loads(task_dict['config'])
    task_dict['logs'] = [dict(log) for log in logs]
    
    # Add mock metrics if completed
    if task_dict['status'] == 'completed':
        task_dict['metrics'] = {
            'best_accuracy': task_dict['best_accuracy'],
            'final_loss': task_dict['final_loss'],
            'auc': 0.924,
            'f1_score': 0.841
        }
    
    return jsonify(task_dict)

# Dataset API Routes

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    """Get all datasets"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM datasets ORDER BY created_at DESC')
    datasets = cursor.fetchall()
    conn.close()
    
    return jsonify({'datasets': [dict(dataset) for dataset in datasets]})

@app.route('/api/datasets/upload', methods=['POST'])
def upload_dataset():
    """Upload a new dataset"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
        
        # Get form data
        name = request.form.get('name', '')
        version = request.form.get('version', '')
        architecture = request.form.get('architecture', '')
        optimization = request.form.get('optimization', '')
        description = request.form.get('description', '')
        
        if not all([name, architecture, optimization]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Generate dataset ID
        dataset_id = f"{name}_{version}_{architecture}_{optimization}".replace('.', '_')
        
        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{dataset_id}_{filename}")
        file.save(file_path)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        # Mock analysis for sample counts
        samples = np.random.randint(5000, 25000)
        positive_samples = int(samples * (0.45 + np.random.random() * 0.1))
        negative_samples = samples - positive_samples
        
        # Save to database
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO datasets (
                id, name, version, architecture, optimization, description,
                file_path, samples, size_bytes, positive_samples, negative_samples,
                created_at, modified_at, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            dataset_id, name, version, architecture, optimization, description,
            file_path, samples, file_size, positive_samples, negative_samples,
            datetime.now(), datetime.now(), 'ready'
        ))
        conn.commit()
        conn.close()
        
        return jsonify({
            'dataset_id': dataset_id,
            'status': 'uploaded',
            'samples': samples,
            'size': file_size
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/datasets/<dataset_id>/info', methods=['GET'])
def get_dataset_info(dataset_id):
    """Get detailed information about a dataset"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM datasets WHERE id = ?', (dataset_id,))
    dataset = cursor.fetchone()
    conn.close()
    
    if not dataset:
        return jsonify({'error': 'Dataset not found'}), 404
    
    return jsonify(dict(dataset))

@app.route('/api/datasets/<dataset_id>/preprocess', methods=['POST'])
def preprocess_dataset(dataset_id):
    """Start preprocessing for a dataset"""
    try:
        config = request.get_json()
        
        # Update dataset status
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            'UPDATE datasets SET status = "processing", modified_at = ? WHERE id = ?',
            (datetime.now(), dataset_id)
        )
        conn.commit()
        conn.close()
        
        # Mock preprocessing (in real implementation, start async process)
        def mock_preprocessing():
            time.sleep(5)  # Simulate processing time
            
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE datasets SET status = "ready", modified_at = ? WHERE id = ?',
                (datetime.now(), dataset_id)
            )
            conn.commit()
            conn.close()
        
        threading.Thread(target=mock_preprocessing).start()
        
        return jsonify({'status': 'preprocessing_started'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/datasets/<dataset_id>/download', methods=['GET'])
def download_dataset(dataset_id):
    """Download a dataset"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT file_path FROM datasets WHERE id = ?', (dataset_id,))
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        return jsonify({'error': 'Dataset not found'}), 404
    
    file_path = result['file_path']
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(file_path, as_attachment=True)

@app.route('/api/datasets/<dataset_id>', methods=['DELETE'])
def delete_dataset(dataset_id):
    """Delete a dataset"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get file path before deletion
        cursor.execute('SELECT file_path FROM datasets WHERE id = ?', (dataset_id,))
        result = cursor.fetchone()
        
        if result:
            file_path = result['file_path']
            
            # Delete from database
            cursor.execute('DELETE FROM datasets WHERE id = ?', (dataset_id,))
            conn.commit()
            
            # Delete file
            if os.path.exists(file_path):
                os.remove(file_path)
        
        conn.close()
        
        return jsonify({'status': 'deleted'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# WebSocket Events

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    print(f'Client connected: {request.sid}')
    emit('connected', {'message': 'Connected to BVC training server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    print(f'Client disconnected: {request.sid}')

@socketio.on('subscribe_training')
def handle_subscribe_training(data):
    """Subscribe to training updates for a specific task"""
    task_id = data.get('task_id')
    if task_id and task_id in training_state['active_tasks']:
        # Send current status
        emit('training_status', training_state['active_tasks'][task_id])

# Error handlers

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({'error': 'File too large. Maximum size is 500MB.'}), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Main execution
if __name__ == '__main__':
    init_db()
    print("BVC Vulnerability Detection System API Server")
    print("Server starting on http://localhost:5000")
    print("WebSocket available at ws://localhost:5000")
    
    # Add some mock datasets if database is empty
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM datasets')
    count = cursor.fetchone()[0]
    
    if count == 0:
        mock_datasets = [
            ('openssl_1.0.1f_arm_O2', 'openssl', '1.0.1f', 'arm', 'O2', 'OpenSSL库的ARM架构O2优化版本数据集', 15420, 245700000, 7850, 7570),
            ('libsqlite3_0.8.6_x86_O1', 'libsqlite3', '0.8.6', 'x86', 'O1', 'SQLite数据库引擎x86架构O1优化数据集', 8930, 156200000, 4420, 4510),
            ('coreutils_6.12_mips_O3', 'coreutils', '6.12', 'mips', 'O3', 'GNU核心工具集MIPS架构O3优化数据集', 12680, 198500000, 6340, 6340),
            ('nginx_1.18.0_x64_O2', 'nginx', '1.18.0', 'x64', 'O2', 'Nginx Web服务器x64架构O2优化数据集', 22150, 387300000, 11200, 10950)
        ]
        
        for dataset in mock_datasets:
            cursor.execute('''
                INSERT INTO datasets (
                    id, name, version, architecture, optimization, description,
                    file_path, samples, size_bytes, positive_samples, negative_samples,
                    created_at, modified_at, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', dataset + ('mock_path', datetime.now(), datetime.now(), 'ready'))
        
        conn.commit()
    
    conn.close()
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)