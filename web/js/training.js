// Training process control and visualization
class TrainingController {
    constructor() {
        this.isTraining = false;
        this.isPaused = false;
        this.currentTask = null;
        this.startTime = null;
        this.chartManager = new ChartManager();
        this.wsConnection = null;
        
        this.initializeElements();
        this.setupEventListeners();
        this.connectWebSocket();
        this.loadTrainingHistory();
    }

    initializeElements() {
        // Control buttons
        this.startBtn = document.getElementById('startTraining');
        this.pauseBtn = document.getElementById('pauseTraining');
        this.stopBtn = document.getElementById('stopTraining');
        
        // Form elements
        this.form = document.getElementById('trainingForm');
        this.modelSelect = document.getElementById('modelSelect');
        this.datasetSelect = document.getElementById('datasetSelect');
        
        // Progress elements
        this.epochProgress = document.getElementById('epochProgress');
        this.totalProgress = document.getElementById('totalProgress');
        this.currentEpoch = document.getElementById('currentEpoch');
        this.overallProgress = document.getElementById('overallProgress');
        this.elapsedTime = document.getElementById('elapsedTime');
        this.estimatedTime = document.getElementById('estimatedTime');
        
        // Metric elements
        this.trainLoss = document.getElementById('currentTrainLoss');
        this.valLoss = document.getElementById('currentValLoss');
        this.trainAcc = document.getElementById('currentTrainAcc');
        this.valAcc = document.getElementById('currentValAcc');
        this.currentLR = document.getElementById('currentLR');
        
        // Status and logs
        this.status = document.getElementById('trainingStatus');
        this.logs = document.getElementById('trainingLogs');
        this.historyTable = document.getElementById('trainingHistoryTable');
    }

    setupEventListeners() {
        // Training control buttons
        this.startBtn.addEventListener('click', () => this.startTraining());
        this.pauseBtn.addEventListener('click', () => this.pauseTraining());
        this.stopBtn.addEventListener('click', () => this.stopTraining());
        
        // Form validation
        this.form.addEventListener('input', () => this.validateForm());
        
        // Clear logs button
        document.getElementById('clearLogs').addEventListener('click', () => this.clearLogs());
        
        // Refresh history button
        document.getElementById('refreshHistory').addEventListener('click', () => this.loadTrainingHistory());
        
        // Model selection change
        this.modelSelect.addEventListener('change', () => this.onModelChange());
    }

    connectWebSocket() {
        // Check if Socket.IO is available, otherwise fall back to WebSocket
        if (typeof io !== 'undefined') {
            this.connectSocketIO();
        } else {
            this.connectWebSocketDirect();
        }
    }

    connectSocketIO() {
        this.updateConnectionStatus('connecting');
        this.addLog('正在连接Socket.IO服务器...', 'info');
        
        try {
            // Connect using Socket.IO client with standard path
            this.wsConnection = io({
                transports: ['websocket', 'polling']
            });
            
            this.wsConnection.on('connect', () => {
                this.updateConnectionStatus('connected');
                this.addLog('Socket.IO连接已建立', 'success');
                this.clearConnectionRetries();
            });
            
            this.wsConnection.on('training_progress', (data) => {
                this.updateTrainingProgress(data);
            });
            
            this.wsConnection.on('training_metrics', (data) => {
                this.updateMetrics(data);
            });
            
            this.wsConnection.on('training_log', (data) => {
                this.addLog(data.message, data.level);
            });
            
            this.wsConnection.on('training_status', (data) => {
                this.updateTrainingStatus(data.status);
            });
            
            this.wsConnection.on('training_complete', (data) => {
                this.onTrainingComplete(data);
            });
            
            this.wsConnection.on('training_error', (data) => {
                this.onTrainingError(data);
            });
            
            this.wsConnection.on('disconnect', (reason) => {
                this.updateConnectionStatus('disconnected');
                this.addLog(`Socket.IO连接断开: ${reason}`, 'warning');
                if (reason === 'io server disconnect') {
                    // Server initiated disconnect, try to reconnect
                    this.scheduleReconnect();
                }
            });
            
            this.wsConnection.on('connect_error', (error) => {
                this.updateConnectionStatus('disconnected');
                this.addLog(`Socket.IO连接错误: ${error.message}`, 'error');
                this.scheduleReconnect();
            });
            
        } catch (error) {
            this.updateConnectionStatus('disconnected');
            this.addLog(`无法创建Socket.IO连接: ${error.message}，尝试WebSocket`, 'warning');
            this.connectWebSocketDirect();
        }
    }

    connectWebSocketDirect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/training`;
        
        this.updateConnectionStatus('connecting');
        this.addLog('正在连接WebSocket服务器...', 'info');
        
        try {
            this.wsConnection = new WebSocket(wsUrl);
            
            this.wsConnection.onopen = () => {
                this.updateConnectionStatus('connected');
                this.addLog('WebSocket连接已建立', 'success');
                this.clearConnectionRetries();
            };
            
            this.wsConnection.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    this.addLog(`WebSocket消息解析错误: ${error.message}`, 'error');
                }
            };
            
            this.wsConnection.onclose = (event) => {
                this.updateConnectionStatus('disconnected');
                if (event.wasClean) {
                    this.addLog('WebSocket连接正常关闭', 'info');
                } else {
                    this.addLog(`WebSocket连接意外断开 (code: ${event.code}, reason: ${event.reason || 'unknown'})`, 'warning');
                    this.scheduleReconnect();
                }
            };
            
            this.wsConnection.onerror = (error) => {
                this.updateConnectionStatus('disconnected');
                this.addLog('WebSocket连接错误，可能是服务器未启动或路径不正确', 'error');
                this.scheduleReconnect();
            };
        } catch (error) {
            this.updateConnectionStatus('disconnected');
            this.addLog(`无法创建WebSocket连接: ${error.message}，使用轮询模式`, 'warning');
            this.setupPolling();
        }
    }

    clearConnectionRetries() {
        if (this.reconnectTimeout) {
            clearTimeout(this.reconnectTimeout);
            this.reconnectTimeout = null;
        }
        this.reconnectAttempts = 0;
    }

    scheduleReconnect() {
        // Don't reconnect if we're not training
        if (!this.isTraining) {
            this.addLog('未在训练中，跳过WebSocket重连', 'info');
            return;
        }

        // Clear any existing reconnect timeout
        this.clearConnectionRetries();
        
        // Exponential backoff with max 30 seconds
        this.reconnectAttempts = (this.reconnectAttempts || 0) + 1;
        const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts - 1), 30000);
        
        this.addLog(`${delay / 1000}秒后尝试重新连接WebSocket (第${this.reconnectAttempts}次尝试)`, 'info');
        
        this.reconnectTimeout = setTimeout(() => {
            if (this.reconnectAttempts <= 5) {  // Max 5 attempts
                this.connectWebSocket();
            } else {
                this.addLog('WebSocket重连失败，切换到轮询模式', 'warning');
                this.setupPolling();
            }
        }, delay);
    }

    updateConnectionStatus(status) {
        const statusEl = document.querySelector('.connection-status') || this.createConnectionStatusElement();
        statusEl.className = `connection-status ${status}`;
        
        const statusText = {
            'connected': '已连接',
            'connecting': '连接中...',
            'disconnected': '已断开'
        };
        
        statusEl.textContent = statusText[status] || status;
    }

    createConnectionStatusElement() {
        const statusEl = document.createElement('div');
        statusEl.className = 'connection-status';
        document.body.appendChild(statusEl);
        return statusEl;
    }

    handleWebSocketMessage(data) {
        // Handle different message formats
        // Server can send messages in different formats:
        // 1. Direct event data: { type: 'event_name', payload: {...} }
        // 2. Socket.IO format: just the payload directly
        
        let messageType, payload;
        
        if (data.type && data.payload) {
            // Format 1: { type: 'event_name', payload: {...} }
            messageType = data.type;
            payload = data.payload;
        } else if (typeof data === 'object') {
            // Format 2: Direct payload, try to infer type from content
            payload = data;
            
            if (data.task_id && data.epoch !== undefined) {
                messageType = 'training_progress';
            } else if (data.task_id && data.train_loss !== undefined) {
                messageType = 'training_metrics';
            } else if (data.task_id && data.level && data.message) {
                messageType = 'training_log';
            } else if (data.status) {
                messageType = 'training_status';
            } else if (data.task_id && data.final_metrics) {
                messageType = 'training_complete';
            } else if (data.task_id && data.error) {
                messageType = 'training_error';
            } else {
                this.addLog(`收到未知格式的WebSocket消息: ${JSON.stringify(data)}`, 'warning');
                return;
            }
        } else {
            this.addLog(`无法处理WebSocket消息: ${JSON.stringify(data)}`, 'error');
            return;
        }
        
        switch (messageType) {
            case 'training_progress':
                this.updateTrainingProgress(payload);
                break;
            case 'training_metrics':
                this.updateMetrics(payload);
                break;
            case 'training_log':
                this.addLog(payload.message, payload.level);
                break;
            case 'training_status':
                this.updateTrainingStatus(payload.status);
                break;
            case 'training_complete':
                this.onTrainingComplete(payload);
                break;
            case 'training_error':
                this.onTrainingError(payload);
                break;
            default:
                this.addLog(`收到未知类型的WebSocket消息: ${messageType}`, 'warning');
        }
    }

    setupPolling() {
        // Fallback polling mechanism when WebSocket is not available
        this.pollingInterval = setInterval(() => {
            if (this.isTraining && this.currentTask) {
                this.fetchTrainingStatus();
            }
        }, 2000);
    }

    async fetchTrainingStatus() {
        try {
            const response = await BVC.API.get(`/training/status/${this.currentTask}`);
            this.handleWebSocketMessage({ type: 'training_progress', payload: response });
        } catch (error) {
            console.error('Failed to fetch training status:', error);
        }
    }

    validateForm() {
        const learningRate = parseFloat(document.getElementById('learningRate').value);
        const batchSize = parseInt(document.getElementById('batchSize').value);
        const maxIter = parseInt(document.getElementById('maxIter').value);
        
        let isValid = true;
        
        // Validate learning rate
        if (learningRate <= 0 || learningRate > 1) {
            this.setFieldError('learningRate', '学习率必须在 0 到 1 之间');
            isValid = false;
        } else {
            this.clearFieldError('learningRate');
        }
        
        // Validate batch size
        if (batchSize < 1 || batchSize > 512) {
            this.setFieldError('batchSize', '批次大小必须在 1 到 512 之间');
            isValid = false;
        } else {
            this.clearFieldError('batchSize');
        }
        
        // Validate max iterations
        if (maxIter < 1 || maxIter > 10000) {
            this.setFieldError('maxIter', '迭代次数必须在 1 到 10000 之间');
            isValid = false;
        } else {
            this.clearFieldError('maxIter');
        }
        
        this.startBtn.disabled = !isValid || this.isTraining;
        return isValid;
    }

    setFieldError(fieldId, message) {
        const field = document.getElementById(fieldId);
        field.classList.add('is-invalid');
        
        let feedback = field.parentNode.querySelector('.invalid-feedback');
        if (!feedback) {
            feedback = document.createElement('div');
            feedback.className = 'invalid-feedback';
            field.parentNode.appendChild(feedback);
        }
        feedback.textContent = message;
    }

    clearFieldError(fieldId) {
        const field = document.getElementById(fieldId);
        field.classList.remove('is-invalid');
        
        const feedback = field.parentNode.querySelector('.invalid-feedback');
        if (feedback) {
            feedback.remove();
        }
    }

    async startTraining() {
        if (!this.validateForm()) {
            BVC.Utils.showNotification('请检查并修正表单中的错误', 'danger');
            return;
        }

        const config = this.getTrainingConfig();
        
        try {
            BVC.Utils.showLoading(this.startBtn);
            this.addLog('正在启动训练任务...', 'info');
            
            const response = await BVC.API.post('/training/start', config);
            this.currentTask = response.task_id;
            
            this.isTraining = true;
            this.isPaused = false;
            this.startTime = new Date();
            
            this.updateTrainingStatus('training');
            this.updateControlButtons();
            this.startElapsedTimer();
            
            this.addLog(`训练任务已启动，任务ID: ${this.currentTask}`, 'success');
            BVC.Utils.showNotification('训练任务已成功启动', 'success');
            
        } catch (error) {
            this.addLog(`启动训练失败: ${error.message}`, 'error');
            BVC.Utils.showNotification('启动训练失败', 'danger');
        } finally {
            BVC.Utils.hideLoading(this.startBtn);
        }
    }

    async pauseTraining() {
        if (!this.currentTask) return;
        
        try {
            await BVC.API.post(`/training/pause/${this.currentTask}`);
            this.isPaused = true;
            this.updateTrainingStatus('paused');
            this.updateControlButtons();
            this.addLog('训练已暂停', 'warning');
            
        } catch (error) {
            this.addLog(`暂停训练失败: ${error.message}`, 'error');
        }
    }

    async stopTraining() {
        if (!this.currentTask) return;
        
        if (!confirm('确定要停止当前训练任务吗？')) return;
        
        try {
            await BVC.API.post(`/training/stop/${this.currentTask}`);
            this.onTrainingComplete({ status: 'stopped' });
            this.addLog('训练已停止', 'warning');
            
        } catch (error) {
            this.addLog(`停止训练失败: ${error.message}`, 'error');
        }
    }

    getTrainingConfig() {
        return {
            model: this.modelSelect.value,
            dataset: this.datasetSelect.value,
            learning_rate: parseFloat(document.getElementById('learningRate').value),
            batch_size: parseInt(document.getElementById('batchSize').value),
            max_iter: parseInt(document.getElementById('maxIter').value),
            embedding_size: parseInt(document.getElementById('embeddingSize').value),
            dropout: parseFloat(document.getElementById('dropout').value),
            seed: parseInt(document.getElementById('seed').value),
            early_stopping: document.getElementById('earlyStopping').checked
        };
    }

    updateTrainingProgress(progress) {
        const { epoch, total_epochs, batch, total_batches, overall_progress } = progress;
        
        // Update epoch progress
        const epochPercent = (batch / total_batches) * 100;
        this.epochProgress.style.width = `${epochPercent}%`;
        this.currentEpoch.textContent = `${epoch}/${total_epochs}`;
        
        // Update overall progress
        this.totalProgress.style.width = `${overall_progress}%`;
        this.overallProgress.textContent = `${Math.round(overall_progress)}%`;
        
        // Update estimated time
        if (this.startTime && overall_progress > 0) {
            const elapsed = (new Date() - this.startTime) / 1000;
            const estimated = (elapsed / overall_progress) * (100 - overall_progress);
            this.estimatedTime.textContent = BVC.Utils.formatDuration(Math.round(estimated));
        }
    }

    updateMetrics(metrics) {
        const { train_loss, val_loss, train_acc, val_acc, learning_rate } = metrics;
        
        // Update metric displays with animation
        this.updateMetricValue(this.trainLoss, train_loss);
        this.updateMetricValue(this.valLoss, val_loss);
        this.updateMetricValue(this.trainAcc, `${(train_acc * 100).toFixed(2)}%`);
        this.updateMetricValue(this.valAcc, `${(val_acc * 100).toFixed(2)}%`);
        this.updateMetricValue(this.currentLR, learning_rate.toExponential(2));
        
        // Update charts
        this.chartManager.addDataPoint('loss', {
            train: train_loss,
            validation: val_loss
        });
        
        this.chartManager.addDataPoint('accuracy', {
            train: train_acc,
            validation: val_acc
        });
    }

    updateMetricValue(element, value) {
        element.textContent = value;
        element.classList.add('updating');
        setTimeout(() => element.classList.remove('updating'), 500);
    }

    updateTrainingStatus(status) {
        this.status.textContent = status;
        this.status.className = `status-indicator ${status}`;
    }

    updateControlButtons() {
        this.startBtn.disabled = this.isTraining;
        this.pauseBtn.disabled = !this.isTraining || this.isPaused;
        this.stopBtn.disabled = !this.isTraining;
    }

    startElapsedTimer() {
        this.elapsedTimer = setInterval(() => {
            if (this.startTime && this.isTraining && !this.isPaused) {
                const elapsed = (new Date() - this.startTime) / 1000;
                this.elapsedTime.textContent = BVC.Utils.formatDuration(Math.round(elapsed));
            }
        }, 1000);
    }

    onTrainingComplete(result) {
        this.isTraining = false;
        this.isPaused = false;
        this.currentTask = null;
        
        clearInterval(this.elapsedTimer);
        
        this.updateTrainingStatus('ready');
        this.updateControlButtons();
        
        this.addLog('训练任务完成', 'success');
        BVC.Utils.showNotification('训练任务完成', 'success');
        
        this.loadTrainingHistory();
    }

    onTrainingError(error) {
        this.isTraining = false;
        this.isPaused = false;
        this.currentTask = null;
        
        clearInterval(this.elapsedTimer);
        
        this.updateTrainingStatus('error');
        this.updateControlButtons();
        
        this.addLog(`训练错误: ${error.message}`, 'error');
        BVC.Utils.showNotification('训练过程中出现错误', 'danger');
    }

    onModelChange() {
        const model = this.modelSelect.value;
        this.addLog(`切换到模型: ${model.toUpperCase()}`, 'info');
        
        // Update default parameters based on model
        const defaults = this.getModelDefaults(model);
        Object.keys(defaults).forEach(key => {
            const element = document.getElementById(key);
            if (element) {
                element.value = defaults[key];
            }
        });
    }

    getModelDefaults(model) {
        const defaults = {
            vulseeker: { learningRate: 0.001, embeddingSize: 128 },
            sense: { learningRate: 0.0001, embeddingSize: 256 },
            fit: { learningRate: 0.002, embeddingSize: 128 },
            safe: { learningRate: 0.0005, embeddingSize: 256 }
        };
        
        return defaults[model] || {};
    }

    addLog(message, level = 'info') {
        const timestamp = new Date().toLocaleString('zh-CN');
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry ${level}`;
        logEntry.innerHTML = `
            <span class="log-time">${timestamp}</span>
            <span class="log-message">${message}</span>
        `;
        
        this.logs.appendChild(logEntry);
        this.logs.scrollTop = this.logs.scrollHeight;
        
        // Limit log entries to prevent performance issues
        const maxLogs = 100;
        while (this.logs.children.length > maxLogs) {
            this.logs.removeChild(this.logs.firstChild);
        }
    }

    clearLogs() {
        this.logs.innerHTML = '';
        this.addLog('日志已清空', 'info');
    }

    async loadTrainingHistory() {
        try {
            const history = await BVC.API.get('/training/history');
            this.renderTrainingHistory(history);
        } catch (error) {
            console.error('Failed to load training history:', error);
            this.addLog('加载训练历史失败', 'error');
        }
    }

    renderTrainingHistory(history) {
        this.historyTable.innerHTML = '';
        
        history.forEach(task => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${task.id}</td>
                <td><span class="badge bg-primary">${task.model.toUpperCase()}</span></td>
                <td>${task.dataset}</td>
                <td><span class="badge bg-${this.getStatusBadgeColor(task.status)}">${task.status}</span></td>
                <td>${new Date(task.start_time).toLocaleString('zh-CN')}</td>
                <td>${BVC.Utils.formatDuration(task.duration)}</td>
                <td>${task.best_accuracy ? (task.best_accuracy * 100).toFixed(1) + '%' : '--'}</td>
                <td>
                    <button class="btn btn-sm btn-outline-primary" onclick="viewTrainingDetails('${task.id}')">
                        <i class="fas fa-eye"></i>
                    </button>
                    ${task.status === 'completed' ? `
                        <button class="btn btn-sm btn-outline-success" onclick="downloadModel('${task.id}')">
                            <i class="fas fa-download"></i>
                        </button>
                    ` : ''}
                    ${task.status === 'running' ? `
                        <button class="btn btn-sm btn-outline-danger" onclick="stopTrainingTask('${task.id}')">
                            <i class="fas fa-stop"></i>
                        </button>
                    ` : ''}
                </td>
            `;
            
            this.historyTable.appendChild(row);
        });
    }

    getStatusBadgeColor(status) {
        const colors = {
            'completed': 'success',
            'running': 'warning',
            'paused': 'info',
            'stopped': 'secondary',
            'failed': 'danger'
        };
        
        return colors[status] || 'secondary';
    }
}

// Global functions for training history actions
window.viewTrainingDetails = async function(taskId) {
    try {
        const details = await BVC.API.get(`/training/details/${taskId}`);
        const modal = new bootstrap.Modal(document.getElementById('trainingDetailsModal'));
        
        document.getElementById('trainingDetailsContent').innerHTML = `
            <div class="training-details">
                <h6>基本信息</h6>
                <table class="table table-sm">
                    <tr><td>任务ID:</td><td>${details.id}</td></tr>
                    <tr><td>模型:</td><td>${details.model}</td></tr>
                    <tr><td>数据集:</td><td>${details.dataset}</td></tr>
                    <tr><td>开始时间:</td><td>${new Date(details.start_time).toLocaleString('zh-CN')}</td></tr>
                    <tr><td>训练时长:</td><td>${BVC.Utils.formatDuration(details.duration)}</td></tr>
                    <tr><td>状态:</td><td><span class="badge bg-primary">${details.status}</span></td></tr>
                </table>
                
                <h6 class="mt-3">训练参数</h6>
                <table class="table table-sm">
                    <tr><td>学习率:</td><td>${details.config.learning_rate}</td></tr>
                    <tr><td>批次大小:</td><td>${details.config.batch_size}</td></tr>
                    <tr><td>最大迭代:</td><td>${details.config.max_iter}</td></tr>
                    <tr><td>嵌入维度:</td><td>${details.config.embedding_size}</td></tr>
                </table>
                
                ${details.metrics ? `
                    <h6 class="mt-3">最终指标</h6>
                    <table class="table table-sm">
                        <tr><td>最佳准确率:</td><td>${(details.metrics.best_accuracy * 100).toFixed(2)}%</td></tr>
                        <tr><td>最终损失:</td><td>${details.metrics.final_loss.toFixed(4)}</td></tr>
                        <tr><td>AUC:</td><td>${details.metrics.auc ? details.metrics.auc.toFixed(3) : '--'}</td></tr>
                        <tr><td>F1 Score:</td><td>${details.metrics.f1_score ? details.metrics.f1_score.toFixed(3) : '--'}</td></tr>
                    </table>
                ` : ''}
            </div>
        `;
        
        modal.show();
    } catch (error) {
        BVC.Utils.showNotification('获取训练详情失败', 'danger');
    }
};

window.downloadModel = async function(taskId) {
    try {
        const response = await fetch(`/api/training/download/${taskId}`);
        const blob = await response.blob();
        
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `model_${taskId}.zip`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        
        BVC.Utils.showNotification('模型下载已开始', 'success');
    } catch (error) {
        BVC.Utils.showNotification('模型下载失败', 'danger');
    }
};

window.stopTrainingTask = async function(taskId) {
    if (!confirm('确定要停止这个训练任务吗？')) return;
    
    try {
        await BVC.API.post(`/training/stop/${taskId}`);
        BVC.Utils.showNotification('训练任务已停止', 'success');
        
        // Refresh history after a short delay
        setTimeout(() => {
            if (window.trainingController) {
                window.trainingController.loadTrainingHistory();
            }
        }, 1000);
    } catch (error) {
        BVC.Utils.showNotification('停止训练任务失败', 'danger');
    }
};

// Initialize training controller when page loads
document.addEventListener('DOMContentLoaded', function() {
    window.trainingController = new TrainingController();
    console.log('Training controller initialized');
});