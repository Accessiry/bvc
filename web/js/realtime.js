// Real-time communication and data updates
class RealTimeManager {
    constructor() {
        this.socket = null;
        this.pollingInterval = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000; // Start with 1 second
        
        this.eventHandlers = new Map();
        this.messageQueue = [];
        
        this.initializeConnection();
    }

    initializeConnection() {
        this.connectSocketIO();
    }

    connectSocketIO() {
        try {
            // Check if socket.io is available
            if (typeof io === 'undefined') {
                console.error('Socket.IO client library not found, falling back to polling');
                this.fallbackToPolling();
                return;
            }

            // Create socket.io connection
            this.socket = io({
                transports: ['websocket', 'polling'],
                reconnection: true,
                reconnectionAttempts: this.maxReconnectAttempts,
                reconnectionDelay: this.reconnectDelay
            });
            
            this.socket.on('connect', () => {
                this.onConnectionOpen();
            });
            
            this.socket.on('disconnect', (reason) => {
                this.onConnectionClose({ reason });
            });
            
            this.socket.on('connect_error', (error) => {
                this.onConnectionError(error);
            });

            // Set up training-specific event listeners
            this.setupTrainingEvents();
            
        } catch (error) {
            console.error('Socket.IO connection failed:', error);
            this.fallbackToPolling();
        }
    }

    setupTrainingEvents() {
        if (!this.socket) return;

        // Listen for training events from the server
        this.socket.on('training_progress', (data) => {
            this.emit('training_progress', data);
        });

        this.socket.on('training_metrics', (data) => {
            this.emit('training_metrics', data);
        });

        this.socket.on('training_log', (data) => {
            this.emit('training_log', data);
        });

        this.socket.on('training_complete', (data) => {
            this.emit('training_complete', data);
        });

        this.socket.on('training_error', (data) => {
            this.emit('training_error', data);
        });

        this.socket.on('connected', (data) => {
            console.log('Server connection confirmed:', data);
        });
    }

    onConnectionOpen() {
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.reconnectDelay = 1000;
        
        this.updateConnectionStatus('connected');
        this.emit('connection', { status: 'connected' });
        
        // Send any queued messages
        this.flushMessageQueue();
        
        console.log('WebSocket connection established');
    }

    onConnectionClose(event) {
        this.isConnected = false;
        this.updateConnectionStatus('disconnected');
        
        if (event.wasClean) {
            this.emit('connection', { status: 'disconnected', reason: 'clean_close' });
        } else {
            this.emit('connection', { status: 'disconnected', reason: 'unexpected_close' });
            this.attemptReconnect();
        }
        
        console.log('WebSocket connection closed:', event.code, event.reason);
    }

    onConnectionError(error) {
        console.error('WebSocket error:', error);
        this.emit('connection', { status: 'error', error });
        
        if (!this.isConnected) {
            this.fallbackToPolling();
        }
    }

    attemptReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.log('Max reconnection attempts reached, falling back to polling');
            this.fallbackToPolling();
            return;
        }
        
        this.reconnectAttempts++;
        this.updateConnectionStatus('connecting');
        
        console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
        
        setTimeout(() => {
            this.connectSocketIO();
        }, this.reconnectDelay);
        
        // Exponential backoff
        this.reconnectDelay = Math.min(this.reconnectDelay * 2, 30000);
    }

    fallbackToPolling() {
        console.log('Falling back to polling mode');
        this.updateConnectionStatus('polling');
        
        // Clear any existing polling interval
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
        }
        
        // Start polling for updates
        this.pollingInterval = setInterval(() => {
            this.pollForUpdates();
        }, 3000); // Poll every 3 seconds
        
        this.emit('connection', { status: 'polling' });
    }

    async pollForUpdates() {
        try {
            // Poll training status if there's an active task
            const activeTask = this.getActiveTrainingTask();
            if (activeTask) {
                const response = await BVC.API.get(`/training/status/${activeTask}`);
                this.handleMessage(JSON.stringify({
                    type: 'training_progress',
                    payload: response
                }));
            }
            
            // Poll for new log entries
            const logs = await BVC.API.get('/training/logs/recent');
            logs.forEach(log => {
                this.handleMessage(JSON.stringify({
                    type: 'training_log',
                    payload: log
                }));
            });
            
        } catch (error) {
            console.error('Polling failed:', error);
        }
    }

    getActiveTrainingTask() {
        // This should be managed by the training controller
        return window.trainingController?.currentTask || null;
    }

    handleMessage(data) {
        try {
            // For socket.io, data is already parsed as an object
            const message = typeof data === 'string' ? JSON.parse(data) : data;
            this.emit('message', message);
            
            // Emit specific event based on message type
            if (message.type) {
                this.emit(message.type, message.payload || message);
            }
            
        } catch (error) {
            console.error('Failed to parse message:', error);
        }
    }

    sendMessage(message) {
        if (this.isConnected && this.socket && this.socket.connected) {
            // Use socket.io emit instead of WebSocket send
            this.socket.emit('message', message);
        } else {
            // Queue message for later sending
            this.messageQueue.push(message);
            
            // For HTTP polling mode, send immediately via API
            if (this.pollingInterval) {
                this.sendViaAPI(message);
            }
        }
    }

    async sendViaAPI(message) {
        try {
            await BVC.API.post('/training/command', message);
        } catch (error) {
            console.error('Failed to send command via API:', error);
        }
    }

    flushMessageQueue() {
        while (this.messageQueue.length > 0) {
            const message = this.messageQueue.shift();
            this.sendMessage(message);
        }
    }

    // Event system
    on(event, handler) {
        if (!this.eventHandlers.has(event)) {
            this.eventHandlers.set(event, []);
        }
        this.eventHandlers.get(event).push(handler);
    }

    off(event, handler) {
        if (this.eventHandlers.has(event)) {
            const handlers = this.eventHandlers.get(event);
            const index = handlers.indexOf(handler);
            if (index > -1) {
                handlers.splice(index, 1);
            }
        }
    }

    emit(event, data) {
        if (this.eventHandlers.has(event)) {
            this.eventHandlers.get(event).forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`Error in event handler for ${event}:`, error);
                }
            });
        }
    }

    updateConnectionStatus(status) {
        const statusEl = document.querySelector('.connection-status') || this.createConnectionStatusElement();
        statusEl.className = `connection-status ${status}`;
        
        const statusText = {
            'connected': '实时连接',
            'connecting': '连接中...',
            'disconnected': '连接断开',
            'polling': '轮询模式',
            'error': '连接错误'
        };
        
        statusEl.textContent = statusText[status] || status;
    }

    createConnectionStatusElement() {
        const statusEl = document.createElement('div');
        statusEl.className = 'connection-status';
        document.body.appendChild(statusEl);
        return statusEl;
    }

    // Training-specific methods
    startTrainingMonitoring(taskId) {
        if (this.socket && this.socket.connected) {
            this.socket.emit('subscribe_training', { task_id: taskId });
        } else {
            this.sendMessage({
                type: 'subscribe_training',
                task_id: taskId
            });
        }
    }

    stopTrainingMonitoring(taskId) {
        if (this.socket && this.socket.connected) {
            this.socket.emit('unsubscribe_training', { task_id: taskId });
        } else {
            this.sendMessage({
                type: 'unsubscribe_training',
                task_id: taskId
            });
        }
    }

    pauseTraining(taskId) {
        if (this.socket && this.socket.connected) {
            this.socket.emit('pause_training', { task_id: taskId });
        } else {
            this.sendMessage({
                type: 'pause_training',
                task_id: taskId
            });
        }
    }

    resumeTraining(taskId) {
        if (this.socket && this.socket.connected) {
            this.socket.emit('resume_training', { task_id: taskId });
        } else {
            this.sendMessage({
                type: 'resume_training',
                task_id: taskId
            });
        }
    }

    stopTraining(taskId) {
        if (this.socket && this.socket.connected) {
            this.socket.emit('stop_training', { task_id: taskId });
        } else {
            this.sendMessage({
                type: 'stop_training',
                task_id: taskId
            });
        }
    }

    // Cleanup
    disconnect() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
        }
        
        if (this.socket) {
            this.socket.disconnect();
            this.socket = null;
        }
        
        this.isConnected = false;
        this.eventHandlers.clear();
        this.messageQueue = [];
    }

    // Heartbeat mechanism
    startHeartbeat() {
        this.heartbeatInterval = setInterval(() => {
            if (this.isConnected && this.socket && this.socket.connected) {
                this.socket.emit('ping');
            }
        }, 30000); // Ping every 30 seconds
    }

    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
    }
}

// Training data buffer for smooth real-time updates
class TrainingDataBuffer {
    constructor(maxSize = 1000) {
        this.maxSize = maxSize;
        this.buffer = {
            loss: [],
            accuracy: [],
            metrics: [],
            logs: []
        };
    }

    addLossData(epoch, trainLoss, valLoss) {
        const entry = {
            epoch,
            timestamp: Date.now(),
            train: trainLoss,
            validation: valLoss
        };
        
        this.buffer.loss.push(entry);
        this.trimBuffer('loss');
        
        return entry;
    }

    addAccuracyData(epoch, trainAcc, valAcc) {
        const entry = {
            epoch,
            timestamp: Date.now(),
            train: trainAcc,
            validation: valAcc
        };
        
        this.buffer.accuracy.push(entry);
        this.trimBuffer('accuracy');
        
        return entry;
    }

    addMetrics(metrics) {
        const entry = {
            timestamp: Date.now(),
            ...metrics
        };
        
        this.buffer.metrics.push(entry);
        this.trimBuffer('metrics');
        
        return entry;
    }

    addLog(message, level = 'info') {
        const entry = {
            timestamp: Date.now(),
            message,
            level
        };
        
        this.buffer.logs.push(entry);
        this.trimBuffer('logs');
        
        return entry;
    }

    trimBuffer(type) {
        if (this.buffer[type].length > this.maxSize) {
            this.buffer[type] = this.buffer[type].slice(-this.maxSize);
        }
    }

    getLatest(type, count = 1) {
        const data = this.buffer[type];
        return data.slice(-count);
    }

    getAll(type) {
        return [...this.buffer[type]];
    }

    clear(type) {
        if (type) {
            this.buffer[type] = [];
        } else {
            Object.keys(this.buffer).forEach(key => {
                this.buffer[key] = [];
            });
        }
    }

    export(type) {
        return {
            type,
            data: this.getAll(type),
            exported_at: new Date().toISOString()
        };
    }
}

// Performance monitor for real-time updates
class PerformanceMonitor {
    constructor() {
        this.metrics = {
            messageCount: 0,
            lastMessageTime: null,
            averageLatency: 0,
            latencyHistory: [],
            frameRate: 0,
            lastFrameTime: performance.now()
        };
        
        this.startMonitoring();
    }

    startMonitoring() {
        // Monitor frame rate
        const measureFrameRate = () => {
            const now = performance.now();
            const delta = now - this.metrics.lastFrameTime;
            this.metrics.frameRate = 1000 / delta;
            this.metrics.lastFrameTime = now;
            
            requestAnimationFrame(measureFrameRate);
        };
        
        requestAnimationFrame(measureFrameRate);
        
        // Log performance metrics every 10 seconds
        setInterval(() => {
            this.logMetrics();
        }, 10000);
    }

    recordMessage(latency) {
        this.metrics.messageCount++;
        this.metrics.lastMessageTime = Date.now();
        
        if (latency !== undefined) {
            this.metrics.latencyHistory.push(latency);
            if (this.metrics.latencyHistory.length > 100) {
                this.metrics.latencyHistory.shift();
            }
            
            this.metrics.averageLatency = this.metrics.latencyHistory.reduce((a, b) => a + b, 0) / this.metrics.latencyHistory.length;
        }
    }

    logMetrics() {
        console.log('Real-time Performance Metrics:', {
            messagesReceived: this.metrics.messageCount,
            averageLatency: Math.round(this.metrics.averageLatency),
            frameRate: Math.round(this.metrics.frameRate),
            lastMessageTime: this.metrics.lastMessageTime ? new Date(this.metrics.lastMessageTime).toLocaleTimeString() : 'None'
        });
    }

    getMetrics() {
        return { ...this.metrics };
    }
}

// Initialize real-time manager when needed
let realTimeManager = null;
let dataBuffer = null;
let perfMonitor = null;

function initializeRealTime() {
    if (!realTimeManager) {
        realTimeManager = new RealTimeManager();
        dataBuffer = new TrainingDataBuffer();
        perfMonitor = new PerformanceMonitor();
        
        // Set up event handlers
        realTimeManager.on('training_progress', (data) => {
            perfMonitor.recordMessage();
            // Handle training progress updates
            if (window.trainingController) {
                window.trainingController.updateTrainingProgress(data);
            }
        });
        
        realTimeManager.on('training_metrics', (data) => {
            perfMonitor.recordMessage();
            dataBuffer.addMetrics(data);
            
            // Handle metrics updates
            if (window.trainingController) {
                window.trainingController.updateMetrics(data);
            }
        });
        
        realTimeManager.on('training_log', (data) => {
            perfMonitor.recordMessage();
            dataBuffer.addLog(data.message, data.level);
            
            // Handle log updates
            if (window.trainingController) {
                window.trainingController.addLog(data.message, data.level);
            }
        });
        
        console.log('Real-time manager initialized');
    }
    
    return realTimeManager;
}

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    if (realTimeManager) {
        realTimeManager.disconnect();
    }
});

// Export for use in other modules
window.RealTimeManager = RealTimeManager;
window.TrainingDataBuffer = TrainingDataBuffer;
window.PerformanceMonitor = PerformanceMonitor;
window.initializeRealTime = initializeRealTime;