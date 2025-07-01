// ===== API Functions =====

/**
 * API configuration
 */
const API_CONFIG = {
    baseUrl: '/api',
    timeout: 30000,
    retryAttempts: 3,
    retryDelay: 1000
};

/**
 * Generic API request function
 */
async function apiRequest(endpoint, options = {}) {
    const defaultOptions = {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json',
        },
        timeout: API_CONFIG.timeout
    };
    
    const config = { ...defaultOptions, ...options };
    const url = `${API_CONFIG.baseUrl}${endpoint}`;
    
    try {
        const response = await fetch(url, config);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
            return await response.json();
        } else {
            return await response.text();
        }
    } catch (error) {
        console.error(`API request failed for ${endpoint}:`, error);
        throw error;
    }
}

/**
 * Model Performance API
 */
class ModelAPI {
    static async getModelPerformance(model, architecture = null, optimization = null) {
        const params = new URLSearchParams();
        if (architecture) params.append('arch', architecture);
        if (optimization) params.append('opt', optimization);
        
        const query = params.toString() ? `?${params.toString()}` : '';
        return await apiRequest(`/models/${model}/performance${query}`);
    }
    
    static async getModelComparison(models = ['vulseeker', 'sense', 'fit', 'safe']) {
        const params = new URLSearchParams();
        models.forEach(model => params.append('models', model));
        
        return await apiRequest(`/models/compare?${params.toString()}`);
    }
    
    static async getROCData(model, architecture = null, optimization = null) {
        const params = new URLSearchParams();
        if (architecture) params.append('arch', architecture);
        if (optimization) params.append('opt', optimization);
        
        const query = params.toString() ? `?${params.toString()}` : '';
        return await apiRequest(`/models/${model}/roc${query}`);
    }
    
    static async getModelList() {
        return await apiRequest('/models');
    }
    
    static async getModelDetails(model) {
        return await apiRequest(`/models/${model}`);
    }
}

/**
 * Results API
 */
class ResultsAPI {
    static async getDetectionResults(filters = {}) {
        const params = new URLSearchParams();
        Object.keys(filters).forEach(key => {
            if (filters[key] !== null && filters[key] !== undefined) {
                params.append(key, filters[key]);
            }
        });
        
        const query = params.toString() ? `?${params.toString()}` : '';
        return await apiRequest(`/results${query}`);
    }
    
    static async getResultDetails(resultId) {
        return await apiRequest(`/results/${resultId}`);
    }
    
    static async getVulnerabilityDistribution(program = null, architecture = null) {
        const params = new URLSearchParams();
        if (program) params.append('program', program);
        if (architecture) params.append('arch', architecture);
        
        const query = params.toString() ? `?${params.toString()}` : '';
        return await apiRequest(`/results/vulnerability-distribution${query}`);
    }
    
    static async getConfidenceHeatmap(model, program = null) {
        const params = new URLSearchParams();
        if (program) params.append('program', program);
        
        const query = params.toString() ? `?${params.toString()}` : '';
        return await apiRequest(`/results/${model}/confidence-heatmap${query}`);
    }
    
    static async exportResults(format = 'csv', filters = {}) {
        const params = new URLSearchParams();
        params.append('format', format);
        Object.keys(filters).forEach(key => {
            if (filters[key] !== null && filters[key] !== undefined) {
                params.append(key, filters[key]);
            }
        });
        
        const response = await fetch(`${API_CONFIG.baseUrl}/results/export?${params.toString()}`);
        if (!response.ok) {
            throw new Error(`Export failed: ${response.status}`);
        }
        
        return await response.blob();
    }
}

/**
 * Analysis API
 */
class AnalysisAPI {
    static async uploadFile(file, options = {}) {
        const formData = new FormData();
        formData.append('file', file);
        
        Object.keys(options).forEach(key => {
            formData.append(key, options[key]);
        });
        
        return await apiRequest('/analysis/upload', {
            method: 'POST',
            body: formData,
            headers: {} // Remove Content-Type to let browser set it for FormData
        });
    }
    
    static async getAnalysisStatus(taskId) {
        return await apiRequest(`/analysis/${taskId}/status`);
    }
    
    static async getAnalysisResult(taskId) {
        return await apiRequest(`/analysis/${taskId}/result`);
    }
    
    static async getAnalysisHistory(limit = 50, offset = 0) {
        const params = new URLSearchParams();
        params.append('limit', limit);
        params.append('offset', offset);
        
        return await apiRequest(`/analysis/history?${params.toString()}`);
    }
    
    static async deleteAnalysis(taskId) {
        return await apiRequest(`/analysis/${taskId}`, {
            method: 'DELETE'
        });
    }
}

/**
 * Statistics API
 */
class StatisticsAPI {
    static async getSystemStats() {
        return await apiRequest('/stats/system');
    }
    
    static async getArchitectureStats() {
        return await apiRequest('/stats/architectures');
    }
    
    static async getTimeSeriesData(metric, timeRange = '7d') {
        const params = new URLSearchParams();
        params.append('range', timeRange);
        
        return await apiRequest(`/stats/timeseries/${metric}?${params.toString()}`);
    }
    
    static async getPerformanceTrends(model = null, timeRange = '30d') {
        const params = new URLSearchParams();
        params.append('range', timeRange);
        if (model) params.append('model', model);
        
        return await apiRequest(`/stats/trends?${params.toString()}`);
    }
    
    static async getRealTimeMetrics() {
        return await apiRequest('/stats/realtime');
    }
}

/**
 * Mock Data Service (for development)
 */
class MockDataService {
    static generateMockPerformanceData(model) {
        return {
            model: model,
            accuracy: 0.90 + Math.random() * 0.08,
            precision: 0.88 + Math.random() * 0.10,
            recall: 0.85 + Math.random() * 0.12,
            f1Score: 0.87 + Math.random() * 0.10,
            auc: 0.92 + Math.random() * 0.06,
            architecture: 'arm',
            optimization: 'O2',
            lastUpdated: new Date().toISOString()
        };
    }
    
    static generateMockROCData(points = 100) {
        const data = [];
        for (let i = 0; i <= points; i++) {
            const fpr = i / points;
            const tpr = Math.min(1, fpr + 0.1 + Math.random() * 0.3);
            data.push({ fpr, tpr });
        }
        return data.sort((a, b) => a.fpr - b.fpr);
    }
    
    static generateMockDetectionResults(count = 100) {
        const programs = ['coreutils', 'openssl', 'busybox', 'libsqlite3', 'libgmp'];
        const architectures = ['arm', 'mips', 'x86', 'arm-mips'];
        const optimizations = ['O0', 'O1', 'O2', 'O3'];
        const models = ['vulseeker', 'sense', 'fit', 'safe'];
        
        return Array.from({ length: count }, (_, i) => ({
            id: i + 1,
            program: programs[Math.floor(Math.random() * programs.length)],
            architecture: architectures[Math.floor(Math.random() * architectures.length)],
            optimization: optimizations[Math.floor(Math.random() * optimizations.length)],
            model: models[Math.floor(Math.random() * models.length)],
            confidence: Math.random(),
            isVulnerable: Math.random() > 0.7,
            detectedAt: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString(),
            functionName: `func_${Math.floor(Math.random() * 1000)}`,
            vulnerabilityType: Math.random() > 0.5 ? 'buffer_overflow' : 'use_after_free'
        }));
    }
    
    static generateMockSystemStats() {
        return {
            totalDetections: 12547,
            vulnerabilitiesFound: 1248,
            modelsActive: 4,
            architecturesSupported: 7,
            avgAccuracy: 0.958,
            lastScanTime: new Date().toISOString(),
            systemUptime: 2457600, // seconds
            processingQueue: 23
        };
    }
    
    static async getMockData(type, ...args) {
        // Simulate network delay
        await new Promise(resolve => setTimeout(resolve, 300 + Math.random() * 700));
        
        switch (type) {
            case 'performance':
                return this.generateMockPerformanceData(args[0]);
            case 'roc':
                return this.generateMockROCData(args[0]);
            case 'results':
                return this.generateMockDetectionResults(args[0]);
            case 'stats':
                return this.generateMockSystemStats();
            default:
                throw new Error(`Unknown mock data type: ${type}`);
        }
    }
}

/**
 * Data caching service
 */
class CacheService {
    constructor(ttl = 5 * 60 * 1000) { // 5 minutes default TTL
        this.cache = new Map();
        this.ttl = ttl;
    }
    
    set(key, value) {
        this.cache.set(key, {
            value,
            timestamp: Date.now()
        });
    }
    
    get(key) {
        const item = this.cache.get(key);
        if (!item) return null;
        
        if (Date.now() - item.timestamp > this.ttl) {
            this.cache.delete(key);
            return null;
        }
        
        return item.value;
    }
    
    clear() {
        this.cache.clear();
    }
    
    has(key) {
        const item = this.cache.get(key);
        if (!item) return false;
        
        if (Date.now() - item.timestamp > this.ttl) {
            this.cache.delete(key);
            return false;
        }
        
        return true;
    }
}

// Create global cache instance
const apiCache = new CacheService();

/**
 * Cached API wrapper
 */
class CachedAPI {
    static async getWithCache(endpoint, cacheKey, apiFunction, ...args) {
        // Check cache first
        const cached = apiCache.get(cacheKey);
        if (cached) {
            return cached;
        }
        
        try {
            const result = await apiFunction(...args);
            apiCache.set(cacheKey, result);
            return result;
        } catch (error) {
            console.error(`Cached API request failed for ${cacheKey}:`, error);
            throw error;
        }
    }
    
    static async getModelPerformance(model, architecture = null, optimization = null) {
        const cacheKey = `model_performance_${model}_${architecture}_${optimization}`;
        return await this.getWithCache(
            `/models/${model}/performance`,
            cacheKey,
            ModelAPI.getModelPerformance,
            model,
            architecture,
            optimization
        );
    }
    
    static async getSystemStats() {
        const cacheKey = 'system_stats';
        return await this.getWithCache(
            '/stats/system',
            cacheKey,
            StatisticsAPI.getSystemStats
        );
    }
}

/**
 * WebSocket for real-time updates
 */
class RealTimeAPI {
    constructor() {
        this.ws = null;
        this.listeners = new Map();
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;
    }
    
    connect() {
        try {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected');
                this.reconnectAttempts = 0;
                this.emit('connected');
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.emit(data.type, data.payload);
                } catch (error) {
                    console.error('Failed to parse WebSocket message:', error);
                }
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.emit('disconnected');
                this.reconnect();
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.emit('error', error);
            };
        } catch (error) {
            console.error('Failed to connect WebSocket:', error);
            this.reconnect();
        }
    }
    
    reconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
            
            setTimeout(() => {
                console.log(`Attempting to reconnect WebSocket (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
                this.connect();
            }, delay);
        }
    }
    
    on(event, callback) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, []);
        }
        this.listeners.get(event).push(callback);
    }
    
    off(event, callback) {
        if (this.listeners.has(event)) {
            const callbacks = this.listeners.get(event);
            const index = callbacks.indexOf(callback);
            if (index > -1) {
                callbacks.splice(index, 1);
            }
        }
    }
    
    emit(event, data) {
        if (this.listeners.has(event)) {
            this.listeners.get(event).forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Error in WebSocket event handler for ${event}:`, error);
                }
            });
        }
    }
    
    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        }
    }
    
    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
    }
}

// Create global real-time API instance
const realTimeAPI = new RealTimeAPI();

// Export API classes
window.API = {
    ModelAPI,
    ResultsAPI,
    AnalysisAPI,
    StatisticsAPI,
    MockDataService,
    CachedAPI,
    RealTimeAPI: realTimeAPI,
    apiRequest,
    apiCache
};