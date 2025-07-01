// ===== Main Application Logic =====

/**
 * Application state management
 */
class AppState {
    constructor() {
        this.currentPage = window.location.pathname;
        this.data = {
            models: null,
            statistics: null,
            results: null
        };
        this.loading = new Set();
        this.errors = new Map();
        this.listeners = new Map();
    }
    
    setState(key, value) {
        this.data[key] = value;
        this.emit('stateChange', { key, value });
    }
    
    getState(key) {
        return this.data[key];
    }
    
    setLoading(operation, isLoading) {
        if (isLoading) {
            this.loading.add(operation);
        } else {
            this.loading.delete(operation);
        }
        this.emit('loadingChange', { operation, isLoading, loading: this.loading });
    }
    
    isLoading(operation) {
        return this.loading.has(operation);
    }
    
    setError(operation, error) {
        if (error) {
            this.errors.set(operation, error);
        } else {
            this.errors.delete(operation);
        }
        this.emit('errorChange', { operation, error, errors: this.errors });
    }
    
    getError(operation) {
        return this.errors.get(operation);
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
                    console.error(`Error in event handler for ${event}:`, error);
                }
            });
        }
    }
}

// Global application state
const appState = new AppState();

/**
 * Main application initialization
 */
async function initializeApp() {
    try {
        // Initialize language
        Utils.updateLanguageDisplay();
        
        // Initialize real-time connection if available
        try {
            API.RealTimeAPI.connect();
        } catch (error) {
            console.log('Real-time connection not available:', error);
        }
        
        // Load initial data based on page
        const page = window.location.pathname.split('/').pop() || 'index.html';
        
        switch (page) {
            case 'index.html':
            case '':
                await initializeHomePage();
                break;
            case 'models.html':
                await initializeModelsPage();
                break;
            case 'results.html':
                await initializeResultsPage();
                break;
            case 'analysis.html':
                await initializeAnalysisPage();
                break;
            case 'dashboard.html':
                await initializeDashboardPage();
                break;
        }
        
        // Set up global event listeners
        setupGlobalEventListeners();
        
        console.log('Application initialized successfully');
    } catch (error) {
        console.error('Failed to initialize application:', error);
        showGlobalError('Application initialization failed');
    }
}

/**
 * Home page initialization
 */
async function initializeHomePage() {
    try {
        // Load system statistics
        appState.setLoading('stats', true);
        
        // In development, use mock data
        const stats = await API.MockDataService.getMockData('stats');
        appState.setState('statistics', stats);
        
        // Update stats display
        updateSystemStats(stats);
        
        // Load and display charts
        await loadHomePageCharts();
        
        appState.setLoading('stats', false);
    } catch (error) {
        console.error('Failed to initialize home page:', error);
        appState.setError('stats', error);
        appState.setLoading('stats', false);
    }
}

/**
 * Update system statistics display
 */
function updateSystemStats(stats) {
    // Update hero stats with animation
    const elements = [
        { id: 'model-count', value: 4 },
        { id: 'arch-count', value: 7 },
        { id: 'detection-count', value: stats.totalDetections },
        { id: 'accuracy-rate', value: Utils.formatPercentage(stats.avgAccuracy) }
    ];
    
    elements.forEach(({ id, value }) => {
        const element = document.getElementById(id);
        if (element) {
            if (typeof value === 'number' && id !== 'accuracy-rate') {
                Utils.animateNumber(element, 0, value, 2000);
            } else {
                element.textContent = value;
            }
        }
    });
}

/**
 * Load home page charts
 */
async function loadHomePageCharts() {
    try {
        // Performance comparison chart
        const performanceData = {
            labels: ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'AUC'],
            datasets: [
                {
                    label: 'VulSeeker',
                    data: [0.92, 0.88, 0.90, 0.91, 0.94]
                },
                {
                    label: 'SENSE',
                    data: [0.89, 0.91, 0.90, 0.89, 0.92]
                },
                {
                    label: 'FIT',
                    data: [0.87, 0.85, 0.86, 0.88, 0.90]
                },
                {
                    label: 'SAFE',
                    data: [0.90, 0.89, 0.89, 0.90, 0.93]
                }
            ]
        };
        
        Charts.createPerformanceChart('performanceChart', performanceData);
        
        // Architecture distribution chart
        const archData = {
            labels: ['ARM', 'MIPS', 'x86', 'ARM-MIPS', 'ARM-x86', 'MIPS-x86', 'ARM-MIPS-x86'],
            values: [25, 20, 18, 15, 10, 8, 4]
        };
        
        Charts.createArchitectureChart('archChart', archData);
        
    } catch (error) {
        console.error('Failed to load home page charts:', error);
    }
}

/**
 * Models page initialization
 */
async function initializeModelsPage() {
    try {
        appState.setLoading('models', true);
        
        // Load model data (mock for development)
        const models = ['vulseeker', 'sense', 'fit', 'safe'];
        const modelData = await Promise.all(
            models.map(async model => ({
                name: model,
                ...(await API.MockDataService.getMockData('performance', model))
            }))
        );
        
        appState.setState('models', modelData);
        appState.setLoading('models', false);
        
        // Initialize model comparison interface
        if (typeof initializeModelComparison === 'function') {
            initializeModelComparison(modelData);
        }
        
    } catch (error) {
        console.error('Failed to initialize models page:', error);
        appState.setError('models', error);
        appState.setLoading('models', false);
    }
}

/**
 * Results page initialization
 */
async function initializeResultsPage() {
    try {
        appState.setLoading('results', true);
        
        // Load detection results (mock for development)
        const results = await API.MockDataService.getMockData('results', 50);
        appState.setState('results', results);
        
        appState.setLoading('results', false);
        
        // Initialize results table and filters
        if (typeof initializeResultsTable === 'function') {
            initializeResultsTable(results);
        }
        
    } catch (error) {
        console.error('Failed to initialize results page:', error);
        appState.setError('results', error);
        appState.setLoading('results', false);
    }
}

/**
 * Analysis page initialization
 */
async function initializeAnalysisPage() {
    try {
        // Initialize file upload interface
        if (typeof initializeFileUpload === 'function') {
            initializeFileUpload();
        }
        
        // Load analysis history
        if (typeof loadAnalysisHistory === 'function') {
            await loadAnalysisHistory();
        }
        
    } catch (error) {
        console.error('Failed to initialize analysis page:', error);
    }
}

/**
 * Dashboard page initialization
 */
async function initializeDashboardPage() {
    try {
        appState.setLoading('dashboard', true);
        
        // Load real-time metrics
        const metrics = await API.MockDataService.getMockData('stats');
        
        // Initialize dashboard widgets
        if (typeof initializeDashboard === 'function') {
            initializeDashboard(metrics);
        }
        
        appState.setLoading('dashboard', false);
        
    } catch (error) {
        console.error('Failed to initialize dashboard page:', error);
        appState.setError('dashboard', error);
        appState.setLoading('dashboard', false);
    }
}

/**
 * Global event listeners
 */
function setupGlobalEventListeners() {
    // Language switch
    document.addEventListener('click', function(e) {
        if (e.target.matches('[onclick*="setLanguage"]')) {
            e.preventDefault();
            const lang = e.target.textContent === '中文' ? 'zh' : 'en';
            Utils.setLanguage(lang);
        }
    });
    
    // Navigation handling
    document.addEventListener('click', function(e) {
        if (e.target.matches('a[href^="pages/"]')) {
            // Add loading indicator for page transitions
            const targetPage = e.target.getAttribute('href');
            console.log(`Navigating to ${targetPage}`);
        }
    });
    
    // Chart export buttons
    document.addEventListener('click', function(e) {
        if (e.target.matches('.export-btn[data-chart]')) {
            e.preventDefault();
            const chartId = e.target.getAttribute('data-chart');
            const filename = e.target.getAttribute('data-filename') || 'chart';
            Charts.exportChart(chartId, filename);
        }
    });
    
    // Real-time updates
    API.RealTimeAPI.on('metrics_update', function(data) {
        console.log('Received real-time metrics update:', data);
        updateRealTimeMetrics(data);
    });
    
    API.RealTimeAPI.on('detection_complete', function(data) {
        console.log('Detection completed:', data);
        showDetectionNotification(data);
    });
    
    // State change listeners
    appState.on('stateChange', function({ key, value }) {
        console.log(`State changed: ${key}`, value);
    });
    
    appState.on('loadingChange', function({ operation, isLoading }) {
        console.log(`Loading state changed: ${operation} = ${isLoading}`);
        updateLoadingIndicators(operation, isLoading);
    });
    
    appState.on('errorChange', function({ operation, error }) {
        if (error) {
            console.error(`Error in ${operation}:`, error);
            showErrorNotification(operation, error);
        }
    });
}

/**
 * Update real-time metrics
 */
function updateRealTimeMetrics(data) {
    // Update gauge charts on dashboard
    if (window.location.pathname.includes('dashboard')) {
        if (data.accuracy && document.getElementById('accuracyGauge')) {
            Charts.updateChartData('accuracyGauge', data.accuracy);
        }
        
        if (data.throughput && document.getElementById('throughputGauge')) {
            Charts.updateChartData('throughputGauge', data.throughput);
        }
    }
    
    // Update system stats on home page
    if (window.location.pathname.includes('index') || window.location.pathname === '/') {
        updateSystemStats(data);
    }
}

/**
 * Show detection notification
 */
function showDetectionNotification(data) {
    const notification = document.createElement('div');
    notification.className = 'alert alert-info alert-dismissible fade show position-fixed';
    notification.style.cssText = 'top: 100px; right: 20px; z-index: 9999; min-width: 300px;';
    
    const isVuln = data.isVulnerable;
    const alertClass = isVuln ? 'alert-warning' : 'alert-success';
    const icon = isVuln ? 'fas fa-exclamation-triangle' : 'fas fa-check-circle';
    const message = isVuln ? 
        (Utils.currentLanguage === 'en' ? 'Vulnerability detected!' : '检测到漏洞！') :
        (Utils.currentLanguage === 'en' ? 'No vulnerabilities found' : '未发现漏洞');
    
    notification.innerHTML = `
        <div class="d-flex align-items-center">
            <i class="${icon} me-2"></i>
            <div>
                <strong>${message}</strong><br>
                <small>${Utils.currentLanguage === 'en' ? 'Confidence' : '置信度'}: ${Utils.formatPercentage(data.confidence)}</small>
            </div>
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    notification.className = notification.className.replace('alert-info', alertClass);
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

/**
 * Show error notification
 */
function showErrorNotification(operation, error) {
    const notification = document.createElement('div');
    notification.className = 'alert alert-danger alert-dismissible fade show position-fixed';
    notification.style.cssText = 'top: 100px; right: 20px; z-index: 9999; min-width: 300px;';
    
    const message = Utils.currentLanguage === 'en' ? 
        `Error in ${operation}: ${error.message}` :
        `${operation} 操作失败: ${error.message}`;
    
    notification.innerHTML = `
        <div class="d-flex align-items-center">
            <i class="fas fa-exclamation-circle me-2"></i>
            <div>
                <strong>${Utils.currentLanguage === 'en' ? 'Error' : '错误'}</strong><br>
                <small>${message}</small>
            </div>
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 8 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 8000);
}

/**
 * Show global error
 */
function showGlobalError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'alert alert-danger text-center';
    errorDiv.innerHTML = `
        <i class="fas fa-exclamation-triangle me-2"></i>
        ${message}
    `;
    
    const container = document.querySelector('main') || document.body;
    container.insertBefore(errorDiv, container.firstChild);
}

/**
 * Update loading indicators
 */
function updateLoadingIndicators(operation, isLoading) {
    // Update specific loading indicators
    const loadingElements = document.querySelectorAll(`[data-loading="${operation}"]`);
    loadingElements.forEach(element => {
        if (isLoading) {
            element.innerHTML = '<div class="loading"></div>';
        } else {
            element.innerHTML = '';
        }
    });
    
    // Update global loading state
    const hasAnyLoading = appState.loading.size > 0;
    document.body.classList.toggle('app-loading', hasAnyLoading);
}

/**
 * Page-specific cleanup
 */
function cleanupPage() {
    // Destroy all charts
    Charts.chartRegistry.destroyAll();
    
    // Clear timers and intervals
    if (window.dashboardTimer) {
        clearInterval(window.dashboardTimer);
    }
    
    // Clear API cache if needed
    API.apiCache.clear();
}

/**
 * Handle page visibility changes
 */
function handleVisibilityChange() {
    if (document.hidden) {
        // Pause real-time updates when page is hidden
        console.log('Page hidden, pausing updates');
    } else {
        // Resume updates when page becomes visible
        console.log('Page visible, resuming updates');
        
        // Refresh data if page was hidden for more than 5 minutes
        const lastUpdate = localStorage.getItem('lastUpdate');
        if (lastUpdate && Date.now() - parseInt(lastUpdate) > 5 * 60 * 1000) {
            location.reload();
        }
    }
}

/**
 * Save app state periodically
 */
function saveAppState() {
    try {
        const stateToSave = {
            language: Utils.currentLanguage,
            lastUpdate: Date.now(),
            page: window.location.pathname
        };
        
        localStorage.setItem('appState', JSON.stringify(stateToSave));
        localStorage.setItem('lastUpdate', Date.now().toString());
    } catch (error) {
        console.error('Failed to save app state:', error);
    }
}

/**
 * Load saved app state
 */
function loadAppState() {
    try {
        const savedState = localStorage.getItem('appState');
        if (savedState) {
            const state = JSON.parse(savedState);
            if (state.language) {
                Utils.setLanguage(state.language);
            }
        }
    } catch (error) {
        console.error('Failed to load app state:', error);
    }
}

/**
 * Initialize app when DOM is ready
 */
document.addEventListener('DOMContentLoaded', function() {
    loadAppState();
    initializeApp();
    
    // Set up periodic state saving
    setInterval(saveAppState, 30000); // Save every 30 seconds
    
    // Handle page visibility changes
    document.addEventListener('visibilitychange', handleVisibilityChange);
    
    // Handle page unload
    window.addEventListener('beforeunload', function() {
        cleanupPage();
        saveAppState();
    });
});

// Export main functions
window.App = {
    appState,
    initializeApp,
    updateSystemStats,
    showDetectionNotification,
    showErrorNotification,
    cleanupPage
};