// ===== Utility Functions =====

/**
 * Language management
 */
let currentLanguage = 'zh';

function setLanguage(lang) {
    currentLanguage = lang;
    localStorage.setItem('language', lang);
    updateLanguageDisplay();
}

function updateLanguageDisplay() {
    const elements = document.querySelectorAll('[data-en][data-zh]');
    elements.forEach(element => {
        const text = currentLanguage === 'en' ? element.getAttribute('data-en') : element.getAttribute('data-zh');
        if (text) {
            element.textContent = text;
        }
    });
}

function initializeLanguage() {
    const savedLanguage = localStorage.getItem('language') || 'zh';
    setLanguage(savedLanguage);
}

/**
 * Number formatting utilities
 */
function formatNumber(num, decimals = 2) {
    if (num === null || num === undefined) return '--';
    
    if (typeof num === 'string') {
        num = parseFloat(num);
    }
    
    if (isNaN(num)) return '--';
    
    return num.toFixed(decimals);
}

function formatPercentage(num, decimals = 1) {
    if (num === null || num === undefined) return '--';
    
    if (typeof num === 'string') {
        num = parseFloat(num);
    }
    
    if (isNaN(num)) return '--';
    
    return (num * 100).toFixed(decimals) + '%';
}

function formatLargeNumber(num) {
    if (num === null || num === undefined) return '--';
    
    if (typeof num === 'string') {
        num = parseFloat(num);
    }
    
    if (isNaN(num)) return '--';
    
    if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

/**
 * Data parsing utilities
 */
function parseCSVData(csvText) {
    if (!csvText || csvText.trim() === '') {
        return [];
    }
    
    const lines = csvText.trim().split('\n');
    return lines.map(line => {
        const value = parseFloat(line.trim());
        return isNaN(value) ? 0 : value;
    });
}

function calculateStatistics(data) {
    if (!data || data.length === 0) {
        return {
            mean: 0,
            min: 0,
            max: 0,
            std: 0,
            latest: 0
        };
    }
    
    const numbers = data.filter(x => !isNaN(x));
    if (numbers.length === 0) {
        return {
            mean: 0,
            min: 0,
            max: 0,
            std: 0,
            latest: 0
        };
    }
    
    const mean = numbers.reduce((a, b) => a + b, 0) / numbers.length;
    const min = Math.min(...numbers);
    const max = Math.max(...numbers);
    const variance = numbers.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / numbers.length;
    const std = Math.sqrt(variance);
    const latest = numbers[numbers.length - 1];
    
    return { mean, min, max, std, latest };
}

/**
 * Color utilities for charts
 */
const CHART_COLORS = {
    primary: '#00ff88',
    secondary: '#0066ff',
    danger: '#ff4444',
    warning: '#ffaa00',
    success: '#00dd66',
    info: '#8844ff',
    models: {
        vulseeker: '#00ff88',
        sense: '#0066ff',
        fit: '#ff4444',
        safe: '#ffaa00'
    },
    architectures: {
        arm: '#00ff88',
        mips: '#0066ff',
        x86: '#ff4444',
        'arm-mips': '#ffaa00',
        'arm-x86': '#8844ff',
        'mips-x86': '#ff8844',
        'arm-mips-x86': '#88ff44'
    },
    optimizations: {
        O0: '#ff4444',
        O1: '#ffaa00',
        O2: '#00ff88',
        O3: '#0066ff',
        'O0-O1': '#ff8844',
        'O1-O2': '#88ff44',
        'O2-O3': '#44ff88',
        'O0-O1-O2-O3': '#8844ff'
    }
};

function getColorForModel(model) {
    return CHART_COLORS.models[model.toLowerCase()] || CHART_COLORS.primary;
}

function getColorForArchitecture(arch) {
    return CHART_COLORS.architectures[arch.toLowerCase()] || CHART_COLORS.secondary;
}

function getColorForOptimization(opt) {
    return CHART_COLORS.optimizations[opt] || CHART_COLORS.warning;
}

function generateColorPalette(count) {
    const colors = [
        CHART_COLORS.primary,
        CHART_COLORS.secondary,
        CHART_COLORS.danger,
        CHART_COLORS.warning,
        CHART_COLORS.success,
        CHART_COLORS.info
    ];
    
    const result = [];
    for (let i = 0; i < count; i++) {
        result.push(colors[i % colors.length]);
    }
    return result;
}

/**
 * DOM utilities
 */
function showLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `
            <div class="chart-loading">
                <div class="chart-loading-spinner"></div>
                <div class="chart-loading-text">${currentLanguage === 'en' ? 'Loading...' : '加载中...'}</div>
            </div>
        `;
    }
}

function hideLoading(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        const loading = element.querySelector('.chart-loading');
        if (loading) {
            loading.remove();
        }
    }
}

function showError(elementId, message) {
    const element = document.getElementById(elementId);
    if (element) {
        const errorMessage = currentLanguage === 'en' ? 'Error loading data' : '数据加载失败';
        element.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-exclamation-triangle me-2"></i>
                ${message || errorMessage}
            </div>
        `;
    }
}

function createCard(title, content, className = '') {
    return `
        <div class="card ${className}">
            <div class="card-header">
                <h5 class="mb-0">${title}</h5>
            </div>
            <div class="card-body">
                ${content}
            </div>
        </div>
    `;
}

/**
 * File utilities
 */
function downloadData(data, filename, type = 'application/json') {
    let content;
    
    if (type === 'application/json') {
        content = JSON.stringify(data, null, 2);
    } else if (type === 'text/csv') {
        if (Array.isArray(data) && data.length > 0) {
            const headers = Object.keys(data[0]).join(',');
            const rows = data.map(row => Object.values(row).join(',')).join('\n');
            content = headers + '\n' + rows;
        } else {
            content = data.toString();
        }
    } else {
        content = data.toString();
    }
    
    const blob = new Blob([content], { type: type });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
}

/**
 * Animation utilities
 */
function animateNumber(element, start, end, duration = 1000) {
    const startTime = performance.now();
    const startNumber = parseFloat(start) || 0;
    const endNumber = parseFloat(end) || 0;
    
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        const current = startNumber + (endNumber - startNumber) * easeOutQuart(progress);
        
        if (element) {
            if (endNumber % 1 === 0) {
                element.textContent = Math.floor(current);
            } else {
                element.textContent = current.toFixed(1);
            }
        }
        
        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }
    
    requestAnimationFrame(update);
}

function easeOutQuart(t) {
    return 1 - (--t) * t * t * t;
}

function addFadeInAnimation(element, delay = 0) {
    element.style.opacity = '0';
    element.style.transform = 'translateY(20px)';
    element.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
    
    setTimeout(() => {
        element.style.opacity = '1';
        element.style.transform = 'translateY(0)';
    }, delay);
}

/**
 * Validation utilities
 */
function validateFile(file, allowedTypes = ['.bin', '.exe', '.so', '.dll'], maxSize = 100 * 1024 * 1024) {
    const errors = [];
    
    if (!file) {
        errors.push(currentLanguage === 'en' ? 'No file selected' : '未选择文件');
        return errors;
    }
    
    // Check file size
    if (file.size > maxSize) {
        const maxSizeMB = maxSize / (1024 * 1024);
        errors.push(currentLanguage === 'en' ? 
            `File size must be less than ${maxSizeMB}MB` : 
            `文件大小必须小于 ${maxSizeMB}MB`);
    }
    
    // Check file type
    const fileName = file.name.toLowerCase();
    const hasValidExtension = allowedTypes.some(type => fileName.endsWith(type.toLowerCase()));
    
    if (!hasValidExtension) {
        errors.push(currentLanguage === 'en' ? 
            `Allowed file types: ${allowedTypes.join(', ')}` : 
            `允许的文件类型: ${allowedTypes.join(', ')}`);
    }
    
    return errors;
}

/**
 * URL utilities
 */
function getUrlParams() {
    const params = new URLSearchParams(window.location.search);
    const result = {};
    for (const [key, value] of params) {
        result[key] = value;
    }
    return result;
}

function updateUrlParams(params) {
    const url = new URL(window.location);
    Object.keys(params).forEach(key => {
        if (params[key] !== null && params[key] !== undefined) {
            url.searchParams.set(key, params[key]);
        } else {
            url.searchParams.delete(key);
        }
    });
    window.history.replaceState({}, '', url);
}

/**
 * Date utilities
 */
function formatDate(date, format = 'YYYY-MM-DD HH:mm:ss') {
    if (!date) return '--';
    
    const d = new Date(date);
    if (isNaN(d.getTime())) return '--';
    
    const year = d.getFullYear();
    const month = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    const hours = String(d.getHours()).padStart(2, '0');
    const minutes = String(d.getMinutes()).padStart(2, '0');
    const seconds = String(d.getSeconds()).padStart(2, '0');
    
    return format
        .replace('YYYY', year)
        .replace('MM', month)
        .replace('DD', day)
        .replace('HH', hours)
        .replace('mm', minutes)
        .replace('ss', seconds);
}

function getRelativeTime(date) {
    if (!date) return '--';
    
    const d = new Date(date);
    if (isNaN(d.getTime())) return '--';
    
    const now = new Date();
    const diff = now - d;
    const seconds = Math.floor(diff / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);
    
    if (currentLanguage === 'en') {
        if (seconds < 60) return 'just now';
        if (minutes < 60) return `${minutes} minutes ago`;
        if (hours < 24) return `${hours} hours ago`;
        if (days < 7) return `${days} days ago`;
        return formatDate(d, 'YYYY-MM-DD');
    } else {
        if (seconds < 60) return '刚刚';
        if (minutes < 60) return `${minutes} 分钟前`;
        if (hours < 24) return `${hours} 小时前`;
        if (days < 7) return `${days} 天前`;
        return formatDate(d, 'YYYY-MM-DD');
    }
}

/**
 * Debounce utility
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Initialize utilities on page load
 */
document.addEventListener('DOMContentLoaded', function() {
    initializeLanguage();
});

// Export functions for use in other modules
window.Utils = {
    setLanguage,
    updateLanguageDisplay,
    formatNumber,
    formatPercentage,
    formatLargeNumber,
    parseCSVData,
    calculateStatistics,
    getColorForModel,
    getColorForArchitecture,
    getColorForOptimization,
    generateColorPalette,
    showLoading,
    hideLoading,
    showError,
    createCard,
    downloadData,
    animateNumber,
    addFadeInAnimation,
    validateFile,
    getUrlParams,
    updateUrlParams,
    formatDate,
    getRelativeTime,
    debounce,
    CHART_COLORS
};