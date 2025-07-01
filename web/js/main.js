// Main JavaScript functionality for the academic vulnerability detection system

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all components
    initStatCounters();
    initSmoothScrolling();
    initTooltips();
    initLoadingStates();
    
    console.log('BVC Academic System initialized');
});

// Animated statistics counters
function initStatCounters() {
    const statNumbers = document.querySelectorAll('.stat-number');
    
    const observerOptions = {
        threshold: 0.7
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                animateCounter(entry.target);
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    statNumbers.forEach(stat => {
        observer.observe(stat);
    });
}

function animateCounter(element) {
    const target = parseInt(element.getAttribute('data-count'));
    const duration = 2000; // 2 seconds
    const increment = target / (duration / 16); // 60fps
    let current = 0;
    
    const timer = setInterval(() => {
        current += increment;
        if (current >= target) {
            current = target;
            clearInterval(timer);
        }
        element.textContent = Math.floor(current);
    }, 16);
}

// Smooth scrolling for navigation links
function initSmoothScrolling() {
    const navLinks = document.querySelectorAll('a[href^="#"]');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Initialize Bootstrap tooltips
function initTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Loading states for interactive elements
function initLoadingStates() {
    const buttons = document.querySelectorAll('.btn');
    
    buttons.forEach(button => {
        button.addEventListener('click', function() {
            if (this.classList.contains('btn-loading')) return;
            
            const originalText = this.innerHTML;
            this.classList.add('btn-loading');
            this.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 加载中...';
            this.disabled = true;
            
            // Simulate loading (remove this in real implementation)
            setTimeout(() => {
                this.classList.remove('btn-loading');
                this.innerHTML = originalText;
                this.disabled = false;
            }, 1500);
        });
    });
}

// Utility functions for common operations
const Utils = {
    // Format numbers with appropriate suffixes
    formatNumber: function(num) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        }
        if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toString();
    },
    
    // Format file sizes
    formatFileSize: function(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },
    
    // Format time duration
    formatDuration: function(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;
        
        if (hours > 0) {
            return `${hours}h ${minutes}m ${secs}s`;
        } else if (minutes > 0) {
            return `${minutes}m ${secs}s`;
        } else {
            return `${secs}s`;
        }
    },
    
    // Show notification
    showNotification: function(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        notification.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(notification);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    },
    
    // Show loading overlay
    showLoading: function(element) {
        element.classList.add('loading');
        element.style.pointerEvents = 'none';
    },
    
    // Hide loading overlay
    hideLoading: function(element) {
        element.classList.remove('loading');
        element.style.pointerEvents = 'auto';
    }
};

// API helper functions
const API = {
    baseURL: '/api',
    
    // Generic fetch wrapper
    request: async function(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const config = {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };
        
        try {
            const response = await fetch(url, config);
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.message || 'Request failed');
            }
            
            return data;
        } catch (error) {
            console.error('API Error:', error);
            Utils.showNotification(`错误: ${error.message}`, 'danger');
            throw error;
        }
    },
    
    // GET request
    get: function(endpoint) {
        return this.request(endpoint);
    },
    
    // POST request
    post: function(endpoint, data) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    },
    
    // PUT request
    put: function(endpoint, data) {
        return this.request(endpoint, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    },
    
    // DELETE request
    delete: function(endpoint) {
        return this.request(endpoint, {
            method: 'DELETE'
        });
    }
};

// Theme management
const Theme = {
    current: localStorage.getItem('theme') || 'light',
    
    init: function() {
        this.apply(this.current);
        this.addToggleButton();
    },
    
    apply: function(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        this.current = theme;
        localStorage.setItem('theme', theme);
    },
    
    toggle: function() {
        const newTheme = this.current === 'light' ? 'dark' : 'light';
        this.apply(newTheme);
    },
    
    addToggleButton: function() {
        const button = document.createElement('button');
        button.className = 'btn btn-outline-secondary position-fixed';
        button.style.cssText = 'bottom: 20px; right: 20px; z-index: 9999; border-radius: 50%; width: 50px; height: 50px;';
        button.innerHTML = '<i class="fas fa-moon"></i>';
        button.title = '切换主题';
        
        button.addEventListener('click', () => {
            this.toggle();
            button.innerHTML = this.current === 'dark' ? '<i class="fas fa-sun"></i>' : '<i class="fas fa-moon"></i>';
        });
        
        document.body.appendChild(button);
    }
};

// Navigation highlighting
function initNavHighlighting() {
    const currentPage = window.location.pathname.split('/').pop() || 'index.html';
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
    
    navLinks.forEach(link => {
        const href = link.getAttribute('href');
        if (href === currentPage) {
            link.classList.add('active');
        } else {
            link.classList.remove('active');
        }
    });
}

// Initialize theme and navigation
document.addEventListener('DOMContentLoaded', function() {
    Theme.init();
    initNavHighlighting();
});

// Export for use in other modules
window.BVC = {
    Utils,
    API,
    Theme
};