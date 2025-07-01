// Main JavaScript for BVC Research Platform

class BVCPlatform {
    constructor() {
        this.currentPage = this.getCurrentPage();
        this.init();
    }

    getCurrentPage() {
        const path = window.location.pathname;
        if (path.includes('results.html')) return 'results';
        if (path.includes('analysis.html')) return 'analysis';
        if (path.includes('docs.html')) return 'docs';
        return 'index';
    }

    init() {
        this.setupEventListeners();
        this.initializePage();
    }

    setupEventListeners() {
        // Filter controls on results page
        if (this.currentPage === 'results') {
            const programSelect = document.getElementById('program-select');
            const archSelect = document.getElementById('arch-select');
            const optSelect = document.getElementById('opt-select');

            [programSelect, archSelect, optSelect].forEach(select => {
                if (select) {
                    select.addEventListener('change', () => {
                        this.updateFilters();
                    });
                }
            });
        }

        // Analysis page filter controls
        if (this.currentPage === 'analysis') {
            this.setupAnalysisFilters();
        }

        // Smooth scrolling for internal links
        document.addEventListener('click', (e) => {
            if (e.target.matches('a[href^="#"]')) {
                e.preventDefault();
                const target = document.querySelector(e.target.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
        });

        // Responsive navigation
        this.setupResponsiveNav();
    }

    setupResponsiveNav() {
        const navToggle = document.querySelector('.nav-toggle');
        const navMenu = document.querySelector('.nav-menu');
        
        if (navToggle && navMenu) {
            navToggle.addEventListener('click', () => {
                navMenu.classList.toggle('active');
            });
        }
    }

    initializePage() {
        switch (this.currentPage) {
            case 'results':
                this.initializeResultsPage();
                break;
            case 'analysis':
                this.initializeAnalysisPage();
                break;
            case 'docs':
                this.initializeDocsPage();
                break;
            default:
                this.initializeIndexPage();
        }
    }

    initializeIndexPage() {
        // Add any index page specific initialization
        this.animateMetrics();
    }

    initializeResultsPage() {
        // Initialize charts after a short delay to ensure DOM is ready
        setTimeout(() => {
            if (typeof chartManager !== 'undefined') {
                chartManager.initializeCharts();
            }
        }, 100);
    }

    initializeAnalysisPage() {
        // Initialize analysis page
        if (typeof analysisManager !== 'undefined') {
            analysisManager.initialize();
        }
    }

    initializeDocsPage() {
        // Initialize docs page - table of contents, syntax highlighting etc.
        this.generateTableOfContents();
    }

    updateFilters() {
        const programSelect = document.getElementById('program-select');
        const archSelect = document.getElementById('arch-select');
        const optSelect = document.getElementById('opt-select');

        const filters = {
            program: programSelect?.value || 'all',
            architecture: archSelect?.value || 'all',
            optimization: optSelect?.value || 'all'
        };

        // Apply filters to data manager
        if (typeof dataManager !== 'undefined') {
            dataManager.applyFilters(filters);
        }

        // Update charts
        if (typeof chartManager !== 'undefined') {
            chartManager.updateCharts();
        }

        // Update analysis page if applicable
        if (this.currentPage === 'analysis' && typeof analysisManager !== 'undefined') {
            analysisManager.updateData();
        }
    }

    setupAnalysisFilters() {
        // Model filter buttons
        const modelButtons = document.querySelectorAll('.model-filter-btn');
        modelButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                // Toggle active state
                btn.classList.toggle('active');
                this.updateAnalysisFilters();
            });
        });

        // Metric filter buttons
        const metricButtons = document.querySelectorAll('.metric-filter-btn');
        metricButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                // Radio button behavior for metrics
                metricButtons.forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.updateAnalysisFilters();
            });
        });

        // Date range picker
        const dateRangeInputs = document.querySelectorAll('.date-range input');
        dateRangeInputs.forEach(input => {
            input.addEventListener('change', () => {
                this.updateAnalysisFilters();
            });
        });
    }

    updateAnalysisFilters() {
        // Get active models
        const activeModels = Array.from(document.querySelectorAll('.model-filter-btn.active'))
            .map(btn => btn.dataset.model);

        // Get active metric
        const activeMetric = document.querySelector('.metric-filter-btn.active')?.dataset.metric || 'auc';

        // Get date range
        const startDate = document.getElementById('start-date')?.value;
        const endDate = document.getElementById('end-date')?.value;

        // Apply filters
        if (typeof analysisManager !== 'undefined') {
            analysisManager.applyFilters({
                models: activeModels,
                metric: activeMetric,
                dateRange: { start: startDate, end: endDate }
            });
        }
    }

    animateMetrics() {
        // Animate metric cards on the index page
        const metricCards = document.querySelectorAll('.metric-card .metric-value');
        
        metricCards.forEach(card => {
            const targetValue = parseFloat(card.textContent);
            const duration = 2000; // 2 seconds
            const startTime = performance.now();
            
            const animate = (currentTime) => {
                const elapsed = currentTime - startTime;
                const progress = Math.min(elapsed / duration, 1);
                
                const currentValue = targetValue * this.easeOutCubic(progress);
                card.textContent = currentValue.toFixed(3);
                
                if (progress < 1) {
                    requestAnimationFrame(animate);
                }
            };
            
            card.textContent = '0.000';
            requestAnimationFrame(animate);
        });
    }

    easeOutCubic(t) {
        return 1 - Math.pow(1 - t, 3);
    }

    generateTableOfContents() {
        const toc = document.getElementById('table-of-contents');
        if (!toc) return;

        const headings = document.querySelectorAll('h2, h3, h4');
        const tocList = document.createElement('ul');
        tocList.className = 'toc-list';

        headings.forEach((heading, index) => {
            // Add ID to heading if it doesn't have one
            if (!heading.id) {
                heading.id = `heading-${index}`;
            }

            const li = document.createElement('li');
            li.className = `toc-item toc-${heading.tagName.toLowerCase()}`;
            
            const link = document.createElement('a');
            link.href = `#${heading.id}`;
            link.textContent = heading.textContent;
            link.className = 'toc-link';
            
            li.appendChild(link);
            tocList.appendChild(li);
        });

        toc.appendChild(tocList);
    }

    // Utility functions
    formatNumber(num, decimals = 3) {
        return parseFloat(num).toFixed(decimals);
    }

    formatPercentage(num, decimals = 1) {
        return (parseFloat(num) * 100).toFixed(decimals) + '%';
    }

    debounce(func, wait) {
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

    // Show loading state
    showLoading(element) {
        if (element) {
            element.innerHTML = `
                <div class="chart-loading">
                    <div class="loading-spinner"></div>
                    <span>加载中...</span>
                </div>
            `;
        }
    }

    // Show error state
    showError(element, message = '数据加载失败') {
        if (element) {
            element.innerHTML = `
                <div class="chart-error">
                    <p>⚠️ ${message}</p>
                    <button onclick="location.reload()" class="retry-btn">重试</button>
                </div>
            `;
        }
    }

    // Copy text to clipboard
    copyToClipboard(text) {
        navigator.clipboard.writeText(text).then(() => {
            this.showToast('已复制到剪贴板');
        }).catch(err => {
            console.error('Failed to copy: ', err);
            this.showToast('复制失败', 'error');
        });
    }

    // Show toast notification
    showToast(message, type = 'success') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        
        document.body.appendChild(toast);
        
        // Trigger animation
        setTimeout(() => toast.classList.add('show'), 100);
        
        // Remove toast
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => document.body.removeChild(toast), 300);
        }, 3000);
    }

    // Download data as file
    downloadFile(data, filename, type = 'text/plain') {
        const blob = new Blob([data], { type });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }
}

// Global error handler
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    // You might want to send this to a logging service
});

// Initialize the platform when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.bvcPlatform = new BVCPlatform();
});

// Handle page visibility changes
document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
        // Page became visible - refresh data if needed
        if (window.bvcPlatform && window.bvcPlatform.currentPage === 'results') {
            // Optionally refresh charts
        }
    }
});

// Handle window resize
window.addEventListener('resize', () => {
    // Debounce resize events
    if (!window.resizeTimeout) {
        window.resizeTimeout = setTimeout(() => {
            // Resize charts if they exist
            if (typeof chartManager !== 'undefined') {
                Object.values(chartManager.charts).forEach(chart => {
                    if (chart && chart.resize) {
                        chart.resize();
                    }
                });
            }
            window.resizeTimeout = null;
        }, 250);
    }
});

// Export for global access
window.BVCPlatform = BVCPlatform;