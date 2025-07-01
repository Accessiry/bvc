// ===== Chart Management =====

/**
 * Chart configuration and themes
 */
const CHART_CONFIG = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
        intersect: false,
        mode: 'index'
    },
    plugins: {
        legend: {
            labels: {
                color: '#b8c7d1',
                font: {
                    family: "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif"
                }
            }
        },
        tooltip: {
            backgroundColor: 'rgba(26, 31, 53, 0.95)',
            titleColor: '#00ff88',
            bodyColor: '#b8c7d1',
            borderColor: '#00ff88',
            borderWidth: 1,
            cornerRadius: 8,
            displayColors: true,
            titleFont: {
                weight: 'bold'
            }
        }
    },
    scales: {
        x: {
            grid: {
                color: 'rgba(255, 255, 255, 0.1)'
            },
            ticks: {
                color: '#b8c7d1'
            },
            title: {
                color: '#b8c7d1',
                font: {
                    weight: 'bold'
                }
            }
        },
        y: {
            grid: {
                color: 'rgba(255, 255, 255, 0.1)'
            },
            ticks: {
                color: '#b8c7d1'
            },
            title: {
                color: '#b8c7d1',
                font: {
                    weight: 'bold'
                }
            }
        }
    }
};

/**
 * Chart registry to manage chart instances
 */
class ChartRegistry {
    constructor() {
        this.charts = new Map();
    }
    
    register(id, chart) {
        this.charts.set(id, chart);
    }
    
    get(id) {
        return this.charts.get(id);
    }
    
    destroy(id) {
        const chart = this.charts.get(id);
        if (chart) {
            chart.destroy();
            this.charts.delete(id);
        }
    }
    
    destroyAll() {
        this.charts.forEach(chart => chart.destroy());
        this.charts.clear();
    }
    
    update(id, data, options = {}) {
        const chart = this.charts.get(id);
        if (chart) {
            if (data) {
                chart.data = data;
            }
            if (options) {
                chart.options = { ...chart.options, ...options };
            }
            chart.update();
        }
    }
}

// Global chart registry
const chartRegistry = new ChartRegistry();

/**
 * Performance comparison chart
 */
function createPerformanceChart(canvasId, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    // Destroy existing chart
    chartRegistry.destroy(canvasId);
    
    const chart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: data.labels || ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'AUC'],
            datasets: data.datasets.map((dataset, index) => ({
                label: dataset.label,
                data: dataset.data,
                borderColor: Utils.getColorForModel(dataset.label),
                backgroundColor: Utils.getColorForModel(dataset.label) + '20',
                borderWidth: 2,
                pointBackgroundColor: Utils.getColorForModel(dataset.label),
                pointBorderColor: '#ffffff',
                pointBorderWidth: 2,
                pointRadius: 5,
                pointHoverRadius: 7
            }))
        },
        options: {
            ...CHART_CONFIG,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    pointLabels: {
                        color: '#b8c7d1',
                        font: {
                            weight: 'bold'
                        }
                    },
                    ticks: {
                        color: '#b8c7d1',
                        stepSize: 0.2,
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    }
                }
            },
            plugins: {
                ...CHART_CONFIG.plugins,
                title: {
                    display: true,
                    text: Utils.currentLanguage === 'en' ? 'Model Performance Comparison' : '模型性能对比',
                    color: '#00ff88',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                }
            }
        }
    });
    
    chartRegistry.register(canvasId, chart);
    return chart;
}

/**
 * ROC curve chart
 */
function createROCChart(canvasId, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    chartRegistry.destroy(canvasId);
    
    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                ...data.models.map(model => ({
                    label: model.name,
                    data: model.points,
                    borderColor: Utils.getColorForModel(model.name),
                    backgroundColor: 'transparent',
                    borderWidth: 3,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    tension: 0.1
                })),
                {
                    label: Utils.currentLanguage === 'en' ? 'Random Classifier' : '随机分类器',
                    data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
                    borderColor: '#666666',
                    backgroundColor: 'transparent',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    pointRadius: 0
                }
            ]
        },
        options: {
            ...CHART_CONFIG,
            scales: {
                x: {
                    ...CHART_CONFIG.scales.x,
                    type: 'linear',
                    position: 'bottom',
                    min: 0,
                    max: 1,
                    title: {
                        display: true,
                        text: Utils.currentLanguage === 'en' ? 'False Positive Rate' : '假阳性率',
                        color: '#b8c7d1',
                        font: { weight: 'bold' }
                    }
                },
                y: {
                    ...CHART_CONFIG.scales.y,
                    min: 0,
                    max: 1,
                    title: {
                        display: true,
                        text: Utils.currentLanguage === 'en' ? 'True Positive Rate' : '真阳性率',
                        color: '#b8c7d1',
                        font: { weight: 'bold' }
                    }
                }
            },
            plugins: {
                ...CHART_CONFIG.plugins,
                title: {
                    display: true,
                    text: Utils.currentLanguage === 'en' ? 'ROC Curve Comparison' : 'ROC曲线对比',
                    color: '#00ff88',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                }
            }
        }
    });
    
    chartRegistry.register(canvasId, chart);
    return chart;
}

/**
 * Architecture distribution pie chart
 */
function createArchitectureChart(canvasId, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    chartRegistry.destroy(canvasId);
    
    const chart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: data.labels,
            datasets: [{
                data: data.values,
                backgroundColor: data.labels.map(label => Utils.getColorForArchitecture(label)),
                borderColor: '#1a1f35',
                borderWidth: 2,
                hoverBorderWidth: 3
            }]
        },
        options: {
            ...CHART_CONFIG,
            cutout: '60%',
            plugins: {
                ...CHART_CONFIG.plugins,
                title: {
                    display: true,
                    text: Utils.currentLanguage === 'en' ? 'Architecture Distribution' : '架构分布',
                    color: '#00ff88',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                }
            }
        }
    });
    
    chartRegistry.register(canvasId, chart);
    return chart;
}

/**
 * Time series chart for performance trends
 */
function createTimeSeriesChart(canvasId, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    chartRegistry.destroy(canvasId);
    
    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.timestamps,
            datasets: data.metrics.map(metric => ({
                label: metric.name,
                data: metric.values,
                borderColor: Utils.getColorForModel(metric.name.toLowerCase()),
                backgroundColor: Utils.getColorForModel(metric.name.toLowerCase()) + '20',
                borderWidth: 2,
                fill: false,
                pointRadius: 3,
                pointHoverRadius: 6,
                tension: 0.2
            }))
        },
        options: {
            ...CHART_CONFIG,
            scales: {
                ...CHART_CONFIG.scales,
                x: {
                    ...CHART_CONFIG.scales.x,
                    type: 'time',
                    time: {
                        displayFormats: {
                            hour: 'HH:mm',
                            day: 'MM-DD',
                            week: 'MM-DD',
                            month: 'MM-DD'
                        }
                    },
                    title: {
                        display: true,
                        text: Utils.currentLanguage === 'en' ? 'Time' : '时间',
                        color: '#b8c7d1',
                        font: { weight: 'bold' }
                    }
                },
                y: {
                    ...CHART_CONFIG.scales.y,
                    beginAtZero: true,
                    max: 1,
                    title: {
                        display: true,
                        text: Utils.currentLanguage === 'en' ? 'Performance Score' : '性能得分',
                        color: '#b8c7d1',
                        font: { weight: 'bold' }
                    },
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    }
                }
            }
        }
    });
    
    chartRegistry.register(canvasId, chart);
    return chart;
}

/**
 * Vulnerability distribution bar chart
 */
function createVulnerabilityChart(canvasId, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    chartRegistry.destroy(canvasId);
    
    const chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.types,
            datasets: [{
                label: Utils.currentLanguage === 'en' ? 'Detected Vulnerabilities' : '检测到的漏洞',
                data: data.counts,
                backgroundColor: data.types.map((_, index) => 
                    Utils.generateColorPalette(data.types.length)[index] + '80'
                ),
                borderColor: data.types.map((_, index) => 
                    Utils.generateColorPalette(data.types.length)[index]
                ),
                borderWidth: 2,
                borderRadius: 4,
                borderSkipped: false
            }]
        },
        options: {
            ...CHART_CONFIG,
            indexAxis: 'y',
            scales: {
                x: {
                    ...CHART_CONFIG.scales.x,
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: Utils.currentLanguage === 'en' ? 'Count' : '数量',
                        color: '#b8c7d1',
                        font: { weight: 'bold' }
                    }
                },
                y: {
                    ...CHART_CONFIG.scales.y,
                    title: {
                        display: true,
                        text: Utils.currentLanguage === 'en' ? 'Vulnerability Type' : '漏洞类型',
                        color: '#b8c7d1',
                        font: { weight: 'bold' }
                    }
                }
            }
        }
    });
    
    chartRegistry.register(canvasId, chart);
    return chart;
}

/**
 * Confidence heatmap using Chart.js matrix
 */
function createConfidenceHeatmap(canvasId, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    chartRegistry.destroy(canvasId);
    
    // Transform data for heatmap
    const heatmapData = [];
    data.matrix.forEach((row, y) => {
        row.forEach((value, x) => {
            heatmapData.push({
                x: data.xLabels[x],
                y: data.yLabels[y],
                v: value
            });
        });
    });
    
    const chart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: Utils.currentLanguage === 'en' ? 'Confidence' : '置信度',
                data: heatmapData,
                backgroundColor: function(context) {
                    const value = context.parsed.v || 0;
                    const alpha = Math.max(0.1, value);
                    return `rgba(0, 255, 136, ${alpha})`;
                },
                borderColor: 'rgba(0, 255, 136, 0.8)',
                borderWidth: 1,
                pointRadius: 15
            }]
        },
        options: {
            ...CHART_CONFIG,
            scales: {
                x: {
                    ...CHART_CONFIG.scales.x,
                    type: 'category',
                    position: 'bottom',
                    title: {
                        display: true,
                        text: data.xTitle || (Utils.currentLanguage === 'en' ? 'Programs' : '程序'),
                        color: '#b8c7d1',
                        font: { weight: 'bold' }
                    }
                },
                y: {
                    ...CHART_CONFIG.scales.y,
                    type: 'category',
                    title: {
                        display: true,
                        text: data.yTitle || (Utils.currentLanguage === 'en' ? 'Models' : '模型'),
                        color: '#b8c7d1',
                        font: { weight: 'bold' }
                    }
                }
            },
            plugins: {
                ...CHART_CONFIG.plugins,
                tooltip: {
                    ...CHART_CONFIG.plugins.tooltip,
                    callbacks: {
                        title: function(context) {
                            const point = context[0];
                            return `${point.parsed.y} - ${point.parsed.x}`;
                        },
                        label: function(context) {
                            const confidence = context.parsed.v;
                            return `${Utils.currentLanguage === 'en' ? 'Confidence' : '置信度'}: ${Utils.formatPercentage(confidence)}`;
                        }
                    }
                }
            }
        }
    });
    
    chartRegistry.register(canvasId, chart);
    return chart;
}

/**
 * Model comparison metrics chart
 */
function createModelComparisonChart(canvasId, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    chartRegistry.destroy(canvasId);
    
    const chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: data.models,
            datasets: data.metrics.map(metric => ({
                label: metric.name,
                data: metric.values,
                backgroundColor: Utils.getColorForModel(metric.name.toLowerCase()) + '80',
                borderColor: Utils.getColorForModel(metric.name.toLowerCase()),
                borderWidth: 2,
                borderRadius: 4
            }))
        },
        options: {
            ...CHART_CONFIG,
            scales: {
                ...CHART_CONFIG.scales,
                y: {
                    ...CHART_CONFIG.scales.y,
                    beginAtZero: true,
                    max: 1,
                    title: {
                        display: true,
                        text: Utils.currentLanguage === 'en' ? 'Score' : '得分',
                        color: '#b8c7d1',
                        font: { weight: 'bold' }
                    },
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    }
                }
            },
            plugins: {
                ...CHART_CONFIG.plugins,
                title: {
                    display: true,
                    text: Utils.currentLanguage === 'en' ? 'Model Performance Metrics' : '模型性能指标',
                    color: '#00ff88',
                    font: {
                        size: 16,
                        weight: 'bold'
                    }
                }
            }
        }
    });
    
    chartRegistry.register(canvasId, chart);
    return chart;
}

/**
 * Real-time metrics gauge chart
 */
function createGaugeChart(canvasId, value, maxValue = 1, label = '') {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    chartRegistry.destroy(canvasId);
    
    const percentage = (value / maxValue) * 100;
    
    const chart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            datasets: [{
                data: [percentage, 100 - percentage],
                backgroundColor: ['#00ff88', 'rgba(255, 255, 255, 0.1)'],
                borderWidth: 0,
                cutout: '80%'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    enabled: false
                }
            }
        },
        plugins: [{
            beforeDraw: function(chart) {
                const ctx = chart.ctx;
                const centerX = chart.chartArea.left + (chart.chartArea.right - chart.chartArea.left) / 2;
                const centerY = chart.chartArea.top + (chart.chartArea.bottom - chart.chartArea.top) / 2;
                
                ctx.save();
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillStyle = '#00ff88';
                ctx.font = 'bold 24px "Segoe UI"';
                ctx.fillText(Utils.formatPercentage(value), centerX, centerY - 10);
                
                ctx.fillStyle = '#b8c7d1';
                ctx.font = '12px "Segoe UI"';
                ctx.fillText(label, centerX, centerY + 15);
                ctx.restore();
            }
        }]
    });
    
    chartRegistry.register(canvasId, chart);
    return chart;
}

/**
 * Chart export functionality
 */
function exportChart(chartId, filename, format = 'png') {
    const chart = chartRegistry.get(chartId);
    if (!chart) return;
    
    const url = chart.toBase64Image();
    const link = document.createElement('a');
    link.download = `${filename}.${format}`;
    link.href = url;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

/**
 * Chart update utilities
 */
function updateChartData(chartId, newData) {
    const chart = chartRegistry.get(chartId);
    if (chart) {
        chart.data = newData;
        chart.update('active');
    }
}

function animateChart(chartId, duration = 1000) {
    const chart = chartRegistry.get(chartId);
    if (chart) {
        chart.update('active', {
            duration: duration,
            easing: 'easeOutQuart'
        });
    }
}

/**
 * Responsive chart handling
 */
function handleResize() {
    chartRegistry.charts.forEach(chart => {
        chart.resize();
    });
}

// Add resize listener
window.addEventListener('resize', Utils.debounce(handleResize, 250));

// Export chart functions
window.Charts = {
    createPerformanceChart,
    createROCChart,
    createArchitectureChart,
    createTimeSeriesChart,
    createVulnerabilityChart,
    createConfidenceHeatmap,
    createModelComparisonChart,
    createGaugeChart,
    exportChart,
    updateChartData,
    animateChart,
    chartRegistry,
    CHART_CONFIG
};