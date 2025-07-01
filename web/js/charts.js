// Chart.js configuration and management for training visualization
class ChartManager {
    constructor() {
        this.charts = {};
        this.chartData = {
            loss: { train: [], validation: [] },
            accuracy: { train: [], validation: [] }
        };
        
        this.initializeCharts();
    }

    initializeCharts() {
        this.createLossChart();
        this.createAccuracyChart();
    }

    createLossChart() {
        const ctx = document.getElementById('lossChart').getContext('2d');
        
        this.charts.loss = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: '训练损失',
                        data: [],
                        borderColor: '#e74c3c',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.1,
                        pointRadius: 3,
                        pointHoverRadius: 5
                    },
                    {
                        label: '验证损失',
                        data: [],
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.1,
                        pointRadius: 3,
                        pointHoverRadius: 5
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: '损失函数变化趋势',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: 'white',
                        bodyColor: 'white',
                        borderColor: '#ddd',
                        borderWidth: 1,
                        cornerRadius: 6,
                        callbacks: {
                            title: function(tooltipItems) {
                                return `Epoch ${tooltipItems[0].label}`;
                            },
                            label: function(context) {
                                return `${context.dataset.label}: ${context.parsed.y.toFixed(4)}`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Epoch'
                        },
                        grid: {
                            display: true,
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Loss'
                        },
                        grid: {
                            display: true,
                            color: 'rgba(0, 0, 0, 0.1)'
                        },
                        beginAtZero: false
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                },
                animation: {
                    duration: 750,
                    easing: 'easeInOutQuart'
                }
            }
        });
    }

    createAccuracyChart() {
        const ctx = document.getElementById('accuracyChart').getContext('2d');
        
        this.charts.accuracy = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: '训练准确率',
                        data: [],
                        borderColor: '#27ae60',
                        backgroundColor: 'rgba(39, 174, 96, 0.1)',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.1,
                        pointRadius: 3,
                        pointHoverRadius: 5
                    },
                    {
                        label: '验证准确率',
                        data: [],
                        borderColor: '#f39c12',
                        backgroundColor: 'rgba(243, 156, 18, 0.1)',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.1,
                        pointRadius: 3,
                        pointHoverRadius: 5
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: '准确率变化趋势',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: 'white',
                        bodyColor: 'white',
                        borderColor: '#ddd',
                        borderWidth: 1,
                        cornerRadius: 6,
                        callbacks: {
                            title: function(tooltipItems) {
                                return `Epoch ${tooltipItems[0].label}`;
                            },
                            label: function(context) {
                                return `${context.dataset.label}: ${(context.parsed.y * 100).toFixed(2)}%`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Epoch'
                        },
                        grid: {
                            display: true,
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Accuracy'
                        },
                        grid: {
                            display: true,
                            color: 'rgba(0, 0, 0, 0.1)'
                        },
                        min: 0,
                        max: 1,
                        ticks: {
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        }
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                },
                animation: {
                    duration: 750,
                    easing: 'easeInOutQuart'
                }
            }
        });
    }

    addDataPoint(chartType, data) {
        if (!this.charts[chartType]) return;
        
        const chart = this.charts[chartType];
        const currentEpoch = chart.data.labels.length + 1;
        
        // Add epoch label
        chart.data.labels.push(currentEpoch);
        
        // Add data points
        if (chartType === 'loss') {
            chart.data.datasets[0].data.push(data.train);
            chart.data.datasets[1].data.push(data.validation);
            
            // Store data for export/analysis
            this.chartData.loss.train.push({ epoch: currentEpoch, value: data.train });
            this.chartData.loss.validation.push({ epoch: currentEpoch, value: data.validation });
        } else if (chartType === 'accuracy') {
            chart.data.datasets[0].data.push(data.train);
            chart.data.datasets[1].data.push(data.validation);
            
            // Store data for export/analysis
            this.chartData.accuracy.train.push({ epoch: currentEpoch, value: data.train });
            this.chartData.accuracy.validation.push({ epoch: currentEpoch, value: data.validation });
        }
        
        // Limit data points to prevent performance issues
        const maxDataPoints = 1000;
        if (chart.data.labels.length > maxDataPoints) {
            chart.data.labels.shift();
            chart.data.datasets.forEach(dataset => {
                dataset.data.shift();
            });
        }
        
        // Update chart with animation
        chart.update('none'); // Use 'none' for real-time updates to improve performance
        
        // Trigger custom event for other components
        this.triggerChartUpdate(chartType, data);
    }

    triggerChartUpdate(chartType, data) {
        const event = new CustomEvent('chartUpdate', {
            detail: { chartType, data }
        });
        document.dispatchEvent(event);
    }

    clearCharts() {
        Object.keys(this.charts).forEach(chartType => {
            const chart = this.charts[chartType];
            chart.data.labels = [];
            chart.data.datasets.forEach(dataset => {
                dataset.data = [];
            });
            chart.update();
        });
        
        // Clear stored data
        this.chartData = {
            loss: { train: [], validation: [] },
            accuracy: { train: [], validation: [] }
        };
    }

    exportChartData(chartType) {
        const data = this.chartData[chartType];
        if (!data) return null;
        
        const csvData = [];
        csvData.push(['Epoch', 'Train', 'Validation']);
        
        const maxLength = Math.max(data.train.length, data.validation.length);
        for (let i = 0; i < maxLength; i++) {
            const trainValue = data.train[i] ? data.train[i].value : '';
            const valValue = data.validation[i] ? data.validation[i].value : '';
            csvData.push([i + 1, trainValue, valValue]);
        }
        
        return csvData.map(row => row.join(',')).join('\n');
    }

    downloadChartData(chartType) {
        const csvData = this.exportChartData(chartType);
        if (!csvData) return;
        
        const blob = new Blob([csvData], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${chartType}_data.csv`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    }

    saveChartImage(chartType) {
        const chart = this.charts[chartType];
        if (!chart) return;
        
        const url = chart.toBase64Image('image/png', 1.0);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${chartType}_chart.png`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }

    setChartTheme(theme) {
        const isDark = theme === 'dark';
        const textColor = isDark ? '#ffffff' : '#2c3e50';
        const gridColor = isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
        
        Object.values(this.charts).forEach(chart => {
            // Update scale colors
            chart.options.scales.x.ticks.color = textColor;
            chart.options.scales.x.title.color = textColor;
            chart.options.scales.x.grid.color = gridColor;
            
            chart.options.scales.y.ticks.color = textColor;
            chart.options.scales.y.title.color = textColor;
            chart.options.scales.y.grid.color = gridColor;
            
            // Update plugin colors
            chart.options.plugins.title.color = textColor;
            chart.options.plugins.legend.labels.color = textColor;
            
            chart.update();
        });
    }

    resizeCharts() {
        // Force chart resize when container size changes
        Object.values(this.charts).forEach(chart => {
            chart.resize();
        });
    }

    getChartStatistics(chartType) {
        const data = this.chartData[chartType];
        if (!data || !data.train.length) return null;
        
        const trainValues = data.train.map(d => d.value);
        const valValues = data.validation.map(d => d.value);
        
        const stats = {
            train: {
                min: Math.min(...trainValues),
                max: Math.max(...trainValues),
                avg: trainValues.reduce((a, b) => a + b, 0) / trainValues.length,
                latest: trainValues[trainValues.length - 1]
            },
            validation: {
                min: Math.min(...valValues),
                max: Math.max(...valValues),
                avg: valValues.reduce((a, b) => a + b, 0) / valValues.length,
                latest: valValues[valValues.length - 1]
            }
        };
        
        return stats;
    }

    setChartZoom(chartType, enabled) {
        const chart = this.charts[chartType];
        if (!chart) return;
        
        if (enabled) {
            chart.options.plugins.zoom = {
                pan: {
                    enabled: true,
                    mode: 'x'
                },
                zoom: {
                    wheel: {
                        enabled: true
                    },
                    pinch: {
                        enabled: true
                    },
                    mode: 'x'
                }
            };
        } else {
            delete chart.options.plugins.zoom;
        }
        
        chart.update();
    }

    resetChartZoom(chartType) {
        const chart = this.charts[chartType];
        if (chart && chart.resetZoom) {
            chart.resetZoom();
        }
    }
}

// Chart utility functions
const ChartUtils = {
    // Generate color palette for datasets
    generateColors: function(count) {
        const colors = [
            '#e74c3c', '#3498db', '#27ae60', '#f39c12',
            '#9b59b6', '#e67e22', '#1abc9c', '#34495e'
        ];
        
        const result = [];
        for (let i = 0; i < count; i++) {
            result.push(colors[i % colors.length]);
        }
        
        return result;
    },
    
    // Convert color to rgba with opacity
    colorToRgba: function(color, opacity) {
        const div = document.createElement('div');
        div.style.color = color;
        document.body.appendChild(div);
        const rgb = window.getComputedStyle(div).color;
        document.body.removeChild(div);
        
        return rgb.replace('rgb', 'rgba').replace(')', `, ${opacity})`);
    },
    
    // Format numbers for display
    formatNumber: function(num, decimals = 2) {
        if (Math.abs(num) < 0.001) {
            return num.toExponential(2);
        }
        return num.toFixed(decimals);
    },
    
    // Calculate moving average
    movingAverage: function(data, windowSize) {
        const result = [];
        for (let i = 0; i < data.length; i++) {
            const start = Math.max(0, i - windowSize + 1);
            const window = data.slice(start, i + 1);
            const avg = window.reduce((sum, val) => sum + val, 0) / window.length;
            result.push(avg);
        }
        return result;
    },
    
    // Detect trends in data
    detectTrend: function(data, minLength = 5) {
        if (data.length < minLength) return 'insufficient_data';
        
        const recent = data.slice(-minLength);
        const increasing = recent.every((val, i) => i === 0 || val >= recent[i - 1]);
        const decreasing = recent.every((val, i) => i === 0 || val <= recent[i - 1]);
        
        if (increasing) return 'increasing';
        if (decreasing) return 'decreasing';
        return 'stable';
    }
};

// Export for use in other modules
window.ChartManager = ChartManager;
window.ChartUtils = ChartUtils;