// Chart.js configuration and rendering for BVC Research Platform

// Chart color scheme
const chartColors = {
    bert: '#2E4BC6',
    vulseeker: '#FF6B35', 
    safe: '#28a745',
    sense: '#6f42c1',
    fit: '#20c997'
};

const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            position: 'top',
            labels: {
                usePointStyle: true,
                padding: 20,
                font: {
                    family: 'Source Sans Pro',
                    size: 12
                }
            }
        },
        tooltip: {
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            titleColor: '#fff',
            bodyColor: '#fff',
            borderColor: '#2E4BC6',
            borderWidth: 1,
            cornerRadius: 4,
            titleFont: {
                family: 'Source Sans Pro',
                size: 13,
                weight: 'bold'
            },
            bodyFont: {
                family: 'Source Sans Pro',
                size: 12
            }
        }
    },
    scales: {
        x: {
            grid: {
                color: 'rgba(46, 75, 198, 0.1)'
            },
            ticks: {
                font: {
                    family: 'Source Sans Pro',
                    size: 11
                }
            }
        },
        y: {
            grid: {
                color: 'rgba(46, 75, 198, 0.1)'
            },
            ticks: {
                font: {
                    family: 'Source Sans Pro',
                    size: 11
                }
            }
        }
    }
};

class ChartManager {
    constructor() {
        this.charts = {};
        this.currentData = null;
    }

    // Initialize all charts
    initializeCharts() {
        this.currentData = dataManager.getFilteredData();
        if (!this.currentData) return;

        this.createAUCChart();
        this.createPRFChart();
        this.createROCChart();
        this.createProgramPerformanceChart();
        this.createArchPerformanceChart();
        this.createTrainingProgressChart();
        this.updateMetricsCards();
        this.updatePerformanceTable();
    }

    // Update all charts when filters change
    updateCharts() {
        this.currentData = dataManager.getFilteredData();
        if (!this.currentData) return;

        Object.values(this.charts).forEach(chart => {
            if (chart) chart.destroy();
        });
        this.charts = {};

        this.initializeCharts();
    }

    // Create AUC comparison chart
    createAUCChart() {
        const ctx = document.getElementById('auc-chart');
        if (!ctx) return;

        const averages = dataManager.calculateAverages(this.currentData);
        if (!averages) return;

        const data = {
            labels: Object.keys(averages.models),
            datasets: [{
                label: 'AUC Score',
                data: Object.values(averages.models).map(model => model.auc),
                backgroundColor: Object.keys(averages.models).map(model => 
                    chartColors[model.toLowerCase()] || '#6c757d'
                ),
                borderColor: Object.keys(averages.models).map(model => 
                    chartColors[model.toLowerCase()] || '#6c757d'
                ),
                borderWidth: 2,
                borderRadius: 4
            }]
        };

        this.charts.auc = new Chart(ctx, {
            type: 'bar',
            data: data,
            options: {
                ...chartOptions,
                scales: {
                    ...chartOptions.scales,
                    y: {
                        ...chartOptions.scales.y,
                        beginAtZero: false,
                        min: 0.85,
                        max: 1.0,
                        ticks: {
                            ...chartOptions.scales.y.ticks,
                            callback: function(value) {
                                return value.toFixed(3);
                            }
                        }
                    }
                },
                plugins: {
                    ...chartOptions.plugins,
                    tooltip: {
                        ...chartOptions.plugins.tooltip,
                        callbacks: {
                            label: function(context) {
                                return `${context.dataset.label}: ${context.parsed.y.toFixed(4)}`;
                            }
                        }
                    }
                }
            }
        });
    }

    // Create Precision-Recall-F1 comparison chart
    createPRFChart() {
        const ctx = document.getElementById('prf-chart');
        if (!ctx) return;

        const averages = dataManager.calculateAverages(this.currentData);
        if (!averages) return;

        const models = Object.keys(averages.models);
        const data = {
            labels: models,
            datasets: [
                {
                    label: '精确率 (Precision)',
                    data: models.map(model => averages.models[model].precision),
                    backgroundColor: 'rgba(46, 75, 198, 0.6)',
                    borderColor: '#2E4BC6',
                    borderWidth: 2
                },
                {
                    label: '召回率 (Recall)',
                    data: models.map(model => averages.models[model].recall),
                    backgroundColor: 'rgba(255, 107, 53, 0.6)',
                    borderColor: '#FF6B35',
                    borderWidth: 2
                },
                {
                    label: 'F1-Score',
                    data: models.map(model => averages.models[model].f1),
                    backgroundColor: 'rgba(40, 167, 69, 0.6)',
                    borderColor: '#28a745',
                    borderWidth: 2
                }
            ]
        };

        this.charts.prf = new Chart(ctx, {
            type: 'radar',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        beginAtZero: false,
                        min: 0.8,
                        max: 1.0,
                        ticks: {
                            stepSize: 0.05,
                            callback: function(value) {
                                return value.toFixed(2);
                            },
                            font: {
                                family: 'Source Sans Pro',
                                size: 10
                            }
                        },
                        grid: {
                            color: 'rgba(46, 75, 198, 0.1)'
                        },
                        pointLabels: {
                            font: {
                                family: 'Source Sans Pro',
                                size: 12
                            }
                        }
                    }
                },
                plugins: {
                    ...chartOptions.plugins,
                    tooltip: {
                        ...chartOptions.plugins.tooltip,
                        callbacks: {
                            label: function(context) {
                                return `${context.dataset.label}: ${context.parsed.r.toFixed(4)}`;
                            }
                        }
                    }
                }
            }
        });
    }

    // Create ROC curve chart
    createROCChart() {
        const ctx = document.getElementById('roc-chart');
        if (!ctx) return;

        const rocData = this.currentData.rocData;
        const datasets = Object.keys(rocData).map(model => ({
            label: model,
            data: rocData[model].fpr.map((fpr, index) => ({
                x: fpr,
                y: rocData[model].tpr[index]
            })),
            borderColor: chartColors[model.toLowerCase()] || '#6c757d',
            backgroundColor: 'transparent',
            borderWidth: 3,
            pointRadius: 4,
            pointBackgroundColor: chartColors[model.toLowerCase()] || '#6c757d',
            pointBorderColor: '#fff',
            pointBorderWidth: 2,
            tension: 0.1
        }));

        // Add diagonal reference line
        datasets.push({
            label: '随机分类器',
            data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
            borderColor: '#999',
            backgroundColor: 'transparent',
            borderWidth: 2,
            borderDash: [5, 5],
            pointRadius: 0
        });

        this.charts.roc = new Chart(ctx, {
            type: 'line',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        min: 0,
                        max: 1,
                        title: {
                            display: true,
                            text: '假正率 (False Positive Rate)',
                            font: {
                                family: 'Source Sans Pro',
                                size: 14,
                                weight: 'bold'
                            }
                        },
                        grid: {
                            color: 'rgba(46, 75, 198, 0.1)'
                        },
                        ticks: {
                            stepSize: 0.2,
                            font: {
                                family: 'Source Sans Pro',
                                size: 11
                            }
                        }
                    },
                    y: {
                        min: 0,
                        max: 1,
                        title: {
                            display: true,
                            text: '真正率 (True Positive Rate)',
                            font: {
                                family: 'Source Sans Pro',
                                size: 14,
                                weight: 'bold'
                            }
                        },
                        grid: {
                            color: 'rgba(46, 75, 198, 0.1)'
                        },
                        ticks: {
                            stepSize: 0.2,
                            font: {
                                family: 'Source Sans Pro',
                                size: 11
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            usePointStyle: true,
                            padding: 20,
                            font: {
                                family: 'Source Sans Pro',
                                size: 12
                            }
                        }
                    },
                    tooltip: {
                        ...chartOptions.plugins.tooltip,
                        callbacks: {
                            label: function(context) {
                                return `${context.dataset.label}: (${context.parsed.x.toFixed(3)}, ${context.parsed.y.toFixed(3)})`;
                            }
                        }
                    }
                }
            }
        });
    }

    // Create program performance chart
    createProgramPerformanceChart() {
        const ctx = document.getElementById('program-performance-chart');
        if (!ctx) return;

        const programData = dataManager.getPerformanceByProgram();
        if (!programData) return;

        const programs = Object.keys(programData);
        const models = this.currentData.models;

        const datasets = models.map(model => ({
            label: model,
            data: programs.map(program => programData[program][model]),
            backgroundColor: chartColors[model.toLowerCase()] || '#6c757d',
            borderColor: chartColors[model.toLowerCase()] || '#6c757d',
            borderWidth: 2,
            borderRadius: 4
        }));

        this.charts.programPerformance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: programs.map(p => p.charAt(0).toUpperCase() + p.slice(1)),
                datasets: datasets
            },
            options: {
                ...chartOptions,
                scales: {
                    ...chartOptions.scales,
                    y: {
                        ...chartOptions.scales.y,
                        beginAtZero: false,
                        min: 0.85,
                        max: 1.0,
                        title: {
                            display: true,
                            text: 'AUC Score',
                            font: {
                                family: 'Source Sans Pro',
                                size: 14,
                                weight: 'bold'
                            }
                        }
                    },
                    x: {
                        ...chartOptions.scales.x,
                        title: {
                            display: true,
                            text: '测试程序',
                            font: {
                                family: 'Source Sans Pro',
                                size: 14,
                                weight: 'bold'
                            }
                        }
                    }
                }
            }
        });
    }

    // Create architecture performance chart
    createArchPerformanceChart() {
        const ctx = document.getElementById('arch-performance-chart');
        if (!ctx) return;

        const archData = dataManager.getPerformanceByArchitecture();
        if (!archData) return;

        const architectures = Object.keys(archData);
        const models = this.currentData.models;

        const datasets = models.map(model => ({
            label: model,
            data: architectures.map(arch => archData[arch][model]),
            backgroundColor: chartColors[model.toLowerCase()] || '#6c757d',
            borderColor: chartColors[model.toLowerCase()] || '#6c757d',
            borderWidth: 2,
            borderRadius: 4
        }));

        this.charts.archPerformance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: architectures.map(a => a.toUpperCase()),
                datasets: datasets
            },
            options: {
                ...chartOptions,
                scales: {
                    ...chartOptions.scales,
                    y: {
                        ...chartOptions.scales.y,
                        beginAtZero: false,
                        min: 0.85,
                        max: 1.0,
                        title: {
                            display: true,
                            text: 'AUC Score',
                            font: {
                                family: 'Source Sans Pro',
                                size: 14,
                                weight: 'bold'
                            }
                        }
                    },
                    x: {
                        ...chartOptions.scales.x,
                        title: {
                            display: true,
                            text: '处理器架构',
                            font: {
                                family: 'Source Sans Pro',
                                size: 14,
                                weight: 'bold'
                            }
                        }
                    }
                }
            }
        });
    }

    // Create training progress chart
    createTrainingProgressChart() {
        const ctx = document.getElementById('training-progress-chart');
        if (!ctx) return;

        const trainingData = this.currentData.trainingProgress;
        const models = Object.keys(trainingData);

        const datasets = [];
        
        // Loss datasets
        models.forEach(model => {
            datasets.push({
                label: `${model} Loss`,
                data: trainingData[model].epochs.map((epoch, index) => ({
                    x: epoch,
                    y: trainingData[model].loss[index]
                })),
                borderColor: chartColors[model.toLowerCase()] || '#6c757d',
                backgroundColor: 'transparent',
                borderWidth: 2,
                yAxisID: 'y',
                tension: 0.1
            });
        });

        // Accuracy datasets
        models.forEach(model => {
            datasets.push({
                label: `${model} Accuracy`,
                data: trainingData[model].epochs.map((epoch, index) => ({
                    x: epoch,
                    y: trainingData[model].accuracy[index]
                })),
                borderColor: chartColors[model.toLowerCase()] || '#6c757d',
                backgroundColor: 'transparent',
                borderWidth: 2,
                borderDash: [5, 5],
                yAxisID: 'y1',
                tension: 0.1
            });
        });

        this.charts.trainingProgress = new Chart(ctx, {
            type: 'line',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        title: {
                            display: true,
                            text: '训练轮次 (Epochs)',
                            font: {
                                family: 'Source Sans Pro',
                                size: 14,
                                weight: 'bold'
                            }
                        },
                        grid: {
                            color: 'rgba(46, 75, 198, 0.1)'
                        }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: '损失值 (Loss)',
                            font: {
                                family: 'Source Sans Pro',
                                size: 14,
                                weight: 'bold'
                            }
                        },
                        grid: {
                            color: 'rgba(46, 75, 198, 0.1)'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: true,
                        position: 'right',
                        title: {
                            display: true,
                            text: '准确率 (Accuracy)',
                            font: {
                                family: 'Source Sans Pro',
                                size: 14,
                                weight: 'bold'
                            }
                        },
                        grid: {
                            drawOnChartArea: false,
                        },
                        min: 0.5,
                        max: 1.0
                    }
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            usePointStyle: true,
                            padding: 20,
                            font: {
                                family: 'Source Sans Pro',
                                size: 11
                            }
                        }
                    },
                    tooltip: {
                        ...chartOptions.plugins.tooltip,
                        callbacks: {
                            label: function(context) {
                                const value = context.parsed.y.toFixed(4);
                                return `${context.dataset.label}: ${value}`;
                            }
                        }
                    }
                }
            }
        });
    }

    // Update metrics cards
    updateMetricsCards() {
        const averages = dataManager.calculateAverages(this.currentData);
        if (!averages) return;

        const overall = averages.overall;
        
        document.getElementById('avg-auc').textContent = overall.auc.toFixed(3);
        document.getElementById('avg-precision').textContent = overall.precision.toFixed(3);
        document.getElementById('avg-recall').textContent = overall.recall.toFixed(3);
        document.getElementById('avg-f1').textContent = overall.f1.toFixed(3);
    }

    // Update performance table
    updatePerformanceTable() {
        const tableBody = document.getElementById('performance-table-body');
        if (!tableBody) return;

        const exportData = dataManager.getExportData();
        
        tableBody.innerHTML = exportData.map(row => `
            <tr>
                <td><strong>${row.model}</strong></td>
                <td>${row.program}</td>
                <td>${row.architecture.toUpperCase()}</td>
                <td>${row.optimization}</td>
                <td class="number">${row.auc}</td>
                <td class="number">${row.precision}</td>
                <td class="number">${row.recall}</td>
                <td class="number">${row.f1}</td>
                <td class="number">${row.accuracy}</td>
            </tr>
        `).join('');
    }
}

// Export chart manager instance
const chartManager = new ChartManager();

// Export functions for CSV and PDF
function exportToCSV() {
    const data = dataManager.getExportData();
    const headers = ['Model', 'Program', 'Architecture', 'Optimization', 'AUC', 'Precision', 'Recall', 'F1-Score', 'Accuracy'];
    
    const csvContent = [
        headers.join(','),
        ...data.map(row => [
            row.model, row.program, row.architecture, row.optimization,
            row.auc, row.precision, row.recall, row.f1, row.accuracy
        ].join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'bvc_performance_results.csv';
    link.click();
}

function exportToPDF() {
    // Simple implementation - in a real app you'd use a PDF library
    alert('PDF导出功能正在开发中。请使用CSV导出或浏览器打印功能。');
}