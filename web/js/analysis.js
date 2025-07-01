// Analysis page functionality for BVC Research Platform

class AnalysisManager {
    constructor() {
        this.currentFilters = {
            models: ['bert', 'vulseeker', 'safe', 'sense'],
            metric: 'auc',
            program: 'all',
            architecture: 'all',
            optimization: 'all'
        };
        this.currentData = null;
        this.currentPage = 1;
        this.itemsPerPage = 10;
        this.charts = {};
    }

    initialize() {
        this.currentData = dataManager.getFilteredData();
        this.updateStatistics();
        this.createDistributionChart();
        this.createBoxPlotChart();
        this.createCorrelationChart();
        this.updateDataTable();
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Search functionality
        const searchInput = document.getElementById('data-search');
        if (searchInput) {
            searchInput.addEventListener('input', () => {
                this.updateDataTable();
            });
        }

        // Sort functionality
        const sortSelect = document.getElementById('sort-by');
        if (sortSelect) {
            sortSelect.addEventListener('change', () => {
                this.updateDataTable();
            });
        }

        // Table header sorting
        document.querySelectorAll('th[data-sort]').forEach(header => {
            header.addEventListener('click', () => {
                const sortBy = header.dataset.sort;
                this.sortTable(sortBy);
            });
        });

        // Filter controls
        const programSelect = document.getElementById('analysis-program-select');
        const archSelect = document.getElementById('analysis-arch-select');
        const optSelect = document.getElementById('analysis-opt-select');

        [programSelect, archSelect, optSelect].forEach(select => {
            if (select) {
                select.addEventListener('change', () => {
                    this.updateMainFilters();
                });
            }
        });
    }

    applyFilters(filters) {
        this.currentFilters = { ...this.currentFilters, ...filters };
        this.updateData();
    }

    updateMainFilters() {
        const programSelect = document.getElementById('analysis-program-select');
        const archSelect = document.getElementById('analysis-arch-select');
        const optSelect = document.getElementById('analysis-opt-select');

        const filters = {
            program: programSelect?.value || 'all',
            architecture: archSelect?.value || 'all',
            optimization: optSelect?.value || 'all'
        };

        this.currentFilters = { ...this.currentFilters, ...filters };
        dataManager.applyFilters(filters);
        this.updateData();
    }

    updateData() {
        this.currentData = dataManager.getFilteredData();
        this.updateStatistics();
        this.updateCharts();
        this.updateDataTable();
    }

    updateStatistics() {
        if (!this.currentData) return;

        const exportData = dataManager.getExportData();
        const filteredData = this.filterDataByModels(exportData);
        
        // Calculate statistics
        const totalDataPoints = filteredData.length;
        const metricValues = filteredData.map(item => parseFloat(item[this.currentFilters.metric]));
        
        const bestPerformance = Math.max(...metricValues);
        const averagePerformance = metricValues.reduce((a, b) => a + b, 0) / metricValues.length;
        const std = this.calculateStandardDeviation(metricValues);

        // Update UI
        document.getElementById('total-data-points').textContent = totalDataPoints;
        document.getElementById('best-performance').textContent = bestPerformance.toFixed(4);
        document.getElementById('average-performance').textContent = averagePerformance.toFixed(4);
        document.getElementById('performance-std').textContent = std.toFixed(4);

        // Update description
        const bestItem = filteredData.find(item => parseFloat(item[this.currentFilters.metric]) === bestPerformance);
        if (bestItem) {
            const description = `${bestItem.model}在${bestItem.program} ${bestItem.architecture.toUpperCase()} ${bestItem.optimization}上的${this.getMetricName(this.currentFilters.metric)}`;
            document.querySelector('#best-performance').nextElementSibling.textContent = description;
        }
    }

    filterDataByModels(data) {
        return data.filter(item => this.currentFilters.models.includes(item.model.toLowerCase()));
    }

    calculateStandardDeviation(values) {
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const squaredDiffs = values.map(value => Math.pow(value - mean, 2));
        const avgSquaredDiff = squaredDiffs.reduce((a, b) => a + b, 0) / squaredDiffs.length;
        return Math.sqrt(avgSquaredDiff);
    }

    getMetricName(metric) {
        const names = {
            auc: 'AUC',
            precision: '精确率',
            recall: '召回率',
            f1: 'F1-Score',
            accuracy: '准确率'
        };
        return names[metric] || metric.toUpperCase();
    }

    createDistributionChart() {
        const ctx = document.getElementById('distribution-chart');
        if (!ctx) return;

        const exportData = dataManager.getExportData();
        const filteredData = this.filterDataByModels(exportData);
        const values = filteredData.map(item => parseFloat(item[this.currentFilters.metric]));

        // Create histogram
        const binCount = 10;
        const min = Math.min(...values);
        const max = Math.max(...values);
        const binWidth = (max - min) / binCount;
        
        const bins = Array(binCount).fill(0);
        const binLabels = [];
        
        for (let i = 0; i < binCount; i++) {
            const binStart = min + i * binWidth;
            const binEnd = min + (i + 1) * binWidth;
            binLabels.push(`${binStart.toFixed(3)}-${binEnd.toFixed(3)}`);
            
            values.forEach(value => {
                if (value >= binStart && (value < binEnd || i === binCount - 1)) {
                    bins[i]++;
                }
            });
        }

        this.charts.distribution = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: binLabels,
                datasets: [{
                    label: '频次',
                    data: bins,
                    backgroundColor: 'rgba(46, 75, 198, 0.6)',
                    borderColor: '#2E4BC6',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: '频次'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: this.getMetricName(this.currentFilters.metric)
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }

    createBoxPlotChart() {
        const ctx = document.getElementById('boxplot-chart');
        if (!ctx) return;

        const exportData = dataManager.getExportData();
        const modelData = {};
        
        this.currentFilters.models.forEach(model => {
            const modelItems = exportData.filter(item => item.model.toLowerCase() === model);
            const values = modelItems.map(item => parseFloat(item[this.currentFilters.metric]));
            modelData[model] = this.calculateBoxPlotStats(values);
        });

        // Simple box plot using bar chart
        const labels = Object.keys(modelData);
        const data = {
            labels: labels.map(l => l.toUpperCase()),
            datasets: [
                {
                    label: '最小值',
                    data: labels.map(model => modelData[model].min),
                    backgroundColor: 'rgba(255, 107, 53, 0.3)',
                    borderColor: '#FF6B35',
                    borderWidth: 1
                },
                {
                    label: '第一四分位数',
                    data: labels.map(model => modelData[model].q1),
                    backgroundColor: 'rgba(46, 75, 198, 0.3)',
                    borderColor: '#2E4BC6',
                    borderWidth: 1
                },
                {
                    label: '中位数',
                    data: labels.map(model => modelData[model].median),
                    backgroundColor: 'rgba(40, 167, 69, 0.6)',
                    borderColor: '#28a745',
                    borderWidth: 2
                },
                {
                    label: '第三四分位数',
                    data: labels.map(model => modelData[model].q3),
                    backgroundColor: 'rgba(111, 66, 193, 0.3)',
                    borderColor: '#6f42c1',
                    borderWidth: 1
                },
                {
                    label: '最大值',
                    data: labels.map(model => modelData[model].max),
                    backgroundColor: 'rgba(32, 201, 151, 0.3)',
                    borderColor: '#20c997',
                    borderWidth: 1
                }
            ]
        };

        this.charts.boxplot = new Chart(ctx, {
            type: 'bar',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        title: {
                            display: true,
                            text: this.getMetricName(this.currentFilters.metric)
                        }
                    }
                },
                plugins: {
                    legend: {
                        position: 'top'
                    }
                }
            }
        });
    }

    calculateBoxPlotStats(values) {
        const sorted = values.sort((a, b) => a - b);
        const len = sorted.length;
        
        return {
            min: sorted[0],
            q1: sorted[Math.floor(len * 0.25)],
            median: sorted[Math.floor(len * 0.5)],
            q3: sorted[Math.floor(len * 0.75)],
            max: sorted[len - 1]
        };
    }

    createCorrelationChart() {
        const ctx = document.getElementById('correlation-chart');
        if (!ctx) return;

        const exportData = dataManager.getExportData();
        const filteredData = this.filterDataByModels(exportData);
        
        const metrics = ['auc', 'precision', 'recall', 'f1', 'accuracy'];
        const correlationMatrix = this.calculateCorrelationMatrix(filteredData, metrics);

        // Create heatmap using scatter plot
        const data = [];
        const labels = [];
        
        for (let i = 0; i < metrics.length; i++) {
            for (let j = 0; j < metrics.length; j++) {
                data.push({
                    x: j,
                    y: i,
                    v: correlationMatrix[i][j]
                });
            }
        }

        this.charts.correlation = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: '相关性',
                    data: data,
                    backgroundColor: (context) => {
                        const value = context.parsed.v;
                        const alpha = Math.abs(value);
                        return value > 0 ? `rgba(46, 75, 198, ${alpha})` : `rgba(255, 107, 53, ${alpha})`;
                    },
                    pointRadius: 20
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        position: 'bottom',
                        min: -0.5,
                        max: metrics.length - 0.5,
                        ticks: {
                            stepSize: 1,
                            callback: function(value) {
                                return metrics[value] ? metrics[value].toUpperCase() : '';
                            }
                        }
                    },
                    y: {
                        min: -0.5,
                        max: metrics.length - 0.5,
                        ticks: {
                            stepSize: 1,
                            callback: function(value) {
                                return metrics[value] ? metrics[value].toUpperCase() : '';
                            }
                        }
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const xMetric = metrics[context.parsed.x];
                                const yMetric = metrics[context.parsed.y];
                                const correlation = context.parsed.v;
                                return `${xMetric} vs ${yMetric}: ${correlation.toFixed(3)}`;
                            }
                        }
                    }
                }
            }
        });
    }

    calculateCorrelationMatrix(data, metrics) {
        const matrix = [];
        
        for (let i = 0; i < metrics.length; i++) {
            matrix[i] = [];
            for (let j = 0; j < metrics.length; j++) {
                const values1 = data.map(item => parseFloat(item[metrics[i]]));
                const values2 = data.map(item => parseFloat(item[metrics[j]]));
                matrix[i][j] = this.calculateCorrelation(values1, values2);
            }
        }
        
        return matrix;
    }

    calculateCorrelation(x, y) {
        const n = x.length;
        const sumX = x.reduce((a, b) => a + b, 0);
        const sumY = y.reduce((a, b) => a + b, 0);
        const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
        const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
        const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);
        
        const numerator = n * sumXY - sumX * sumY;
        const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
        
        return denominator === 0 ? 0 : numerator / denominator;
    }

    updateCharts() {
        Object.values(this.charts).forEach(chart => {
            if (chart) chart.destroy();
        });
        this.charts = {};

        this.createDistributionChart();
        this.createBoxPlotChart();
        this.createCorrelationChart();
    }

    updateDataTable() {
        const tableBody = document.getElementById('analysis-table-body');
        if (!tableBody) return;

        let data = dataManager.getExportData();
        data = this.filterDataByModels(data);

        // Apply search filter
        const searchTerm = document.getElementById('data-search')?.value.toLowerCase() || '';
        if (searchTerm) {
            data = data.filter(item => 
                Object.values(item).some(value => 
                    value.toString().toLowerCase().includes(searchTerm)
                )
            );
        }

        // Apply sorting
        const sortBy = document.getElementById('sort-by')?.value || 'auc-desc';
        const [metric, direction] = sortBy.split('-');
        
        data.sort((a, b) => {
            const aVal = parseFloat(a[metric]);
            const bVal = parseFloat(b[metric]);
            return direction === 'desc' ? bVal - aVal : aVal - bVal;
        });

        // Add ranking
        data.forEach((item, index) => {
            item.rank = index + 1;
        });

        // Pagination
        const totalPages = Math.ceil(data.length / this.itemsPerPage);
        const startIndex = (this.currentPage - 1) * this.itemsPerPage;
        const endIndex = startIndex + this.itemsPerPage;
        const pageData = data.slice(startIndex, endIndex);

        // Update table
        tableBody.innerHTML = pageData.map(row => `
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
                <td class="number">#${row.rank}</td>
            </tr>
        `).join('');

        // Update pagination
        this.updatePagination(totalPages);
    }

    updatePagination(totalPages) {
        const pagination = document.getElementById('table-pagination');
        if (!pagination) return;

        const buttons = [];
        
        // Previous button
        if (this.currentPage > 1) {
            buttons.push(`<button onclick="analysisManager.changePage(${this.currentPage - 1})">上一页</button>`);
        }

        // Page numbers
        for (let i = 1; i <= totalPages; i++) {
            const active = i === this.currentPage ? 'active' : '';
            buttons.push(`<button class="${active}" onclick="analysisManager.changePage(${i})">${i}</button>`);
        }

        // Next button
        if (this.currentPage < totalPages) {
            buttons.push(`<button onclick="analysisManager.changePage(${this.currentPage + 1})">下一页</button>`);
        }

        pagination.innerHTML = buttons.join('');
    }

    changePage(page) {
        this.currentPage = page;
        this.updateDataTable();
    }

    sortTable(sortBy) {
        const currentSort = document.getElementById('sort-by').value;
        const [currentMetric, currentDirection] = currentSort.split('-');
        
        let newDirection = 'desc';
        if (currentMetric === sortBy && currentDirection === 'desc') {
            newDirection = 'asc';
        }

        document.getElementById('sort-by').value = `${sortBy}-${newDirection}`;
        this.updateDataTable();
    }
}

// Export functions for global access
function exportFilteredData() {
    const data = dataManager.getExportData();
    const filteredData = analysisManager.filterDataByModels(data);
    
    const headers = ['Model', 'Program', 'Architecture', 'Optimization', 'AUC', 'Precision', 'Recall', 'F1-Score', 'Accuracy'];
    const csvContent = [
        headers.join(','),
        ...filteredData.map(row => [
            row.model, row.program, row.architecture, row.optimization,
            row.auc, row.precision, row.recall, row.f1, row.accuracy
        ].join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'bvc_filtered_analysis.csv';
    link.click();
}

function exportStatistics() {
    const data = dataManager.getExportData();
    const filteredData = analysisManager.filterDataByModels(data);
    
    const report = `BVC Research Platform - Statistical Report
Generated: ${new Date().toLocaleDateString()}

Data Summary:
- Total data points: ${filteredData.length}
- Models analyzed: ${analysisManager.currentFilters.models.join(', ')}
- Metric: ${analysisManager.getMetricName(analysisManager.currentFilters.metric)}

Performance Statistics:
- Best performance: ${document.getElementById('best-performance').textContent}
- Average performance: ${document.getElementById('average-performance').textContent}
- Standard deviation: ${document.getElementById('performance-std').textContent}

Filter Settings:
- Program: ${analysisManager.currentFilters.program}
- Architecture: ${analysisManager.currentFilters.architecture}
- Optimization: ${analysisManager.currentFilters.optimization}
`;

    const blob = new Blob([report], { type: 'text/plain;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'bvc_statistics_report.txt';
    link.click();
}

function generateComparison() {
    const model1 = document.getElementById('compare-model-1')?.value;
    const model2 = document.getElementById('compare-model-2')?.value;
    
    if (!model1 || !model2) return;
    
    const resultsDiv = document.getElementById('comparison-results');
    if (!resultsDiv) return;
    
    const data = dataManager.getExportData();
    const model1Data = data.filter(item => item.model.toLowerCase() === model1);
    const model2Data = data.filter(item => item.model.toLowerCase() === model2);
    
    const metrics = ['auc', 'precision', 'recall', 'f1', 'accuracy'];
    const model1Avg = {};
    const model2Avg = {};
    
    metrics.forEach(metric => {
        model1Avg[metric] = model1Data.reduce((sum, item) => sum + parseFloat(item[metric]), 0) / model1Data.length;
        model2Avg[metric] = model2Data.reduce((sum, item) => sum + parseFloat(item[metric]), 0) / model2Data.length;
    });
    
    resultsDiv.innerHTML = `
        <div class="comparison-chart">
            <h3>${model1.toUpperCase()} vs ${model2.toUpperCase()} 性能对比</h3>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>指标</th>
                        <th class="number">${model1.toUpperCase()}</th>
                        <th class="number">${model2.toUpperCase()}</th>
                        <th class="number">差异</th>
                    </tr>
                </thead>
                <tbody>
                    ${metrics.map(metric => {
                        const diff = model1Avg[metric] - model2Avg[metric];
                        const diffClass = diff > 0 ? 'positive' : 'negative';
                        return `
                            <tr>
                                <td>${analysisManager.getMetricName(metric)}</td>
                                <td class="number">${model1Avg[metric].toFixed(4)}</td>
                                <td class="number">${model2Avg[metric].toFixed(4)}</td>
                                <td class="number ${diffClass}">${diff > 0 ? '+' : ''}${diff.toFixed(4)}</td>
                            </tr>
                        `;
                    }).join('')}
                </tbody>
            </table>
        </div>
    `;
}

// Create analysis manager instance
const analysisManager = new AnalysisManager();