// Dataset management functionality
class DatasetManager {
    constructor() {
        this.datasets = [];
        this.filteredDatasets = [];
        this.selectedDataset = null;
        this.currentView = 'list';
        this.charts = {};
        
        this.initializeElements();
        this.setupEventListeners();
        this.loadDatasets();
        this.updateStatistics();
    }

    initializeElements() {
        // Filter elements
        this.architectureFilter = document.getElementById('architectureFilter');
        this.optimizationFilter = document.getElementById('optimizationFilter');
        this.searchInput = document.getElementById('searchDataset');
        
        // View elements
        this.datasetList = document.getElementById('datasetList');
        this.datasetDetails = document.getElementById('datasetDetails');
        this.dataAnalysis = document.getElementById('dataAnalysis');
        this.datasetComparison = document.getElementById('datasetComparison');
        
        // Upload elements
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('datasetFile');
        this.filePreview = document.getElementById('filePreview');
        this.uploadProgress = document.getElementById('uploadProgress');
        this.uploadButton = document.getElementById('uploadButton');
        
        // Statistics elements
        this.totalDatasets = document.getElementById('totalDatasets');
        this.totalSamples = document.getElementById('totalSamples');
        this.totalArchitectures = document.getElementById('totalArchitectures');
        this.totalSize = document.getElementById('totalSize');
    }

    setupEventListeners() {
        // Filter controls
        document.getElementById('applyFilters').addEventListener('click', () => this.applyFilters());
        document.getElementById('clearFilters').addEventListener('click', () => this.clearFilters());
        this.searchInput.addEventListener('input', () => this.applyFilters());
        
        // View toggle
        document.getElementById('listView').addEventListener('click', () => this.setView('list'));
        document.getElementById('gridView').addEventListener('click', () => this.setView('grid'));
        
        // Upload functionality
        this.setupUploadListeners();
        
        // Dataset actions
        this.setupDatasetActions();
        
        // Preprocessing
        this.setupPreprocessingListeners();
        
        // Data split ratio updates
        this.setupDataSplitListeners();
    }

    setupUploadListeners() {
        // Drag and drop
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.classList.add('dragover');
        });
        this.uploadArea.addEventListener('dragleave', () => {
            this.uploadArea.classList.remove('dragover');
        });
        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelection(files[0]);
            }
        });
        
        // File selection
        this.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelection(e.target.files[0]);
            }
        });
        
        // Upload button
        this.uploadButton.addEventListener('click', () => this.startUpload());
    }

    setupDatasetActions() {
        document.getElementById('preprocessDataset').addEventListener('click', () => this.openPreprocessModal());
        document.getElementById('downloadDataset').addEventListener('click', () => this.downloadDataset());
        document.getElementById('deleteDataset').addEventListener('click', () => this.deleteDataset());
    }

    setupPreprocessingListeners() {
        document.getElementById('startPreprocessing').addEventListener('click', () => this.startPreprocessing());
    }

    setupDataSplitListeners() {
        const trainRatio = document.getElementById('trainRatio');
        const valRatio = document.getElementById('valRatio');
        const testRatio = document.getElementById('testRatio');
        
        const updateRatios = () => {
            const train = parseFloat(trainRatio.value);
            const val = parseFloat(valRatio.value);
            const test = 1 - train - val;
            
            if (test >= 0 && test <= 1) {
                testRatio.value = test.toFixed(1);
                this.updateSplitVisualization(train, val, test);
            }
        };
        
        trainRatio.addEventListener('input', updateRatios);
        valRatio.addEventListener('input', updateRatios);
        
        // Initial visualization
        updateRatios();
    }

    updateSplitVisualization(train, val, test) {
        const visualization = document.querySelector('.split-visualization') || this.createSplitVisualization();
        
        visualization.querySelector('.split-train').style.width = `${train * 100}%`;
        visualization.querySelector('.split-val').style.width = `${val * 100}%`;
        visualization.querySelector('.split-test').style.width = `${test * 100}%`;
        
        visualization.querySelector('.split-train').textContent = `训练 ${(train * 100).toFixed(0)}%`;
        visualization.querySelector('.split-val').textContent = `验证 ${(val * 100).toFixed(0)}%`;
        visualization.querySelector('.split-test').textContent = `测试 ${(test * 100).toFixed(0)}%`;
    }

    createSplitVisualization() {
        const container = document.getElementById('preprocessForm');
        const visualization = document.createElement('div');
        visualization.className = 'split-visualization';
        visualization.innerHTML = `
            <div class="split-train"></div>
            <div class="split-val"></div>
            <div class="split-test"></div>
        `;
        
        container.appendChild(visualization);
        return visualization;
    }

    async loadDatasets() {
        try {
            const response = await BVC.API.get('/datasets');
            this.datasets = response.datasets || this.getMockDatasets();
            this.filteredDatasets = [...this.datasets];
            this.renderDatasets();
            this.updateStatistics();
        } catch (error) {
            console.error('Failed to load datasets:', error);
            // Use mock data for demonstration
            this.datasets = this.getMockDatasets();
            this.filteredDatasets = [...this.datasets];
            this.renderDatasets();
            this.updateStatistics();
        }
    }

    getMockDatasets() {
        return [
            {
                id: 'openssl_1.0.1f_arm_O2',
                name: 'openssl',
                version: '1.0.1f',
                architecture: 'arm',
                optimization: 'O2',
                samples: 15420,
                size: 245.7, // MB
                positive_samples: 7850,
                negative_samples: 7570,
                created_at: '2024-01-15T10:30:00Z',
                modified_at: '2024-01-15T10:30:00Z',
                status: 'ready',
                description: 'OpenSSL库的ARM架构O2优化版本数据集'
            },
            {
                id: 'libsqlite3_0.8.6_x86_O1',
                name: 'libsqlite3',
                version: '0.8.6',
                architecture: 'x86',
                optimization: 'O1',
                samples: 8930,
                size: 156.2,
                positive_samples: 4420,
                negative_samples: 4510,
                created_at: '2024-01-10T14:20:00Z',
                modified_at: '2024-01-12T09:15:00Z',
                status: 'ready',
                description: 'SQLite数据库引擎x86架构O1优化数据集'
            },
            {
                id: 'coreutils_6.12_mips_O3',
                name: 'coreutils',
                version: '6.12',
                architecture: 'mips',
                optimization: 'O3',
                samples: 12680,
                size: 198.5,
                positive_samples: 6340,
                negative_samples: 6340,
                created_at: '2024-01-08T16:45:00Z',
                modified_at: '2024-01-08T16:45:00Z',
                status: 'ready',
                description: 'GNU核心工具集MIPS架构O3优化数据集'
            },
            {
                id: 'nginx_1.18.0_x64_O2',
                name: 'nginx',
                version: '1.18.0',
                architecture: 'x64',
                optimization: 'O2',
                samples: 22150,
                size: 387.3,
                positive_samples: 11200,
                negative_samples: 10950,
                created_at: '2024-01-20T08:30:00Z',
                modified_at: '2024-01-20T08:30:00Z',
                status: 'processing',
                description: 'Nginx Web服务器x64架构O2优化数据集'
            }
        ];
    }

    applyFilters() {
        const archFilter = this.architectureFilter.value;
        const optFilter = this.optimizationFilter.value;
        const searchTerm = this.searchInput.value.toLowerCase();
        
        this.filteredDatasets = this.datasets.filter(dataset => {
            const matchesArch = !archFilter || dataset.architecture === archFilter;
            const matchesOpt = !optFilter || dataset.optimization === optFilter;
            const matchesSearch = !searchTerm || 
                dataset.name.toLowerCase().includes(searchTerm) ||
                dataset.version.toLowerCase().includes(searchTerm) ||
                dataset.description.toLowerCase().includes(searchTerm);
            
            return matchesArch && matchesOpt && matchesSearch;
        });
        
        this.renderDatasets();
    }

    clearFilters() {
        this.architectureFilter.value = '';
        this.optimizationFilter.value = '';
        this.searchInput.value = '';
        this.filteredDatasets = [...this.datasets];
        this.renderDatasets();
    }

    setView(view) {
        this.currentView = view;
        
        // Update button states
        document.getElementById('listView').classList.toggle('active', view === 'list');
        document.getElementById('gridView').classList.toggle('active', view === 'grid');
        
        this.renderDatasets();
    }

    renderDatasets() {
        if (this.filteredDatasets.length === 0) {
            this.renderEmptyState();
            return;
        }
        
        if (this.currentView === 'list') {
            this.renderListView();
        } else {
            this.renderGridView();
        }
    }

    renderListView() {
        this.datasetList.className = 'dataset-list';
        this.datasetList.innerHTML = '';
        
        this.filteredDatasets.forEach(dataset => {
            const item = this.createDatasetListItem(dataset);
            this.datasetList.appendChild(item);
        });
    }

    renderGridView() {
        this.datasetList.className = 'dataset-grid';
        this.datasetList.innerHTML = '';
        
        this.filteredDatasets.forEach(dataset => {
            const item = this.createDatasetGridItem(dataset);
            this.datasetList.appendChild(item);
        });
    }

    createDatasetListItem(dataset) {
        const item = document.createElement('div');
        item.className = `dataset-item ${dataset.status}`;
        item.dataset.datasetId = dataset.id;
        
        item.innerHTML = `
            <div class="dataset-header">
                <h6 class="dataset-name">${dataset.name}_${dataset.version}</h6>
                <div class="dataset-badges">
                    <span class="badge bg-primary">${dataset.architecture.toUpperCase()}</span>
                    <span class="badge bg-secondary">${dataset.optimization}</span>
                    <span class="badge bg-${this.getStatusBadgeColor(dataset.status)}">${this.getStatusText(dataset.status)}</span>
                </div>
            </div>
            <div class="dataset-meta">
                <span class="dataset-samples">${BVC.Utils.formatNumber(dataset.samples)} 样本</span>
                <span class="dataset-size">${BVC.Utils.formatFileSize(dataset.size * 1024 * 1024)}</span>
            </div>
        `;
        
        item.addEventListener('click', () => this.selectDataset(dataset));
        
        return item;
    }

    createDatasetGridItem(dataset) {
        const item = document.createElement('div');
        item.className = `dataset-card ${dataset.status}`;
        item.dataset.datasetId = dataset.id;
        
        item.innerHTML = `
            <div class="dataset-header mb-3">
                <h6 class="dataset-name">${dataset.name}_${dataset.version}</h6>
                <div class="dataset-badges">
                    <span class="badge bg-primary">${dataset.architecture.toUpperCase()}</span>
                    <span class="badge bg-secondary">${dataset.optimization}</span>
                </div>
            </div>
            <div class="dataset-description mb-3">
                <p class="text-muted small">${dataset.description}</p>
            </div>
            <div class="dataset-stats">
                <div class="row text-center">
                    <div class="col-6">
                        <div class="stat-value">${BVC.Utils.formatNumber(dataset.samples)}</div>
                        <div class="stat-label">样本</div>
                    </div>
                    <div class="col-6">
                        <div class="stat-value">${BVC.Utils.formatFileSize(dataset.size * 1024 * 1024)}</div>
                        <div class="stat-label">大小</div>
                    </div>
                </div>
            </div>
            <div class="dataset-status mt-3">
                <span class="badge bg-${this.getStatusBadgeColor(dataset.status)} w-100">${this.getStatusText(dataset.status)}</span>
            </div>
        `;
        
        item.addEventListener('click', () => this.selectDataset(dataset));
        
        return item;
    }

    renderEmptyState() {
        this.datasetList.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-search"></i>
                <h5>未找到数据集</h5>
                <p>尝试调整筛选条件或上传新的数据集</p>
            </div>
        `;
    }

    selectDataset(dataset) {
        // Update selection state
        document.querySelectorAll('.dataset-item, .dataset-card').forEach(item => {
            item.classList.remove('active');
        });
        
        const selectedItem = document.querySelector(`[data-dataset-id="${dataset.id}"]`);
        if (selectedItem) {
            selectedItem.classList.add('active');
        }
        
        this.selectedDataset = dataset;
        this.showDatasetDetails(dataset);
        this.showDataAnalysis(dataset);
    }

    showDatasetDetails(dataset) {
        document.getElementById('detailName').textContent = `${dataset.name}_${dataset.version}`;
        document.getElementById('detailArchitecture').textContent = dataset.architecture.toUpperCase();
        document.getElementById('detailOptimization').textContent = dataset.optimization;
        document.getElementById('detailSamples').textContent = BVC.Utils.formatNumber(dataset.samples);
        document.getElementById('detailSize').textContent = BVC.Utils.formatFileSize(dataset.size * 1024 * 1024);
        document.getElementById('detailCreated').textContent = new Date(dataset.created_at).toLocaleString('zh-CN');
        document.getElementById('detailModified').textContent = new Date(dataset.modified_at).toLocaleString('zh-CN');
        document.getElementById('detailStatus').innerHTML = `<span class="badge bg-${this.getStatusBadgeColor(dataset.status)}">${this.getStatusText(dataset.status)}</span>`;
        
        this.datasetDetails.style.display = 'block';
    }

    showDataAnalysis(dataset) {
        this.createLabelDistributionChart(dataset);
        this.createArchitectureChart(dataset);
        this.updateQualityMetrics(dataset);
        this.dataAnalysis.style.display = 'block';
    }

    createLabelDistributionChart(dataset) {
        const ctx = document.getElementById('labelDistributionChart').getContext('2d');
        
        if (this.charts.labelDistribution) {
            this.charts.labelDistribution.destroy();
        }
        
        this.charts.labelDistribution = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['正样本', '负样本'],
                datasets: [{
                    data: [dataset.positive_samples, dataset.negative_samples],
                    backgroundColor: ['#27ae60', '#e74c3c'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const percentage = ((context.parsed / dataset.samples) * 100).toFixed(1);
                                return `${context.label}: ${context.parsed} (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        });
    }

    createArchitectureChart(dataset) {
        const ctx = document.getElementById('architectureChart').getContext('2d');
        
        if (this.charts.architecture) {
            this.charts.architecture.destroy();
        }
        
        // Mock data for architecture distribution across all datasets
        const archData = this.getArchitectureDistribution();
        
        this.charts.architecture = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: Object.keys(archData),
                datasets: [{
                    label: '数据集数量',
                    data: Object.values(archData),
                    backgroundColor: ['#3498db', '#e67e22', '#9b59b6', '#1abc9c'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1
                        }
                    }
                }
            }
        });
    }

    getArchitectureDistribution() {
        const distribution = {};
        this.datasets.forEach(dataset => {
            const arch = dataset.architecture.toUpperCase();
            distribution[arch] = (distribution[arch] || 0) + 1;
        });
        return distribution;
    }

    updateQualityMetrics(dataset) {
        // Calculate quality scores
        const completeness = this.calculateCompleteness(dataset);
        const balance = this.calculateBalance(dataset);
        const consistency = this.calculateConsistency(dataset);
        
        document.getElementById('completenessScore').textContent = `${(completeness * 100).toFixed(0)}%`;
        document.getElementById('balanceScore').textContent = `${(balance * 100).toFixed(0)}%`;
        document.getElementById('consistencyScore').textContent = `${(consistency * 100).toFixed(0)}%`;
        
        // Update quality classes
        this.updateQualityClass('completenessScore', completeness);
        this.updateQualityClass('balanceScore', balance);
        this.updateQualityClass('consistencyScore', consistency);
        
        // Generate quality report
        this.generateQualityReport(dataset, completeness, balance, consistency);
    }

    calculateCompleteness(dataset) {
        // Simulate completeness calculation
        return 0.95; // 95% complete
    }

    calculateBalance(dataset) {
        const positive = dataset.positive_samples;
        const negative = dataset.negative_samples;
        const total = positive + negative;
        const ratio = Math.min(positive, negative) / Math.max(positive, negative);
        return ratio;
    }

    calculateConsistency(dataset) {
        // Simulate consistency calculation
        return 0.88; // 88% consistent
    }

    updateQualityClass(elementId, score) {
        const element = document.getElementById(elementId).parentElement;
        element.classList.remove('excellent', 'good', 'poor');
        
        if (score >= 0.9) {
            element.classList.add('excellent');
        } else if (score >= 0.7) {
            element.classList.add('good');
        } else {
            element.classList.add('poor');
        }
    }

    generateQualityReport(dataset, completeness, balance, consistency) {
        const report = document.getElementById('qualityReport');
        let reportText = '';
        let alertClass = 'alert-success';
        
        if (completeness >= 0.9 && balance >= 0.8 && consistency >= 0.85) {
            reportText = '数据集质量优秀，适合直接用于模型训练。';
            alertClass = 'alert-success';
        } else if (completeness >= 0.8 && balance >= 0.6 && consistency >= 0.7) {
            reportText = '数据集质量良好，建议进行轻微预处理后使用。';
            alertClass = 'alert-warning';
        } else {
            reportText = '数据集质量需要改进，建议进行详细的数据清洗和预处理。';
            alertClass = 'alert-danger';
        }
        
        report.className = `alert ${alertClass}`;
        report.textContent = reportText;
    }

    updateStatistics() {
        const totalDatasets = this.datasets.length;
        const totalSamples = this.datasets.reduce((sum, dataset) => sum + dataset.samples, 0);
        const architectures = new Set(this.datasets.map(d => d.architecture)).size;
        const totalSize = this.datasets.reduce((sum, dataset) => sum + dataset.size, 0) / 1024; // Convert to GB
        
        // Animate counters
        this.animateCounter(this.totalDatasets, totalDatasets);
        this.animateCounter(this.totalSamples, totalSamples);
        this.animateCounter(this.totalArchitectures, architectures);
        this.animateCounter(this.totalSize, totalSize, 1);
    }

    animateCounter(element, target, decimals = 0) {
        const duration = 2000;
        const increment = target / (duration / 16);
        let current = 0;
        
        const timer = setInterval(() => {
            current += increment;
            if (current >= target) {
                current = target;
                clearInterval(timer);
            }
            element.textContent = current.toFixed(decimals);
        }, 16);
    }

    getStatusBadgeColor(status) {
        const colors = {
            'ready': 'success',
            'processing': 'warning',
            'error': 'danger',
            'uploading': 'info'
        };
        return colors[status] || 'secondary';
    }

    getStatusText(status) {
        const texts = {
            'ready': '就绪',
            'processing': '处理中',
            'error': '错误',
            'uploading': '上传中'
        };
        return texts[status] || status;
    }

    handleFileSelection(file) {
        // Validate file
        const maxSize = 500 * 1024 * 1024; // 500MB
        const allowedTypes = ['text/csv', 'application/json', 'application/zip'];
        
        if (file.size > maxSize) {
            BVC.Utils.showNotification('文件大小超过500MB限制', 'danger');
            return;
        }
        
        if (!allowedTypes.includes(file.type) && !file.name.match(/\.(csv|json|zip)$/i)) {
            BVC.Utils.showNotification('只支持CSV、JSON和ZIP格式文件', 'danger');
            return;
        }
        
        // Show file preview
        this.showFilePreview(file);
    }

    showFilePreview(file) {
        const preview = this.filePreview;
        preview.innerHTML = `
            <div class="file-info">
                <div class="file-icon">
                    <i class="fas fa-file-${this.getFileIcon(file.type)}"></i>
                </div>
                <div class="file-details">
                    <h6>${file.name}</h6>
                    <p>大小: ${BVC.Utils.formatFileSize(file.size)} | 类型: ${file.type}</p>
                </div>
            </div>
        `;
        preview.style.display = 'block';
        this.uploadButton.disabled = false;
    }

    getFileIcon(mimeType) {
        if (mimeType.includes('csv')) return 'csv';
        if (mimeType.includes('json')) return 'code';
        if (mimeType.includes('zip')) return 'archive';
        return 'alt';
    }

    async startUpload() {
        const file = this.fileInput.files[0];
        if (!file) return;
        
        const formData = this.getUploadFormData(file);
        
        try {
            this.showUploadProgress();
            await this.uploadWithProgress(formData);
            this.hideUploadProgress();
            BVC.Utils.showNotification('数据集上传成功', 'success');
            
            // Close modal and refresh datasets
            bootstrap.Modal.getInstance(document.getElementById('uploadModal')).hide();
            this.loadDatasets();
            
        } catch (error) {
            this.hideUploadProgress();
            BVC.Utils.showNotification(`上传失败: ${error.message}`, 'danger');
        }
    }

    getUploadFormData(file) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('name', document.getElementById('datasetName').value);
        formData.append('version', document.getElementById('datasetVersion').value);
        formData.append('architecture', document.getElementById('datasetArchitecture').value);
        formData.append('optimization', document.getElementById('datasetOptimization').value);
        formData.append('description', document.getElementById('datasetDescription').value);
        
        return formData;
    }

    showUploadProgress() {
        this.uploadProgress.style.display = 'block';
        this.uploadButton.disabled = true;
        document.getElementById('progressText').textContent = '上传中...';
    }

    hideUploadProgress() {
        this.uploadProgress.style.display = 'none';
        this.uploadButton.disabled = false;
        document.getElementById('progressBar').style.width = '0%';
    }

    async uploadWithProgress(formData) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();
            
            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const percentComplete = (e.loaded / e.total) * 100;
                    document.getElementById('progressBar').style.width = percentComplete + '%';
                    document.getElementById('progressText').textContent = `上传中... ${Math.round(percentComplete)}%`;
                }
            });
            
            xhr.addEventListener('load', () => {
                if (xhr.status === 200) {
                    resolve(JSON.parse(xhr.responseText));
                } else {
                    reject(new Error(xhr.statusText));
                }
            });
            
            xhr.addEventListener('error', () => {
                reject(new Error('Upload failed'));
            });
            
            xhr.open('POST', '/api/datasets/upload');
            xhr.send(formData);
        });
    }

    openPreprocessModal() {
        if (!this.selectedDataset) {
            BVC.Utils.showNotification('请先选择一个数据集', 'warning');
            return;
        }
        
        const modal = new bootstrap.Modal(document.getElementById('preprocessModal'));
        modal.show();
    }

    async startPreprocessing() {
        if (!this.selectedDataset) return;
        
        const config = this.getPreprocessingConfig();
        
        try {
            BVC.Utils.showNotification('开始预处理数据集...', 'info');
            
            const response = await BVC.API.post(`/datasets/${this.selectedDataset.id}/preprocess`, config);
            
            BVC.Utils.showNotification('预处理任务已启动', 'success');
            bootstrap.Modal.getInstance(document.getElementById('preprocessModal')).hide();
            
            // Update dataset status
            this.selectedDataset.status = 'processing';
            this.renderDatasets();
            
        } catch (error) {
            BVC.Utils.showNotification(`预处理失败: ${error.message}`, 'danger');
        }
    }

    getPreprocessingConfig() {
        return {
            cleaning: {
                remove_duplicates: document.getElementById('removeDuplicates').checked,
                handle_missing: document.getElementById('handleMissing').checked,
                remove_outliers: document.getElementById('removeOutliers').checked,
                normalize_data: document.getElementById('normalizeData').checked
            },
            split: {
                train_ratio: parseFloat(document.getElementById('trainRatio').value),
                val_ratio: parseFloat(document.getElementById('valRatio').value),
                test_ratio: parseFloat(document.getElementById('testRatio').value)
            },
            features: {
                extract_cfg: document.getElementById('extractCFG').checked,
                extract_dfg: document.getElementById('extractDFG').checked,
                extract_ast: document.getElementById('extractAST').checked,
                extract_semantic: document.getElementById('extractSemantic').checked
            }
        };
    }

    async downloadDataset() {
        if (!this.selectedDataset) return;
        
        try {
            const response = await fetch(`/api/datasets/${this.selectedDataset.id}/download`);
            const blob = await response.blob();
            
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${this.selectedDataset.id}.zip`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            BVC.Utils.showNotification('数据集下载已开始', 'success');
            
        } catch (error) {
            BVC.Utils.showNotification('下载失败', 'danger');
        }
    }

    async deleteDataset() {
        if (!this.selectedDataset) return;
        
        if (!confirm(`确定要删除数据集 "${this.selectedDataset.name}_${this.selectedDataset.version}" 吗？此操作不可撤销。`)) {
            return;
        }
        
        try {
            await BVC.API.delete(`/datasets/${this.selectedDataset.id}`);
            
            BVC.Utils.showNotification('数据集已删除', 'success');
            
            // Remove from local data and refresh
            this.datasets = this.datasets.filter(d => d.id !== this.selectedDataset.id);
            this.filteredDatasets = this.filteredDatasets.filter(d => d.id !== this.selectedDataset.id);
            this.selectedDataset = null;
            
            this.renderDatasets();
            this.updateStatistics();
            this.datasetDetails.style.display = 'none';
            this.dataAnalysis.style.display = 'none';
            
        } catch (error) {
            BVC.Utils.showNotification('删除失败', 'danger');
        }
    }
}

// Initialize dataset manager when page loads
document.addEventListener('DOMContentLoaded', function() {
    window.datasetManager = new DatasetManager();
    console.log('Dataset manager initialized');
});