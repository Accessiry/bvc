// ===== Mock Data for Development =====

/**
 * Generate mock performance data based on existing CSV structure
 */
function generateModelPerformanceData() {
    const models = ['vulseeker', 'sense', 'fit', 'safe'];
    const architectures = ['arm', 'mips', 'x86', 'arm-mips', 'arm-x86', 'mips-x86', 'arm-mips-x86'];
    const optimizations = ['O0', 'O1', 'O2', 'O3', 'O0-O1', 'O1-O2', 'O2-O3', 'O0-O1-O2-O3'];
    const programs = ['coreutils', 'openssl', 'busybox', 'libsqlite3', 'libgmp'];
    
    const data = [];
    
    models.forEach(model => {
        architectures.forEach(arch => {
            optimizations.forEach(opt => {
                programs.forEach(program => {
                    // Generate realistic performance metrics
                    const baseAccuracy = 0.85 + Math.random() * 0.1;
                    const precision = Math.max(0.7, Math.min(1.0, baseAccuracy + (Math.random() - 0.5) * 0.1));
                    const recall = Math.max(0.7, Math.min(1.0, baseAccuracy + (Math.random() - 0.5) * 0.1));
                    const f1Score = 2 * (precision * recall) / (precision + recall);
                    const auc = Math.max(0.8, Math.min(1.0, baseAccuracy + Math.random() * 0.1));
                    
                    data.push({
                        model,
                        architecture: arch,
                        optimization: opt,
                        program,
                        precision,
                        recall,
                        f1Score,
                        accuracy: baseAccuracy,
                        auc,
                        timestamp: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString()
                    });
                });
            });
        });
    });
    
    return data;
}

/**
 * Generate mock ROC curve data
 */
function generateROCCurveData(model, points = 100) {
    const data = [];
    
    for (let i = 0; i <= points; i++) {
        const fpr = i / points;
        
        // Different models have different characteristic curves
        let tpr;
        switch (model.toLowerCase()) {
            case 'vulseeker':
                tpr = Math.min(1, Math.pow(fpr, 0.3) + Math.random() * 0.05);
                break;
            case 'sense':
                tpr = Math.min(1, Math.pow(fpr, 0.4) + Math.random() * 0.05);
                break;
            case 'fit':
                tpr = Math.min(1, Math.pow(fpr, 0.5) + Math.random() * 0.05);
                break;
            case 'safe':
                tpr = Math.min(1, Math.pow(fpr, 0.35) + Math.random() * 0.05);
                break;
            default:
                tpr = Math.min(1, fpr + Math.random() * 0.2);
        }
        
        data.push({ fpr, tpr });
    }
    
    return data.sort((a, b) => a.fpr - b.fpr);
}

/**
 * Generate mock detection results
 */
function generateDetectionResults(count = 1000) {
    const models = ['vulseeker', 'sense', 'fit', 'safe'];
    const architectures = ['arm', 'mips', 'x86', 'arm-mips'];
    const optimizations = ['O0', 'O1', 'O2', 'O3'];
    const programs = ['coreutils', 'openssl', 'busybox', 'libsqlite3', 'libgmp'];
    const vulnerabilityTypes = [
        'buffer_overflow', 'use_after_free', 'null_pointer_dereference', 
        'integer_overflow', 'format_string', 'double_free', 'stack_overflow'
    ];
    
    const results = [];
    
    for (let i = 0; i < count; i++) {
        const model = models[Math.floor(Math.random() * models.length)];
        const architecture = architectures[Math.floor(Math.random() * architectures.length)];
        const optimization = optimizations[Math.floor(Math.random() * optimizations.length)];
        const program = programs[Math.floor(Math.random() * programs.length)];
        const vulnerabilityType = vulnerabilityTypes[Math.floor(Math.random() * vulnerabilityTypes.length)];
        
        const confidence = Math.random();
        const isVulnerable = confidence > 0.5; // Higher confidence more likely to be vulnerable
        
        results.push({
            id: i + 1,
            model,
            architecture,
            optimization,
            program,
            functionName: `func_${Math.floor(Math.random() * 10000)}`,
            confidence,
            isVulnerable,
            vulnerabilityType: isVulnerable ? vulnerabilityType : null,
            detectedAt: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString(),
            processingTime: Math.random() * 5000 + 100, // ms
            fileSize: Math.floor(Math.random() * 1000000) + 1000 // bytes
        });
    }
    
    return results;
}

/**
 * Generate mock vulnerability distribution data
 */
function generateVulnerabilityDistribution() {
    const vulnerabilityTypes = [
        { name: 'Buffer Overflow', count: 245, severity: 'high' },
        { name: 'Use After Free', count: 186, severity: 'high' },
        { name: 'Null Pointer Dereference', count: 158, severity: 'medium' },
        { name: 'Integer Overflow', count: 124, severity: 'medium' },
        { name: 'Format String', count: 89, severity: 'high' },
        { name: 'Double Free', count: 67, severity: 'medium' },
        { name: 'Stack Overflow', count: 45, severity: 'high' },
        { name: 'Memory Leak', count: 234, severity: 'low' }
    ];
    
    return vulnerabilityTypes;
}

/**
 * Generate mock confidence heatmap data
 */
function generateConfidenceHeatmap() {
    const models = ['VulSeeker', 'SENSE', 'FIT', 'SAFE'];
    const programs = ['coreutils', 'openssl', 'busybox', 'libsqlite3', 'libgmp'];
    
    const matrix = models.map(model => 
        programs.map(program => Math.random())
    );
    
    return {
        matrix,
        xLabels: programs,
        yLabels: models,
        xTitle: 'Programs',
        yTitle: 'Models'
    };
}

/**
 * Generate mock system statistics
 */
function generateSystemStatistics() {
    return {
        totalDetections: 12547 + Math.floor(Math.random() * 100),
        vulnerabilitiesFound: 1248 + Math.floor(Math.random() * 20),
        falsePositives: 89 + Math.floor(Math.random() * 10),
        falseNegatives: 23 + Math.floor(Math.random() * 5),
        modelsActive: 4,
        architecturesSupported: 7,
        avgAccuracy: 0.955 + Math.random() * 0.01,
        avgPrecision: 0.932 + Math.random() * 0.015,
        avgRecall: 0.898 + Math.random() * 0.02,
        avgF1Score: 0.914 + Math.random() * 0.01,
        systemUptime: 2457600 + Math.floor(Math.random() * 86400), // seconds
        processingQueue: Math.floor(Math.random() * 50),
        lastScanTime: new Date(Date.now() - Math.random() * 60 * 60 * 1000).toISOString(),
        cpuUsage: 0.3 + Math.random() * 0.4,
        memoryUsage: 0.4 + Math.random() * 0.3,
        diskUsage: 0.2 + Math.random() * 0.2
    };
}

/**
 * Generate mock time series data
 */
function generateTimeSeriesData(metric, days = 30) {
    const data = [];
    const now = new Date();
    
    for (let i = days; i >= 0; i--) {
        const timestamp = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
        let value;
        
        switch (metric) {
            case 'accuracy':
                value = 0.92 + Math.sin(i * 0.1) * 0.02 + Math.random() * 0.01;
                break;
            case 'throughput':
                value = 150 + Math.sin(i * 0.2) * 20 + Math.random() * 10;
                break;
            case 'detections':
                value = Math.floor(100 + Math.sin(i * 0.15) * 30 + Math.random() * 20);
                break;
            default:
                value = Math.random();
        }
        
        data.push({
            timestamp: timestamp.toISOString(),
            value
        });
    }
    
    return data;
}

/**
 * Generate mock analysis history
 */
function generateAnalysisHistory(count = 50) {
    const models = ['vulseeker', 'sense', 'fit', 'safe'];
    const statuses = ['completed', 'processing', 'failed', 'queued'];
    const history = [];
    
    for (let i = 0; i < count; i++) {
        const status = statuses[Math.floor(Math.random() * statuses.length)];
        const model = models[Math.floor(Math.random() * models.length)];
        
        history.push({
            id: `task_${i + 1}`,
            fileName: `binary_${i + 1}.exe`,
            model,
            status,
            confidence: status === 'completed' ? Math.random() : null,
            isVulnerable: status === 'completed' ? Math.random() > 0.7 : null,
            submittedAt: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString(),
            completedAt: status === 'completed' ? 
                new Date(Date.now() - Math.random() * 6 * 24 * 60 * 60 * 1000).toISOString() : null,
            processingTime: status === 'completed' ? Math.random() * 300000 + 5000 : null // ms
        });
    }
    
    return history.sort((a, b) => new Date(b.submittedAt) - new Date(a.submittedAt));
}

/**
 * Generate CSV data that matches the project's existing format
 */
function generateCSVCompatibleData(model, metric) {
    const iterations = 10; // Typical training iterations
    const data = [];
    
    let baseValue;
    switch (metric) {
        case 'accuracy':
            baseValue = 0.85;
            break;
        case 'precision':
            baseValue = 0.88;
            break;
        case 'recall':
            baseValue = 0.83;
            break;
        case 'f1score':
            baseValue = 0.85;
            break;
        case 'AUC':
            baseValue = 0.92;
            break;
        default:
            baseValue = 0.5;
    }
    
    for (let i = 0; i < iterations; i++) {
        // Simulate improvement over iterations with some noise
        const improvement = i * 0.01;
        const noise = (Math.random() - 0.5) * 0.02;
        const value = Math.min(1.0, Math.max(0.0, baseValue + improvement + noise));
        data.push(value);
    }
    
    return data;
}

/**
 * Export mock data generation functions
 */
window.MockData = {
    generateModelPerformanceData,
    generateROCCurveData,
    generateDetectionResults,
    generateVulnerabilityDistribution,
    generateConfidenceHeatmap,
    generateSystemStatistics,
    generateTimeSeriesData,
    generateAnalysisHistory,
    generateCSVCompatibleData
};