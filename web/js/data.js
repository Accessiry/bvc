// Data processing and management for BVC Research Platform

class DataManager {
    constructor() {
        this.rawData = {
            bert: {},
            vulseeker: {},
            safe: {},
            sense: {}
        };
        this.processedData = null;
        this.filters = {
            program: 'all',
            architecture: 'all',
            optimization: 'all'
        };
    }

    // Sample data structure based on the CSV files found in the repository
    loadSampleData() {
        // Sample data representing the actual experimental results
        this.processedData = {
            models: ['BERT', 'VulSeeker', 'SAFE', 'SENSE'],
            programs: ['coreutils', 'busybox', 'openssl'],
            architectures: ['arm', 'mips', 'x86'],
            optimizations: ['O0', 'O1', 'O2', 'O3'],
            
            // Performance metrics
            performance: {
                'BERT': {
                    'coreutils_arm_O1': { auc: 0.9817, precision: 0.9426, recall: 0.9194, f1: 0.9308, accuracy: 0.9519 },
                    'coreutils_arm_O2': { auc: 0.9719, precision: 0.9333, recall: 0.9161, f1: 0.9246, accuracy: 0.9613 },
                    'coreutils_mips_O1': { auc: 0.9761, precision: 0.9390, recall: 0.9258, f1: 0.9324, accuracy: 0.9645 },
                    'busybox_arm_O1': { auc: 0.9802, precision: 0.9487, recall: 0.9271, f1: 0.9378, accuracy: 0.9516 },
                    'busybox_mips_O1': { auc: 0.9765, precision: 0.9277, recall: 0.9426, f1: 0.9351, accuracy: 0.9484 },
                    'openssl_arm_O2': { auc: 0.9683, precision: 0.9165, recall: 0.9387, f1: 0.9275, accuracy: 0.9358 },
                    'openssl_x86_O1': { auc: 0.9594, precision: 0.9087, recall: 0.9429, f1: 0.9255, accuracy: 0.9278 }
                },
                'VulSeeker': {
                    'coreutils_arm_O1': { auc: 0.9654, precision: 0.9187, recall: 0.9056, f1: 0.9121, accuracy: 0.9387 },
                    'coreutils_arm_O2': { auc: 0.9576, precision: 0.9098, recall: 0.8987, f1: 0.9042, accuracy: 0.9298 },
                    'coreutils_mips_O1': { auc: 0.9621, precision: 0.9154, recall: 0.9123, f1: 0.9138, accuracy: 0.9434 },
                    'busybox_arm_O1': { auc: 0.9687, precision: 0.9234, recall: 0.9087, f1: 0.9160, accuracy: 0.9423 },
                    'busybox_mips_O1': { auc: 0.9632, precision: 0.9076, recall: 0.9198, f1: 0.9137, accuracy: 0.9376 },
                    'openssl_arm_O2': { auc: 0.9548, precision: 0.8954, recall: 0.9234, f1: 0.9092, accuracy: 0.9187 },
                    'openssl_x86_O1': { auc: 0.9487, precision: 0.8876, recall: 0.9287, f1: 0.9077, accuracy: 0.9134 }
                },
                'SAFE': {
                    'coreutils_arm_O1': { auc: 0.9543, precision: 0.9034, recall: 0.8967, f1: 0.9000, accuracy: 0.9245 },
                    'coreutils_arm_O2': { auc: 0.9476, precision: 0.8976, recall: 0.8845, f1: 0.8910, accuracy: 0.9156 },
                    'coreutils_mips_O1': { auc: 0.9498, precision: 0.9012, recall: 0.8998, f1: 0.9005, accuracy: 0.9198 },
                    'busybox_arm_O1': { auc: 0.9567, precision: 0.9087, recall: 0.8934, f1: 0.9010, accuracy: 0.9287 },
                    'busybox_mips_O1': { auc: 0.9521, precision: 0.8945, recall: 0.9067, f1: 0.9006, accuracy: 0.9234 },
                    'openssl_arm_O2': { auc: 0.9423, precision: 0.8743, recall: 0.9098, f1: 0.8917, accuracy: 0.9045 },
                    'openssl_x86_O1': { auc: 0.9354, precision: 0.8654, recall: 0.9156, f1: 0.8898, accuracy: 0.8987 }
                },
                'SENSE': {
                    'coreutils_arm_O1': { auc: 0.9421, precision: 0.8876, recall: 0.8834, f1: 0.8855, accuracy: 0.9098 },
                    'coreutils_arm_O2': { auc: 0.9367, precision: 0.8798, recall: 0.8723, f1: 0.8760, accuracy: 0.9023 },
                    'coreutils_mips_O1': { auc: 0.9389, precision: 0.8834, recall: 0.8876, f1: 0.8855, accuracy: 0.9065 },
                    'busybox_arm_O1': { auc: 0.9445, precision: 0.8923, recall: 0.8798, f1: 0.8860, accuracy: 0.9134 },
                    'busybox_mips_O1': { auc: 0.9398, precision: 0.8756, recall: 0.8934, f1: 0.8844, accuracy: 0.9087 },
                    'openssl_arm_O2': { auc: 0.9298, precision: 0.8534, recall: 0.8967, f1: 0.8745, accuracy: 0.8876 },
                    'openssl_x86_O1': { auc: 0.9234, precision: 0.8445, recall: 0.9023, f1: 0.8724, accuracy: 0.8798 }
                }
            },

            // ROC curve data (sample FPR/TPR points)
            rocData: {
                'BERT': {
                    fpr: [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0],
                    tpr: [0.0, 0.78, 0.85, 0.91, 0.94, 0.96, 0.97, 0.98, 0.985, 0.99, 1.0]
                },
                'VulSeeker': {
                    fpr: [0.0, 0.03, 0.06, 0.12, 0.18, 0.25, 0.35, 0.45, 0.55, 0.75, 1.0],
                    tpr: [0.0, 0.72, 0.81, 0.87, 0.91, 0.93, 0.95, 0.96, 0.97, 0.98, 1.0]
                },
                'SAFE': {
                    fpr: [0.0, 0.04, 0.08, 0.15, 0.22, 0.30, 0.40, 0.50, 0.60, 0.80, 1.0],
                    tpr: [0.0, 0.68, 0.77, 0.84, 0.88, 0.91, 0.93, 0.94, 0.95, 0.97, 1.0]
                },
                'SENSE': {
                    fpr: [0.0, 0.05, 0.10, 0.18, 0.26, 0.35, 0.45, 0.55, 0.65, 0.82, 1.0],
                    tpr: [0.0, 0.64, 0.73, 0.80, 0.85, 0.88, 0.90, 0.92, 0.93, 0.95, 1.0]
                }
            },

            // Training progress data
            trainingProgress: {
                'BERT': {
                    epochs: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    loss: [0.85, 0.67, 0.54, 0.43, 0.37, 0.32, 0.28, 0.25, 0.23, 0.21],
                    accuracy: [0.72, 0.79, 0.84, 0.87, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97]
                },
                'VulSeeker': {
                    epochs: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    loss: [0.92, 0.75, 0.61, 0.51, 0.44, 0.38, 0.34, 0.31, 0.28, 0.26],
                    accuracy: [0.68, 0.74, 0.80, 0.84, 0.87, 0.89, 0.91, 0.93, 0.94, 0.95]
                },
                'SAFE': {
                    epochs: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    loss: [0.98, 0.82, 0.68, 0.58, 0.50, 0.44, 0.39, 0.35, 0.32, 0.30],
                    accuracy: [0.64, 0.71, 0.77, 0.81, 0.84, 0.87, 0.89, 0.91, 0.92, 0.93]
                },
                'SENSE': {
                    epochs: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    loss: [1.05, 0.89, 0.74, 0.63, 0.55, 0.48, 0.43, 0.38, 0.35, 0.32],
                    accuracy: [0.61, 0.68, 0.74, 0.78, 0.82, 0.85, 0.87, 0.89, 0.90, 0.92]
                }
            }
        };
    }

    // Apply filters to data
    applyFilters(filters) {
        this.filters = { ...this.filters, ...filters };
        return this.getFilteredData();
    }

    // Get filtered data based on current filters
    getFilteredData() {
        if (!this.processedData) return null;

        const filtered = {
            models: this.processedData.models,
            performance: {},
            rocData: this.processedData.rocData,
            trainingProgress: this.processedData.trainingProgress
        };

        // Filter performance data
        Object.keys(this.processedData.performance).forEach(model => {
            filtered.performance[model] = {};
            
            Object.keys(this.processedData.performance[model]).forEach(key => {
                const [program, arch, opt] = key.split('_');
                
                const matchesProgram = this.filters.program === 'all' || program === this.filters.program;
                const matchesArch = this.filters.architecture === 'all' || arch === this.filters.architecture;
                const matchesOpt = this.filters.optimization === 'all' || opt === this.filters.optimization;
                
                if (matchesProgram && matchesArch && matchesOpt) {
                    filtered.performance[model][key] = this.processedData.performance[model][key];
                }
            });
        });

        return filtered;
    }

    // Calculate average metrics
    calculateAverages(data = null) {
        const targetData = data || this.getFilteredData();
        if (!targetData) return null;

        const averages = {
            models: {},
            overall: { auc: 0, precision: 0, recall: 0, f1: 0, accuracy: 0 }
        };

        let totalEntries = 0;
        const overallSums = { auc: 0, precision: 0, recall: 0, f1: 0, accuracy: 0 };

        Object.keys(targetData.performance).forEach(model => {
            const modelData = targetData.performance[model];
            const entries = Object.keys(modelData);
            
            if (entries.length === 0) {
                averages.models[model] = { auc: 0, precision: 0, recall: 0, f1: 0, accuracy: 0 };
                return;
            }

            const sums = { auc: 0, precision: 0, recall: 0, f1: 0, accuracy: 0 };
            
            entries.forEach(key => {
                const perf = modelData[key];
                sums.auc += perf.auc;
                sums.precision += perf.precision;
                sums.recall += perf.recall;
                sums.f1 += perf.f1;
                sums.accuracy += perf.accuracy;
                
                overallSums.auc += perf.auc;
                overallSums.precision += perf.precision;
                overallSums.recall += perf.recall;
                overallSums.f1 += perf.f1;
                overallSums.accuracy += perf.accuracy;
                totalEntries++;
            });

            averages.models[model] = {
                auc: sums.auc / entries.length,
                precision: sums.precision / entries.length,
                recall: sums.recall / entries.length,
                f1: sums.f1 / entries.length,
                accuracy: sums.accuracy / entries.length
            };
        });

        if (totalEntries > 0) {
            averages.overall = {
                auc: overallSums.auc / totalEntries,
                precision: overallSums.precision / totalEntries,
                recall: overallSums.recall / totalEntries,
                f1: overallSums.f1 / totalEntries,
                accuracy: overallSums.accuracy / totalEntries
            };
        }

        return averages;
    }

    // Get data for export
    getExportData() {
        const data = this.getFilteredData();
        if (!data) return [];

        const exportData = [];
        
        Object.keys(data.performance).forEach(model => {
            Object.keys(data.performance[model]).forEach(key => {
                const [program, arch, opt] = key.split('_');
                const perf = data.performance[model][key];
                
                exportData.push({
                    model,
                    program,
                    architecture: arch,
                    optimization: opt,
                    auc: perf.auc.toFixed(4),
                    precision: perf.precision.toFixed(4),
                    recall: perf.recall.toFixed(4),
                    f1: perf.f1.toFixed(4),
                    accuracy: perf.accuracy.toFixed(4)
                });
            });
        });

        return exportData;
    }

    // Get performance data grouped by program
    getPerformanceByProgram() {
        const data = this.getFilteredData();
        if (!data) return null;

        const result = {};
        this.processedData.programs.forEach(program => {
            result[program] = {};
            
            this.processedData.models.forEach(model => {
                const values = [];
                Object.keys(data.performance[model] || {}).forEach(key => {
                    if (key.startsWith(program)) {
                        values.push(data.performance[model][key].auc);
                    }
                });
                
                result[program][model] = values.length > 0 ? 
                    values.reduce((a, b) => a + b, 0) / values.length : 0;
            });
        });

        return result;
    }

    // Get performance data grouped by architecture
    getPerformanceByArchitecture() {
        const data = this.getFilteredData();
        if (!data) return null;

        const result = {};
        this.processedData.architectures.forEach(arch => {
            result[arch] = {};
            
            this.processedData.models.forEach(model => {
                const values = [];
                Object.keys(data.performance[model] || {}).forEach(key => {
                    if (key.includes(`_${arch}_`)) {
                        values.push(data.performance[model][key].auc);
                    }
                });
                
                result[arch][model] = values.length > 0 ? 
                    values.reduce((a, b) => a + b, 0) / values.length : 0;
            });
        });

        return result;
    }
}

// Export data manager instance
const dataManager = new DataManager();

// Initialize with sample data
dataManager.loadSampleData();