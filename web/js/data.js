// Data processing and analysis utilities
class DataProcessor {
    constructor() {
        this.processors = new Map();
        this.registerDefaultProcessors();
    }

    registerDefaultProcessors() {
        this.processors.set('csv', new CSVProcessor());
        this.processors.set('json', new JSONProcessor());
        this.processors.set('tfrecord', new TFRecordProcessor());
    }

    getProcessor(type) {
        return this.processors.get(type);
    }

    async processFile(file, type) {
        const processor = this.getProcessor(type);
        if (!processor) {
            throw new Error(`Unsupported file type: ${type}`);
        }

        return await processor.process(file);
    }

    async analyzeDataset(data) {
        const analyzer = new DatasetAnalyzer();
        return await analyzer.analyze(data);
    }
}

// CSV file processor
class CSVProcessor {
    async process(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const csv = e.target.result;
                    const data = this.parseCSV(csv);
                    resolve(data);
                } catch (error) {
                    reject(error);
                }
            };
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }

    parseCSV(csv) {
        const lines = csv.split('\n').filter(line => line.trim());
        if (lines.length === 0) {
            throw new Error('Empty CSV file');
        }

        const headers = this.parseCSVLine(lines[0]);
        const rows = lines.slice(1).map(line => this.parseCSVLine(line));

        return {
            headers,
            rows,
            rowCount: rows.length,
            columnCount: headers.length
        };
    }

    parseCSVLine(line) {
        const result = [];
        let current = '';
        let inQuotes = false;

        for (let i = 0; i < line.length; i++) {
            const char = line[i];
            const nextChar = line[i + 1];

            if (char === '"') {
                if (inQuotes && nextChar === '"') {
                    current += '"';
                    i++; // Skip next quote
                } else {
                    inQuotes = !inQuotes;
                }
            } else if (char === ',' && !inQuotes) {
                result.push(current);
                current = '';
            } else {
                current += char;
            }
        }

        result.push(current);
        return result;
    }
}

// JSON file processor
class JSONProcessor {
    async process(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                try {
                    const jsonData = JSON.parse(e.target.result);
                    const data = this.normalizeJSON(jsonData);
                    resolve(data);
                } catch (error) {
                    reject(new Error('Invalid JSON format'));
                }
            };
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsText(file);
        });
    }

    normalizeJSON(jsonData) {
        if (Array.isArray(jsonData)) {
            return this.processJSONArray(jsonData);
        } else if (typeof jsonData === 'object') {
            return this.processJSONObject(jsonData);
        } else {
            throw new Error('Unsupported JSON structure');
        }
    }

    processJSONArray(array) {
        if (array.length === 0) {
            throw new Error('Empty JSON array');
        }

        const firstItem = array[0];
        if (typeof firstItem === 'object') {
            const headers = Object.keys(firstItem);
            const rows = array.map(item => headers.map(header => item[header] || ''));

            return {
                headers,
                rows,
                rowCount: rows.length,
                columnCount: headers.length
            };
        } else {
            return {
                headers: ['value'],
                rows: array.map(item => [item]),
                rowCount: array.length,
                columnCount: 1
            };
        }
    }

    processJSONObject(obj) {
        const headers = Object.keys(obj);
        const rows = [headers.map(header => obj[header])];

        return {
            headers,
            rows,
            rowCount: 1,
            columnCount: headers.length
        };
    }
}

// TFRecord file processor (placeholder)
class TFRecordProcessor {
    async process(file) {
        // TFRecord processing would require specific libraries
        // For now, return mock data
        return {
            headers: ['feature_vector', 'label'],
            rows: [],
            rowCount: 0,
            columnCount: 2,
            note: 'TFRecord processing requires backend support'
        };
    }
}

// Dataset analyzer
class DatasetAnalyzer {
    async analyze(data) {
        const analysis = {
            basic: this.analyzeBasicStatistics(data),
            quality: this.analyzeDataQuality(data),
            distribution: this.analyzeDistribution(data),
            correlations: this.analyzeCorrelations(data),
            features: this.analyzeFeatures(data)
        };

        return analysis;
    }

    analyzeBasicStatistics(data) {
        const { headers, rows } = data;
        
        const stats = {
            totalRows: rows.length,
            totalColumns: headers.length,
            memoryUsage: this.estimateMemoryUsage(data),
            dataTypes: this.inferDataTypes(data)
        };

        return stats;
    }

    analyzeDataQuality(data) {
        const { headers, rows } = data;
        
        const quality = {
            completeness: this.calculateCompleteness(data),
            uniqueness: this.calculateUniqueness(data),
            validity: this.calculateValidity(data),
            consistency: this.calculateConsistency(data)
        };

        return quality;
    }

    analyzeDistribution(data) {
        const { headers, rows } = data;
        const distributions = {};

        headers.forEach((header, index) => {
            const column = rows.map(row => row[index]);
            distributions[header] = this.calculateColumnDistribution(column);
        });

        return distributions;
    }

    analyzeCorrelations(data) {
        const { headers, rows } = data;
        const numericColumns = this.getNumericColumns(data);
        
        if (numericColumns.length < 2) {
            return { message: 'Insufficient numeric columns for correlation analysis' };
        }

        const correlations = {};
        for (let i = 0; i < numericColumns.length; i++) {
            for (let j = i + 1; j < numericColumns.length; j++) {
                const col1 = numericColumns[i];
                const col2 = numericColumns[j];
                const correlation = this.calculateCorrelation(
                    rows.map(row => parseFloat(row[col1.index]) || 0),
                    rows.map(row => parseFloat(row[col2.index]) || 0)
                );
                correlations[`${col1.name}_${col2.name}`] = correlation;
            }
        }

        return correlations;
    }

    analyzeFeatures(data) {
        const { headers, rows } = data;
        
        const features = headers.map((header, index) => {
            const column = rows.map(row => row[index]);
            return {
                name: header,
                type: this.inferColumnType(column),
                uniqueValues: new Set(column).size,
                missingValues: column.filter(val => !val || val.trim() === '').length,
                statistics: this.calculateColumnStatistics(column)
            };
        });

        return features;
    }

    calculateCompleteness(data) {
        const { rows } = data;
        let totalCells = 0;
        let filledCells = 0;

        rows.forEach(row => {
            row.forEach(cell => {
                totalCells++;
                if (cell && cell.toString().trim() !== '') {
                    filledCells++;
                }
            });
        });

        return totalCells > 0 ? filledCells / totalCells : 0;
    }

    calculateUniqueness(data) {
        const { rows } = data;
        const rowStrings = rows.map(row => row.join('|'));
        const uniqueRows = new Set(rowStrings);
        
        return rows.length > 0 ? uniqueRows.size / rows.length : 0;
    }

    calculateValidity(data) {
        // Simplified validity check - could be enhanced with specific validation rules
        const { headers, rows } = data;
        let validCells = 0;
        let totalCells = 0;

        rows.forEach(row => {
            row.forEach(cell => {
                totalCells++;
                // Basic validity: not null/undefined and not just whitespace
                if (cell !== null && cell !== undefined && cell.toString().trim() !== '') {
                    validCells++;
                }
            });
        });

        return totalCells > 0 ? validCells / totalCells : 0;
    }

    calculateConsistency(data) {
        const { headers, rows } = data;
        let consistentColumns = 0;

        headers.forEach((header, index) => {
            const column = rows.map(row => row[index]);
            const type = this.inferColumnType(column);
            
            // Check if all non-empty values in column match the inferred type
            const consistent = column.every(value => {
                if (!value || value.toString().trim() === '') return true;
                return this.valueMatchesType(value, type);
            });

            if (consistent) {
                consistentColumns++;
            }
        });

        return headers.length > 0 ? consistentColumns / headers.length : 0;
    }

    calculateColumnDistribution(column) {
        const distribution = {};
        const cleanColumn = column.filter(val => val && val.toString().trim() !== '');

        cleanColumn.forEach(value => {
            distribution[value] = (distribution[value] || 0) + 1;
        });

        const total = cleanColumn.length;
        const percentages = {};
        Object.keys(distribution).forEach(key => {
            percentages[key] = (distribution[key] / total) * 100;
        });

        return {
            counts: distribution,
            percentages: percentages,
            uniqueValues: Object.keys(distribution).length,
            totalValues: total
        };
    }

    getNumericColumns(data) {
        const { headers, rows } = data;
        const numericColumns = [];

        headers.forEach((header, index) => {
            const column = rows.map(row => row[index]);
            if (this.inferColumnType(column) === 'numeric') {
                numericColumns.push({ name: header, index });
            }
        });

        return numericColumns;
    }

    calculateCorrelation(x, y) {
        if (x.length !== y.length || x.length === 0) return 0;

        const n = x.length;
        const sumX = x.reduce((a, b) => a + b, 0);
        const sumY = y.reduce((a, b) => a + b, 0);
        const sumXY = x.reduce((acc, xi, i) => acc + xi * y[i], 0);
        const sumXX = x.reduce((acc, xi) => acc + xi * xi, 0);
        const sumYY = y.reduce((acc, yi) => acc + yi * yi, 0);

        const numerator = n * sumXY - sumX * sumY;
        const denominator = Math.sqrt((n * sumXX - sumX * sumX) * (n * sumYY - sumY * sumY));

        return denominator === 0 ? 0 : numerator / denominator;
    }

    inferDataTypes(data) {
        const { headers, rows } = data;
        const types = {};

        headers.forEach((header, index) => {
            const column = rows.map(row => row[index]);
            types[header] = this.inferColumnType(column);
        });

        return types;
    }

    inferColumnType(column) {
        const nonEmptyValues = column.filter(val => val && val.toString().trim() !== '');
        
        if (nonEmptyValues.length === 0) return 'unknown';

        let numericCount = 0;
        let dateCount = 0;
        let booleanCount = 0;

        nonEmptyValues.forEach(value => {
            const strValue = value.toString().trim();
            
            if (this.isNumeric(strValue)) {
                numericCount++;
            } else if (this.isDate(strValue)) {
                dateCount++;
            } else if (this.isBoolean(strValue)) {
                booleanCount++;
            }
        });

        const total = nonEmptyValues.length;
        const threshold = 0.8; // 80% threshold for type inference

        if (numericCount / total >= threshold) return 'numeric';
        if (dateCount / total >= threshold) return 'date';
        if (booleanCount / total >= threshold) return 'boolean';
        
        return 'text';
    }

    isNumeric(value) {
        return !isNaN(parseFloat(value)) && isFinite(value);
    }

    isDate(value) {
        const date = new Date(value);
        return !isNaN(date.getTime()) && value.length > 5; // Rough date validation
    }

    isBoolean(value) {
        const lowerValue = value.toLowerCase();
        return ['true', 'false', '1', '0', 'yes', 'no', 'y', 'n'].includes(lowerValue);
    }

    valueMatchesType(value, type) {
        const strValue = value.toString().trim();
        
        switch (type) {
            case 'numeric':
                return this.isNumeric(strValue);
            case 'date':
                return this.isDate(strValue);
            case 'boolean':
                return this.isBoolean(strValue);
            case 'text':
                return true; // Text can be anything
            default:
                return true;
        }
    }

    calculateColumnStatistics(column) {
        const cleanColumn = column.filter(val => val && val.toString().trim() !== '');
        
        if (cleanColumn.length === 0) {
            return { isEmpty: true };
        }

        const type = this.inferColumnType(column);
        
        if (type === 'numeric') {
            return this.calculateNumericStatistics(cleanColumn);
        } else {
            return this.calculateCategoricalStatistics(cleanColumn);
        }
    }

    calculateNumericStatistics(column) {
        const numbers = column.map(val => parseFloat(val)).filter(num => !isNaN(num));
        
        if (numbers.length === 0) return { isEmpty: true };

        const sorted = [...numbers].sort((a, b) => a - b);
        const sum = numbers.reduce((a, b) => a + b, 0);
        const mean = sum / numbers.length;
        
        const variance = numbers.reduce((acc, num) => acc + Math.pow(num - mean, 2), 0) / numbers.length;
        const stdDev = Math.sqrt(variance);

        return {
            min: sorted[0],
            max: sorted[sorted.length - 1],
            mean: mean,
            median: this.calculateMedian(sorted),
            mode: this.calculateMode(numbers),
            standardDeviation: stdDev,
            variance: variance,
            quartiles: this.calculateQuartiles(sorted)
        };
    }

    calculateCategoricalStatistics(column) {
        const frequency = {};
        column.forEach(value => {
            frequency[value] = (frequency[value] || 0) + 1;
        });

        const sortedByFreq = Object.entries(frequency).sort((a, b) => b[1] - a[1]);
        const mostCommon = sortedByFreq[0];
        const leastCommon = sortedByFreq[sortedByFreq.length - 1];

        return {
            uniqueValues: Object.keys(frequency).length,
            mostCommon: { value: mostCommon[0], count: mostCommon[1] },
            leastCommon: { value: leastCommon[0], count: leastCommon[1] },
            frequency: frequency
        };
    }

    calculateMedian(sortedNumbers) {
        const length = sortedNumbers.length;
        if (length % 2 === 0) {
            return (sortedNumbers[length / 2 - 1] + sortedNumbers[length / 2]) / 2;
        } else {
            return sortedNumbers[Math.floor(length / 2)];
        }
    }

    calculateMode(numbers) {
        const frequency = {};
        numbers.forEach(num => {
            frequency[num] = (frequency[num] || 0) + 1;
        });

        const maxFreq = Math.max(...Object.values(frequency));
        const modes = Object.keys(frequency).filter(key => frequency[key] === maxFreq);
        
        return modes.length === numbers.length ? null : modes.map(Number);
    }

    calculateQuartiles(sortedNumbers) {
        const length = sortedNumbers.length;
        
        const q1Index = Math.floor(length * 0.25);
        const q3Index = Math.floor(length * 0.75);
        
        return {
            q1: sortedNumbers[q1Index],
            q2: this.calculateMedian(sortedNumbers), // q2 is median
            q3: sortedNumbers[q3Index]
        };
    }

    estimateMemoryUsage(data) {
        const { headers, rows } = data;
        
        // Rough estimation: each character ≈ 1 byte
        let totalSize = 0;
        
        headers.forEach(header => {
            totalSize += header.length;
        });
        
        rows.forEach(row => {
            row.forEach(cell => {
                totalSize += cell ? cell.toString().length : 0;
            });
        });
        
        return totalSize; // bytes
    }
}

// Data validation utilities
class DataValidator {
    static validateDatasetStructure(data) {
        const errors = [];
        
        if (!data.headers || !Array.isArray(data.headers)) {
            errors.push('Invalid headers structure');
        }
        
        if (!data.rows || !Array.isArray(data.rows)) {
            errors.push('Invalid rows structure');
        }
        
        if (data.headers && data.rows) {
            // Check if all rows have the same number of columns
            const expectedColumns = data.headers.length;
            data.rows.forEach((row, index) => {
                if (row.length !== expectedColumns) {
                    errors.push(`Row ${index + 1} has ${row.length} columns, expected ${expectedColumns}`);
                }
            });
        }
        
        return {
            isValid: errors.length === 0,
            errors: errors
        };
    }
    
    static validateDataQuality(data, requirements = {}) {
        const warnings = [];
        const errors = [];
        
        const analysis = new DatasetAnalyzer().analyzeDataQuality(data);
        
        // Check completeness
        if (requirements.minCompleteness && analysis.completeness < requirements.minCompleteness) {
            errors.push(`Data completeness (${(analysis.completeness * 100).toFixed(1)}%) below required minimum (${(requirements.minCompleteness * 100).toFixed(1)}%)`);
        }
        
        // Check uniqueness
        if (requirements.minUniqueness && analysis.uniqueness < requirements.minUniqueness) {
            warnings.push(`Data uniqueness (${(analysis.uniqueness * 100).toFixed(1)}%) below recommended minimum (${(requirements.minUniqueness * 100).toFixed(1)}%)`);
        }
        
        return {
            isValid: errors.length === 0,
            errors: errors,
            warnings: warnings,
            analysis: analysis
        };
    }
}

// Export classes for use in other modules
window.DataProcessor = DataProcessor;
window.DatasetAnalyzer = DatasetAnalyzer;
window.DataValidator = DataValidator;