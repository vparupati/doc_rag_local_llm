// JavaScript for the RAG application

class RAGApp {
    constructor() {
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.checkSystemStatus();
    }

    setupEventListeners() {
        // Upload form
        document.getElementById('uploadForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleFileUpload();
        });

        // Query form
        document.getElementById('queryForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleQuery();
        });
    }

    async checkSystemStatus() {
        try {
            const response = await fetch('/health');
            const data = await response.json();
            
            const statusElement = document.getElementById('systemStatus');
            
            if (data.status === 'healthy') {
                statusElement.innerHTML = `
                    <span class="status-healthy">
                        <i class="fas fa-check-circle me-1"></i>
                        System is healthy
                    </span>
                    <small class="text-muted ms-2">
                        Vector Store: ${data.vector_store.total_documents || 0} documents
                    </small>
                `;
            } else {
                statusElement.innerHTML = `
                    <span class="status-unhealthy">
                        <i class="fas fa-exclamation-triangle me-1"></i>
                        System is unhealthy: ${data.error || 'Unknown error'}
                    </span>
                `;
            }
        } catch (error) {
            document.getElementById('systemStatus').innerHTML = `
                <span class="status-unhealthy">
                    <i class="fas fa-times-circle me-1"></i>
                    Failed to check system status
                </span>
            `;
        }
    }

    async handleFileUpload() {
        const fileInput = document.getElementById('files');
        const files = fileInput.files;
        
        if (files.length === 0) {
            this.showAlert('Please select at least one file to upload.', 'warning');
            return;
        }

        const formData = new FormData();
        for (let file of files) {
            formData.append('files', file);
        }

        const uploadBtn = document.getElementById('uploadBtn');
        const progressDiv = document.getElementById('uploadProgress');
        const resultDiv = document.getElementById('uploadResult');

        // Show loading state
        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Uploading...';
        progressDiv.style.display = 'block';
        resultDiv.innerHTML = '';

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                this.showUploadResult(data);
                this.checkSystemStatus(); // Refresh status
            } else {
                this.showAlert(`Upload failed: ${data.detail}`, 'danger');
            }
        } catch (error) {
            this.showAlert(`Upload error: ${error.message}`, 'danger');
        } finally {
            // Reset UI
            uploadBtn.disabled = false;
            uploadBtn.innerHTML = '<i class="fas fa-upload me-1"></i>Upload Files';
            progressDiv.style.display = 'none';
            fileInput.value = '';
        }
    }

    showUploadResult(data) {
        const resultDiv = document.getElementById('uploadResult');
        
        let html = `
            <div class="alert alert-success fade-in">
                <h6><i class="fas fa-check-circle me-1"></i>Upload Successful</h6>
                <p class="mb-1">Files processed: ${data.files_processed}</p>
                <p class="mb-1">Total chunks created: ${data.total_chunks}</p>
            </div>
        `;

        if (data.file_details && data.file_details.length > 0) {
            html += '<div class="mt-3"><h6>File Details:</h6><ul class="list-group">';
            
            data.file_details.forEach(file => {
                const statusClass = file.status === 'success' ? 'success' : 'danger';
                const statusIcon = file.status === 'success' ? 'check-circle' : 'times-circle';
                
                html += `
                    <li class="list-group-item d-flex justify-content-between align-items-center">
                        <span>
                            <i class="fas fa-${statusIcon} text-${statusClass} me-2"></i>
                            ${file.filename}
                        </span>
                        <span class="badge bg-${statusClass}">${file.chunks} chunks</span>
                    </li>
                `;
            });
            
            html += '</ul></div>';
        }

        resultDiv.innerHTML = html;
    }

    async handleQuery() {
        const questionInput = document.getElementById('question');
        const question = questionInput.value.trim();
        
        if (!question) {
            this.showAlert('Please enter a question.', 'warning');
            return;
        }

        const queryBtn = document.getElementById('queryBtn');
        const resultDiv = document.getElementById('queryResult');

        // Show loading state
        queryBtn.disabled = true;
        queryBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Processing...';
        resultDiv.innerHTML = '<div class="text-center"><div class="spinner-border" role="status"></div><p class="mt-2">Processing your question...</p></div>';

        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ question: question })
            });

            const data = await response.json();

            if (response.ok) {
                this.showQueryResult(data);
            } else {
                this.showAlert(`Query failed: ${data.detail}`, 'danger');
            }
        } catch (error) {
            this.showAlert(`Query error: ${error.message}`, 'danger');
        } finally {
            // Reset UI
            queryBtn.disabled = false;
            queryBtn.innerHTML = '<i class="fas fa-search me-1"></i>Ask Question';
        }
    }

    showQueryResult(data) {
        const resultDiv = document.getElementById('queryResult');
        
        let html = `
            <div class="query-response fade-in">
                <div class="answer">${this.escapeHtml(data.answer)}</div>
                <div class="sources">
                    <i class="fas fa-book me-1"></i>
                    Based on ${data.num_sources} source(s)
                </div>
            </div>
        `;

        if (data.source_documents && data.source_documents.length > 0) {
            html += '<div class="mt-3"><h6>Source Documents:</h6>';
            
            data.source_documents.forEach((doc, index) => {
                html += `
                    <div class="source-document fade-in">
                        <div class="metadata">
                            <strong>Source ${index + 1}:</strong> ${doc.metadata.filename || 'Unknown'}
                            <span class="ms-2 badge bg-secondary">Chunk ${doc.metadata.chunk_id || 'N/A'}</span>
                        </div>
                        <div class="content">${this.escapeHtml(doc.content)}</div>
                    </div>
                `;
            });
            
            html += '</div>';
        }

        resultDiv.innerHTML = html;
    }

    async showStats() {
        const modal = new bootstrap.Modal(document.getElementById('statsModal'));
        const content = document.getElementById('statsContent');
        
        content.innerHTML = '<div class="text-center"><div class="spinner-border" role="status"></div><p class="mt-2">Loading statistics...</p></div>';
        modal.show();

        try {
            const response = await fetch('/stats');
            const data = await response.json();

            let html = `
                <div class="row">
                    <div class="col-md-6">
                        <h6>Vector Store</h6>
                        <ul class="list-group list-group-flush">
                            <li class="list-group-item d-flex justify-content-between">
                                <span>Total Documents:</span>
                                <span class="badge bg-primary">${data.total_documents || 0}</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between">
                                <span>Collection:</span>
                                <span>${data.collection_name || 'N/A'}</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between">
                                <span>Embedding Model:</span>
                                <span class="text-truncate" style="max-width: 200px;">${data.embedding_model || 'N/A'}</span>
                            </li>
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <h6>System Info</h6>
                        <ul class="list-group list-group-flush">
                            <li class="list-group-item d-flex justify-content-between">
                                <span>Model Type:</span>
                                <span>${data.model_type || 'N/A'}</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between">
                                <span>Model Name:</span>
                                <span class="text-truncate" style="max-width: 200px;">${data.model_name || 'N/A'}</span>
                            </li>
                        </ul>
                    </div>
                </div>
            `;

            content.innerHTML = html;
        } catch (error) {
            content.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle me-1"></i>
                    Failed to load statistics: ${error.message}
                </div>
            `;
        }
    }

    async resetVectorStore() {
        if (!confirm('Are you sure you want to reset the vector store? This will delete all uploaded documents.')) {
            return;
        }

        try {
            const response = await fetch('/reset', { method: 'DELETE' });
            const data = await response.json();

            if (response.ok) {
                this.showAlert('Vector store reset successfully.', 'success');
                this.checkSystemStatus();
            } else {
                this.showAlert(`Reset failed: ${data.detail}`, 'danger');
            }
        } catch (error) {
            this.showAlert(`Reset error: ${error.message}`, 'danger');
        }
    }

    showAlert(message, type) {
        const resultDiv = document.getElementById('uploadResult');
        resultDiv.innerHTML = `
            <div class="alert alert-${type} fade-in">
                <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'warning' ? 'exclamation-triangle' : 'times-circle'} me-1"></i>
                ${message}
            </div>
        `;
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Global functions for navbar links
function showStats() {
    app.showStats();
}

function resetVectorStore() {
    app.resetVectorStore();
}

// Initialize the app when DOM is loaded
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new RAGApp();
});
