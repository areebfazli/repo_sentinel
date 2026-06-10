document.addEventListener('DOMContentLoaded', () => {
    const codeInput = document.getElementById('codeInput');
    const scanBtn = document.getElementById('scanBtn');
    const btnText = scanBtn.querySelector('.btn-text');
    const loader = scanBtn.querySelector('.loader');
    
    const welcomeState = document.getElementById('welcomeState');
    const resultsState = document.getElementById('resultsState');

    // Setup marked.js options for security and styling
    marked.setOptions({
        breaks: true,
        gfm: true
    });

    scanBtn.addEventListener('click', async () => {
        const code = codeInput.value.trim();
        
        if (!code) {
            // Shake animation for empty input
            codeInput.style.animation = 'shake 0.5s ease';
            setTimeout(() => codeInput.style.animation = '', 500);
            return;
        }

        // UI Loading State
        btnText.classList.add('hidden');
        loader.classList.remove('hidden');
        scanBtn.style.pointerEvents = 'none';
        
        welcomeState.classList.add('hidden');
        resultsState.classList.remove('hidden');
        resultsState.innerHTML = `
            <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; height:100%; gap:20px; color:var(--text-secondary);">
                <div class="loader" style="width:40px; height:40px; border-width:4px; border-top-color:var(--accent-purple);"></div>
                <p>Running semantic vector search...</p>
            </div>
        `;

        try {
            // Call the FastAPI endpoint
            const response = await fetch('http://127.0.0.1:8000/api/v1/analyze/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    code_snippet: code,
                    language: 'python' // default for MVP
                })
            });

            if (!response.ok) {
                throw new Error(`API Error: ${response.status}`);
            }

            const data = await response.json();
            
            // Render the Markdown report
            resultsState.innerHTML = marked.parse(data.report_markdown);
            
            // Add a subtle slide-up animation to the results
            resultsState.style.animation = 'slideUp 0.5s cubic-bezier(0.4, 0, 0.2, 1)';

        } catch (error) {
            console.error('Scan failed:', error);
            resultsState.innerHTML = `
                <div style="text-align:center; padding:2rem;">
                    <h3 style="color:var(--accent-red); margin-bottom:1rem;">Scan Failed</h3>
                    <p>${error.message}</p>
                    <p style="font-size:0.8rem; margin-top:2rem;">Is the FastAPI backend running?</p>
                </div>
            `;
        } finally {
            // Reset UI
            btnText.classList.remove('hidden');
            loader.classList.add('hidden');
            scanBtn.style.pointerEvents = 'auto';
        }
    });
});

// Add animations dynamically to the document
const style = document.createElement('style');
style.textContent = `
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        75% { transform: translateX(5px); }
    }
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
`;
document.head.appendChild(style);
