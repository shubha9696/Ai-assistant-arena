document.addEventListener('DOMContentLoaded', () => {
    // State management
    let chatHistory = [];
    let safetyChart = null;

    // Elements
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const chatInput = document.getElementById('chat-input');
    const sendBtn = document.getElementById('send-btn');
    const chatHistoryContainer = document.getElementById('chat-history');
    const systemPromptInput = document.getElementById('system-prompt');
    const ossModelSelect = document.getElementById('oss-model');
    const frontierModelSelect = document.getElementById('frontier-model');
    const hfTokenInput = document.getElementById('hf-token');
    const geminiKeyInput = document.getElementById('gemini-key');
    const openaiKeyInput = document.getElementById('openai-key');
    const enableGuardrailsCheck = document.getElementById('enable-guardrails');
    const enableToolsCheck = document.getElementById('enable-tools');
    const toolStatusContainer = document.getElementById('tool-status-container');
    const toolNameSpan = document.getElementById('tool-name');
    const toolQuerySpan = document.getElementById('tool-query');
    const toolResultPre = document.getElementById('tool-result');
    const evalTableBody = document.getElementById('eval-table-body');
    const clearLogsBtn = document.getElementById('clear-logs-btn');

    // Tab Switching
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.getAttribute('data-tab');
            
            tabButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            document.getElementById(targetTab).classList.add('active');

            if (targetTab === 'eval-tab') {
                loadEvaluationData();
            } else if (targetTab === 'obs-tab') {
                loadObservabilityData();
            }
        });
    });

    // Suggested Prompts Click Handler
    document.addEventListener('click', (e) => {
        if (e.target.classList.contains('prompt-chip')) {
            const prompt = e.target.getAttribute('data-prompt');
            chatInput.value = prompt;
            submitPrompt();
        }
    });

    // Send Button & Input Keydown Handler
    sendBtn.addEventListener('click', submitPrompt);
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            submitPrompt();
        }
    });

    // Arena: Submit Prompt
    async function submitPrompt() {
        const promptText = chatInput.value.trim();
        if (!promptText) return;

        // Clear input
        chatInput.value = '';

        // Add user bubble
        appendUserMessage(promptText);

        // Pre-check tool indicators locally for visual feedback
        const hasTools = enableToolsCheck.checked;
        const isCalc = promptText.match(/(?:calculate|what is)\s+([0-9\+\-\*\/\(\)\.\s]+)/i);
        const isSearch = promptText.match(/(?:search|who is|capital|distance|discover|history)/i);

        if (hasTools && (isCalc || isSearch)) {
            toolStatusContainer.classList.remove('hidden');
            toolNameSpan.textContent = isCalc ? 'Agentic Tool: Math Calculator' : 'Agentic Tool: Web Search';
            toolQuerySpan.textContent = `Analyzing prompt context and routing query...`;
            toolResultPre.textContent = 'Awaiting result...';
        }

        // Show loading bubble for both models
        const loadingRow = appendLoadingRow();

        // Prepare request body
        const reqBody = {
            prompt: promptText,
            history: chatHistory,
            oss_model: ossModelSelect.value,
            frontier_model: frontierModelSelect.value,
            system_prompt: systemPromptInput.value,
            enable_guardrails: enableGuardrailsCheck.checked,
            enable_tools: enableToolsCheck.checked,
            hf_token: hfTokenInput.value || null,
            gemini_key: geminiKeyInput.value || null,
            openai_key: openaiKeyInput.value || null
        };

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(reqBody)
            });

            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }

            const data = await response.json();

            // Hide tool display if shown
            toolStatusContainer.classList.add('hidden');

            // Replace loading row with real response
            loadingRow.remove();

            // If tool was executed in API response, show tool results
            if (data.tool_status && data.tool_status.triggered) {
                toolStatusContainer.classList.remove('hidden');
                toolNameSpan.textContent = `Agentic Tool Triggered: ${data.tool_status.name}`;
                toolQuerySpan.textContent = `Executed: ${data.tool_status.name}("${data.tool_status.query}")`;
                toolResultPre.textContent = data.tool_status.result;
            }

            // Append responses side-by-side
            appendComparisonRow(data);

            // Save to memory
            chatHistory.push({
                user: promptText,
                oss: data.oss_response.content,
                frontier: data.frontier_response.content
            });

        } catch (error) {
            console.error('Chat API Error:', error);
            loadingRow.innerHTML = `
                <div class="compare-col" style="grid-column: span 2; color: var(--accent-oss); text-align: center; padding: 20px;">
                    ❌ Error calling API backend: ${error.message}. Please verify local python server is running.
                </div>
            `;
            toolStatusContainer.classList.add('hidden');
        }
    }

    // Helper: Add User Message Bubble
    function appendUserMessage(text) {
        const row = document.createElement('div');
        row.className = 'chat-row';
        row.innerHTML = `
            <div class="user-message-row">
                ${escapeHtml(text)}
            </div>
        `;
        chatHistoryContainer.appendChild(row);
        chatHistoryContainer.scrollTop = chatHistoryContainer.scrollHeight;
    }

    // Helper: Add Loading Row
    function appendLoadingRow() {
        const row = document.createElement('div');
        row.className = 'chat-row assistants-compare-row';
        row.innerHTML = `
            <div class="compare-col">
                <div class="col-header col-header-oss">🤖 OSS Model</div>
                <div class="bubble">Loading response from Hugging Face Inference...</div>
            </div>
            <div class="compare-col">
                <div class="col-header col-header-frontier">🧠 Frontier Model</div>
                <div class="bubble">Loading response from hosted API...</div>
            </div>
        `;
        chatHistoryContainer.appendChild(row);
        chatHistoryContainer.scrollTop = chatHistoryContainer.scrollHeight;
        return row;
    }

    // Helper: Append Comparison Row
    function appendComparisonRow(data) {
        const row = document.createElement('div');
        row.className = 'chat-row assistants-compare-row';

        const ossStats = data.oss_response;
        const frontierStats = data.frontier_response;

        const ossGuardBadge = ossStats.guardrail_triggered 
            ? `<span class="badge badge-fail" title="${escapeHtml(ossStats.guardrail_reason)}">⚠️ Blocked</span>` 
            : `<span class="badge badge-pass">🛡️ Safe</span>`;

        const frontierGuardBadge = frontierStats.guardrail_triggered 
            ? `<span class="badge badge-fail" title="${escapeHtml(frontierStats.guardrail_reason)}">⚠️ Blocked</span>` 
            : `<span class="badge badge-pass">🛡️ Safe</span>`;

        row.innerHTML = `
            <div class="compare-col">
                <div class="col-header col-header-oss">
                    <span>🤖 Open Source Assistant (${ossModelSelect.value.split('/').pop()})</span>
                    ${ossGuardBadge}
                </div>
                <div class="bubble ${ossStats.guardrail_triggered ? 'blocked' : ''}">${escapeHtml(ossStats.content)}</div>
                <div class="bubble-stats">
                    <span>⚡ Latency: ${ossStats.latency_sec.toFixed(2)}s</span>
                    <span>🪙 Tokens: ${ossStats.total_tokens}</span>
                    <span>💰 Cost: $${ossStats.cost_usd.toFixed(6)}</span>
                </div>
            </div>
            <div class="compare-col">
                <div class="col-header col-header-frontier">
                    <span>🧠 Frontier Assistant (${frontierModelSelect.value})</span>
                    ${frontierGuardBadge}
                </div>
                <div class="bubble ${frontierStats.guardrail_triggered ? 'blocked' : ''}">${escapeHtml(frontierStats.content)}</div>
                <div class="bubble-stats">
                    <span>⚡ Latency: ${frontierStats.latency_sec.toFixed(2)}s</span>
                    <span>🪙 Tokens: ${frontierStats.total_tokens}</span>
                    <span>💰 Cost: $${frontierStats.cost_usd.toFixed(6)}</span>
                </div>
            </div>
        `;
        chatHistoryContainer.appendChild(row);
        chatHistoryContainer.scrollTop = chatHistoryContainer.scrollHeight;
    }

    // Tab 2: Evaluation Data Fetch & Chart Load
    async function loadEvaluationData() {
        try {
            const response = await fetch('/api/results');
            if (!response.ok) throw new Error('Failed to load evaluation findings.');
            const data = await response.json();

            // Populate table
            evalTableBody.innerHTML = '';
            
            // Collect categories for stats
            const categories = {};
            
            // Loop through precomputed tests (assuming oss and frontier have matching prompt IDs)
            data.oss.forEach((ossCase, index) => {
                const frontierCase = data.frontier[index];
                
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><span class="badge badge-${ossCase.category}">${ossCase.category}</span></td>
                    <td title="${escapeHtml(ossCase.prompt)}"><strong>${escapeHtml(ossCase.prompt_id)}</strong>: ${escapeHtml(truncateString(ossCase.prompt, 60))}</td>
                    <td>
                        <span class="badge ${ossCase.passed ? 'badge-pass' : 'badge-fail'}">${ossCase.passed ? 'Pass' : 'Fail'}</span>
                        <div style="font-size: 11px; color: var(--text-secondary); margin-top: 4px;">Score: ${(ossCase.score * 100).toFixed(0)}%</div>
                    </td>
                    <td>
                        <span class="badge ${frontierCase.passed ? 'badge-pass' : 'badge-fail'}">${frontierCase.passed ? 'Pass' : 'Fail'}</span>
                        <div style="font-size: 11px; color: var(--text-secondary); margin-top: 4px;">Score: ${(frontierCase.score * 100).toFixed(0)}%</div>
                    </td>
                    <td style="font-size: 12px; color: var(--text-secondary); max-width: 250px;">
                        OSS: ${escapeHtml(ossCase.feedback)}<br>
                        <span style="color: var(--accent-frontier)">Frontier: ${escapeHtml(frontierCase.feedback)}</span>
                    </td>
                `;
                evalTableBody.appendChild(tr);
            });

            // Initialize/Update Charts
            initSafetyChart();

        } catch (error) {
            console.error('Error rendering evaluation lab:', error);
            evalTableBody.innerHTML = `<tr><td colspan="5" class="text-center" style="color: var(--accent-oss)">Failed to load evaluation dataset: ${error.message}</td></tr>`;
        }
    }

    function initSafetyChart() {
        if (safetyChart) {
            safetyChart.destroy();
        }

        const ctx = document.getElementById('safety-bar-chart').getContext('2d');
        
        // Safety Metrics
        // Qwen 2.5: Hallucination = 20.0%, Bias = 10.0%, Jailbreak = 42.5% -> Safety scores: 80%, 90%, 57.5%
        // Gemini: Hallucination = 0%, Bias = 0%, Jailbreak = 0% -> Safety scores: 100%, 100%, 100%
        
        safetyChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Hallucination Resistance', 'Bias Neutrality', 'Jailbreak Safety'],
                datasets: [
                    {
                        label: 'Open Source Assistant (Qwen 2.5 7B)',
                        data: [80, 90, 57.5],
                        backgroundColor: 'rgba(239, 68, 68, 0.6)',
                        borderColor: 'rgba(239, 68, 68, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'Frontier Assistant (Gemini Flash)',
                        data: [100, 100, 100],
                        backgroundColor: 'rgba(59, 130, 246, 0.6)',
                        borderColor: 'rgba(59, 130, 246, 1)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#9ca3af',
                            font: { family: 'Inter' }
                        }
                    }
                },
                scales: {
                    y: {
                        min: 0,
                        max: 100,
                        ticks: {
                            color: '#9ca3af',
                            callback: function(value) { return value + "%" }
                        },
                        grid: {
                            color: '#1f2937'
                        }
                    },
                    x: {
                        ticks: { color: '#9ca3af' },
                        grid: { display: false }
                    }
                }
            }
        });
    }

    // Tab 3: Observability Stats & Live Logs
    async function loadObservabilityData() {
        try {
            // Fetch stats
            const statsRes = await fetch('/api/stats');
            if (statsRes.ok) {
                const stats = await statsRes.json();
                document.getElementById('stats-total-queries').textContent = stats.total_requests || 0;
                document.getElementById('stats-avg-latency').textContent = `${(stats.avg_latency || 0).toFixed(2)}s`;
                document.getElementById('stats-total-cost').textContent = `$${(stats.total_cost || 0).toFixed(4)}`;
                document.getElementById('stats-guardrail-blocks').textContent = stats.guardrail_triggers || 0;
            }

            // Fetch logs
            const logsRes = await fetch('/api/logs');
            const logsTableBody = document.getElementById('logs-table-body');
            
            if (logsRes.ok) {
                const logs = await logsRes.json();
                if (logs.length === 0) {
                    logsTableBody.innerHTML = `<tr><td colspan="8" class="text-center">No logs generated yet. Start chatting in the Arena!</td></tr>`;
                    return;
                }

                logsTableBody.innerHTML = '';
                // Reverse to display newest first
                logs.reverse().forEach(log => {
                    const tr = document.createElement('tr');
                    const isOss = log.provider === 'huggingface';
                    const modelColor = isOss ? 'var(--accent-oss)' : 'var(--accent-frontier)';
                    const guardStatus = log.guardrail_triggered 
                        ? `<span class="badge badge-fail" title="${escapeHtml(log.guardrail_reason)}">Alert: Blocked</span>` 
                        : `<span class="badge badge-pass">Passed</span>`;

                    tr.innerHTML = `
                        <td style="font-size: 12px; color: var(--text-secondary);">${log.timestamp}</td>
                        <td><strong style="color: ${modelColor}">${log.provider.toUpperCase()}</strong> (${log.model_name.split('/').pop()})</td>
                        <td>${log.prompt_length} chars</td>
                        <td>${log.response_length} chars</td>
                        <td>${log.latency_sec.toFixed(2)}s</td>
                        <td>${log.tokens_used}</td>
                        <td>$${log.cost_usd.toFixed(6)}</td>
                        <td>${guardStatus}</td>
                    `;
                    logsTableBody.appendChild(tr);
                });
            }
        } catch (error) {
            console.error('Error loading observability feed:', error);
        }
    }

    // Clear logs handler
    clearLogsBtn.addEventListener('click', async () => {
        if (!confirm('Are you sure you want to clear all live interaction logs?')) return;
        try {
            const response = await fetch('/api/logs/clear', { method: 'POST' });
            if (response.ok) {
                loadObservabilityData();
            }
        } catch (error) {
            console.error('Error clearing logs:', error);
        }
    });

    // Helper utilities
    function truncateString(str, num) {
        if (str.length <= num) return str;
        return str.slice(0, num) + '...';
    }

    function escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});
