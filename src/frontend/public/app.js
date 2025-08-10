// Utility functions
function selectTab(id) {
  // Remove active class from all tabs and buttons
  document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
  
  // Add active class to selected tab and button
  document.querySelector(`.tab-button[data-tab="${id}"]`).classList.add('active');
  document.getElementById(id).classList.add('active');
}

function updateStatus(tabId, status, text) {
  const statusElement = document.querySelector(`#${tabId} .status-indicator`);
  if (statusElement) {
    statusElement.className = `status-indicator status-${status}`;
    statusElement.textContent = text;
  }
}

function showLoading() {
  document.getElementById('loading').classList.remove('hidden');
  document.getElementById('tabs').classList.add('hidden');
}

function hideLoading() {
  document.getElementById('loading').classList.add('hidden');
  document.getElementById('tabs').classList.remove('hidden');
}

function createFileTab(name, content, isFirst = false) {
  const button = document.createElement('button');
  button.className = `btn-secondary text-sm ${isFirst ? 'bg-primary-50 border-primary-300 text-primary-700' : ''}`;
  button.textContent = name;
  
  button.addEventListener('click', () => {
    // Update active state
    document.querySelectorAll('#file-tabs button').forEach(btn => {
      btn.className = 'btn-secondary text-sm';
    });
    button.className = 'btn-secondary text-sm bg-primary-50 border-primary-300 text-primary-700';
    
    // Update content
    document.getElementById('file-content').textContent = content;
  });
  
  return button;
}

// Main application logic
document.addEventListener('DOMContentLoaded', () => {
  // Set up tab navigation
  document.querySelectorAll('.tab-button').forEach(btn => {
    btn.addEventListener('click', () => selectTab(btn.dataset.tab));
  });

  // Set up research start button
  document.getElementById('start-btn').addEventListener('click', async () => {
    const input = document.getElementById('research-input').value.trim();
    
    if (!input) {
      alert('Please describe your experiment first!');
      return;
    }

    showLoading();
    
    try {
      const baseUrl = window.API_BASE_URL || '';
      
      // Step 1: Plan
      updateStatus('plan', 'pending', 'Planning...');
      selectTab('plan');
      
      const planRes = await fetch(`${baseUrl}/api/plan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: input })
      });
      
      if (!planRes.ok) throw new Error(`Plan API error: ${planRes.status}`);
      
      const planData = await planRes.json();
      document.getElementById('plan-output').textContent = planData.experimentSpec || 'No plan generated';
      updateStatus('plan', 'success', 'Complete');

      // Step 2: Code Generation
      updateStatus('codegen', 'pending', 'Generating...');
      selectTab('codegen');
      
      const codegenRes = await fetch(`${baseUrl}/api/codegen`, { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (!codegenRes.ok) throw new Error(`Codegen API error: ${codegenRes.status}`);
      
      const codegenData = await codegenRes.json();
      const fileTabs = document.getElementById('file-tabs');
      const fileContent = document.getElementById('file-content');
      
      fileTabs.innerHTML = '';
      let first = true;
      
      if (codegenData.files && Object.keys(codegenData.files).length > 0) {
        Object.entries(codegenData.files).forEach(([name, content]) => {
          const fileTab = createFileTab(name, content, first);
          fileTabs.appendChild(fileTab);
          
          if (first) {
            fileContent.textContent = content;
            first = false;
          }
        });
        updateStatus('codegen', 'success', 'Complete');
      } else {
        fileContent.textContent = 'No code files generated';
        updateStatus('codegen', 'error', 'No files');
      }

      // Step 3: Execution
      updateStatus('execute', 'pending', 'Running...');
      selectTab('execute');
      
      const executeRes = await fetch(`${baseUrl}/api/execute`, { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (!executeRes.ok) throw new Error(`Execute API error: ${executeRes.status}`);
      
      const executeData = await executeRes.json();
      document.getElementById('execute-output').textContent = JSON.stringify(executeData, null, 2);
      updateStatus('execute', 'success', 'Complete');

      // Step 4: Report Generation
      updateStatus('report', 'pending', 'Analyzing...');
      selectTab('report');
      
      const reportRes = await fetch(`${baseUrl}/api/report`, { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (!reportRes.ok) throw new Error(`Report API error: ${reportRes.status}`);
      
      const reportData = await reportRes.json();
      document.getElementById('report-output').textContent = reportData.summary || 'No report generated';
      updateStatus('report', 'success', 'Complete');

      // Show success message
      setTimeout(() => {
        selectTab('report');
      }, 500);

    } catch (error) {
      console.error('Research execution error:', error);
      
      // Show error in current tab
      const currentTab = document.querySelector('.tab-content.active');
      if (currentTab) {
        const outputId = currentTab.querySelector('pre').id;
        document.getElementById(outputId).textContent = `Error: ${error.message}`;
        updateStatus(currentTab.id, 'error', 'Error');
      }
      
      // Show error alert
      alert(`Research execution failed: ${error.message}`);
    } finally {
      hideLoading();
    }
  });

  // Add enter key support for input
  document.getElementById('research-input').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
      document.getElementById('start-btn').click();
    }
  });

  // Add some nice hover effects and animations
  document.querySelectorAll('.card').forEach(card => {
    card.addEventListener('mouseenter', () => {
      card.classList.add('shadow-md');
      card.classList.remove('shadow-sm');
    });
    
    card.addEventListener('mouseleave', () => {
      card.classList.remove('shadow-md');
      card.classList.add('shadow-sm');
    });
  });
});

