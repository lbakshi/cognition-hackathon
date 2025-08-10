function selectTab(id) {
  document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
  document.querySelector(`.tab-button[data-tab="${id}"]`).classList.add('active');
  document.getElementById(id).classList.add('active');
}

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.tab-button').forEach(btn => {
    btn.addEventListener('click', () => selectTab(btn.dataset.tab));
  });

  document.getElementById('start-btn').addEventListener('click', async () => {
    const input = document.getElementById('research-input').value;
    const tabs = document.getElementById('tabs');
    tabs.style.display = 'block';

    const baseUrl = window.API_BASE_URL || '';

    // Plan
    selectTab('plan');
    const planRes = await fetch(`${baseUrl}/api/plan`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: input })
    });
    const planData = await planRes.json();
    document.getElementById('plan-output').textContent = planData.experimentSpec;

    // Codegen
    selectTab('codegen');
    const codegenRes = await fetch(`${baseUrl}/api/codegen`, { method: 'POST' });
    const codegenData = await codegenRes.json();
    const fileTabs = document.getElementById('file-tabs');
    const fileContent = document.getElementById('file-content');
    fileTabs.innerHTML = '';
    let first = true;
    Object.entries(codegenData.files || {}).forEach(([name, content]) => {
      const b = document.createElement('button');
      b.textContent = name;
      b.addEventListener('click', () => {
        fileContent.textContent = content;
      });
      fileTabs.appendChild(b);
      if (first) {
        fileContent.textContent = content;
        first = false;
      }
    });

    // Execute
    selectTab('execute');
    const executeRes = await fetch(`${baseUrl}/api/execute`, { method: 'POST' });
    const executeData = await executeRes.json();
    document.getElementById('execute-output').textContent = JSON.stringify(executeData, null, 2);

    // Report
    selectTab('report');
    const reportRes = await fetch(`${baseUrl}/api/report`, { method: 'POST' });
    const reportData = await reportRes.json();
    document.getElementById('report-output').textContent = reportData.summary;
  });
});

