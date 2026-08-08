/**
 * Poll model training status and update the progress UI.
 */

function renderMetrics(metrics) {
    const panel = document.getElementById("metricsPanel");
    const card = document.getElementById("metricsPanelCard");
    if (!panel || !metrics) {
        return;
    }

    if (card) {
        card.classList.remove("d-none");
    }

    const rows = [
        ["Accuracy", `${(metrics.accuracy * 100).toFixed(2)}%`],
        ["Precision", `${(metrics.precision * 100).toFixed(2)}%`],
        ["Recall", `${(metrics.recall * 100).toFixed(2)}%`],
        ["F1-score", `${(metrics.f1_score * 100).toFixed(2)}%`],
    ];

    if (metrics.roc_auc !== null && metrics.roc_auc !== undefined) {
        rows.push(["ROC AUC", metrics.roc_auc.toFixed(3)]);
    }

    panel.innerHTML = `<dl class="row small mb-0">${rows
        .map(
            ([label, value]) =>
                `<dt class="col-6 text-muted">${label}</dt><dd class="col-6">${value}</dd>`
        )
        .join("")}</dl>`;
}

function updateTrainingUI(data) {
    const progressBar = document.getElementById("trainingProgressBar");
    const progressLabel = document.getElementById("trainingProgressLabel");
    const statusText = document.getElementById("trainingStatusText");
    const statusValue = document.getElementById("modelStatusValue");
    const logList = document.getElementById("trainingLogList");

    if (progressBar) {
        progressBar.style.width = `${data.progress_percent}%`;
        progressBar.setAttribute("aria-valuenow", data.progress_percent);
        progressBar.classList.toggle("progress-bar-animated", ["queued", "training"].includes(data.status));
    }

    if (progressLabel) {
        progressLabel.textContent = `${data.progress_percent}%`;
    }

    if (statusValue) {
        statusValue.textContent = data.status.charAt(0).toUpperCase() + data.status.slice(1);
    }

    if (statusText) {
        if (data.status === "completed") {
            statusText.textContent = "Training completed successfully.";
        } else if (data.status === "failed") {
            statusText.textContent = "Training failed.";
        } else {
            statusText.textContent = "Training in progress...";
        }
    }

    if (logList && Array.isArray(data.progress_log)) {
        logList.innerHTML = data.progress_log
            .map(
                (entry) => `
                <li class="list-group-item px-0 d-flex justify-content-between gap-3">
                    <span>${entry.message}</span>
                    <span class="text-muted small">${entry.percent}%</span>
                </li>`
            )
            .join("");
    }

    if (data.metrics) {
        renderMetrics(data.metrics);
    }
}

document.addEventListener("DOMContentLoaded", () => {
    const statusUrl = window.mlTrainingStatusUrl;
    const initialStatus = window.mlInitialStatus;

    if (!statusUrl || ["completed", "failed"].includes(initialStatus)) {
        return;
    }

    const poll = () => {
        fetch(statusUrl, { headers: { Accept: "application/json" } })
            .then((response) => response.json())
            .then((data) => {
                updateTrainingUI(data);
                if (!["completed", "failed"].includes(data.status)) {
                    window.setTimeout(poll, 1500);
                } else {
                    window.location.reload();
                }
            })
            .catch(() => window.setTimeout(poll, 2500));
    };

    poll();
});
