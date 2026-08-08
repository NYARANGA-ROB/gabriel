/**
 * Render evaluation Plotly charts on the model detail page.
 */

function renderEvaluationChart(elementId, figure) {
    const element = document.getElementById(elementId);
    if (!element || !figure) {
        return;
    }

    Plotly.newPlot(element, figure.data, figure.layout, {
        responsive: true,
        displayModeBar: true,
        scrollZoom: true,
        displaylogo: false,
    });
}

document.addEventListener("DOMContentLoaded", () => {
    const charts = window.mlEvaluationCharts;
    if (!charts) {
        return;
    }

    const chartMap = {
        confusion_matrix: "eval-chart-confusion-matrix",
        normalized_confusion_matrix: "eval-chart-normalized-cm",
        roc_curve: "eval-chart-roc-curve",
        precision_recall_curve: "eval-chart-pr-curve",
        calibration: "eval-chart-calibration",
        prediction_distribution: "eval-chart-prediction-dist",
        feature_importance: "eval-chart-feature-importance",
    };

    Object.entries(chartMap).forEach(([key, elementId]) => {
        renderEvaluationChart(elementId, charts[key]);
    });
});
