/**
 * Render model comparison Plotly charts.
 */

function renderComparisonChart(elementId, figure) {
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
    const charts = window.mlComparisonCharts;
    if (!charts) {
        return;
    }

    const chartMap = {
        roc_comparison: "compare-chart-roc",
        pr_comparison: "compare-chart-pr",
        roc_auc: "compare-chart-roc-auc",
        accuracy: "compare-chart-accuracy",
        precision: "compare-chart-precision",
        recall: "compare-chart-recall",
        f1_score: "compare-chart-f1",
        training_time: "compare-chart-training-time",
        prediction_time: "compare-chart-prediction-time",
    };

    Object.entries(chartMap).forEach(([key, elementId]) => {
        renderComparisonChart(elementId, charts[key]);
    });
});
