/**
 * Render interactive Plotly charts for the EDA page.
 */

function renderPlotlyChart(elementId, figure) {
    const element = document.getElementById(elementId);
    if (!element || !figure) {
        return;
    }

    const config = {
        responsive: true,
        displayModeBar: true,
        scrollZoom: true,
        displaylogo: false,
    };

    Plotly.newPlot(element, figure.data, figure.layout, config);
}

document.addEventListener("DOMContentLoaded", () => {
    const payload = window.edaChartPayload;
    if (!payload) {
        return;
    }

    const chartMap = {
        missing_values: "chart-missing-values",
        class_distribution: "chart-class-distribution",
        correlation_heatmap: "chart-correlation-heatmap",
        feature_distribution: "chart-feature-distribution",
        histogram: "chart-histogram",
        boxplot: "chart-boxplot",
        scatter_plot: "chart-scatter-plot",
        target_rate_by_category: "chart-target-rate",
        categorical_overview: "chart-categorical-overview",
        numeric_summary: "chart-numeric-summary",
    };

    Object.entries(chartMap).forEach(([key, elementId]) => {
        renderPlotlyChart(elementId, payload[key]);
    });
});
