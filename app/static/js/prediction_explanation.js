/**
 * Render patient-level SHAP charts on the prediction page.
 */

document.addEventListener("DOMContentLoaded", () => {
    const charts = window.predictionExplanationCharts;
    if (!charts || typeof Plotly === "undefined") {
        return;
    }

    const chartMap = {
        waterfall: "prediction-chart-waterfall",
        force: "prediction-chart-force",
    };

    Object.entries(chartMap).forEach(([key, elementId]) => {
        const element = document.getElementById(elementId);
        const figure = charts[key];
        if (!element || !figure) {
            return;
        }

        Plotly.newPlot(element, figure.data, figure.layout, {
            responsive: true,
            displayModeBar: true,
            scrollZoom: true,
            displaylogo: false,
        });
    });
});
