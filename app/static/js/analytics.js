/**
 * Render analytics dashboard Plotly charts.
 */

function renderAnalyticsChart(elementId, figure) {
    const element = document.getElementById(elementId);
    if (!element || !figure) {
        return;
    }

    element.innerHTML = `
        <div class="hc-loading-inline justify-content-center py-5 w-100">
            <span class="hc-spinner"></span>
            <span>Loading chart...</span>
        </div>
    `;

    Plotly.newPlot(element, figure.data, figure.layout, {
        responsive: true,
        displayModeBar: true,
        scrollZoom: true,
        displaylogo: false,
    });
}

document.addEventListener("DOMContentLoaded", () => {
    const charts = window.analyticsCharts;
    if (!charts) {
        return;
    }

    const chartMap = {
        prediction_trends: "analytics-chart-prediction-trends",
        risk_distribution: "analytics-chart-risk-distribution",
        monthly_statistics: "analytics-chart-monthly-statistics",
        feature_importance: "analytics-chart-feature-importance",
    };

    Object.entries(chartMap).forEach(([key, elementId]) => {
        renderAnalyticsChart(elementId, charts[key]);
    });

    window.addEventListener("resize", () => {
        Object.values(chartMap).forEach((elementId) => {
            const element = document.getElementById(elementId);
            if (element) {
                Plotly.Plots.resize(element);
            }
        });
    });
});
