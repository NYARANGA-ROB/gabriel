/**
 * Render SHAP, LIME, and comparison Plotly charts.
 */

function renderExplainabilityChart(elementId, figure) {
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
    const charts = window.explainabilityCharts;
    if (!charts) {
        return;
    }

    const chartMap = {
        summary: "explain-chart-summary",
        feature_importance: "explain-chart-feature-importance",
        dependence: "explain-chart-dependence",
        local_waterfall: "explain-chart-local-waterfall",
        local_force: "explain-chart-local-force",
        lime_lime_weights: "explain-chart-lime-weights",
        lime_interactive: "explain-chart-lime-interactive",
        shap_lime_comparison: "explain-chart-shap-lime-comparison",
    };

    Object.entries(chartMap).forEach(([key, elementId]) => {
        renderExplainabilityChart(elementId, charts[key]);
    });

    const sourceManual = document.getElementById("patientSourceManual");
    const sourceTest = document.getElementById("patientSourceTest");
    const manualPanel = document.getElementById("manualPatientPanel");
    const testPanel = document.getElementById("testPatientPanel");

    function togglePatientPanels() {
        if (!manualPanel || !testPanel || !sourceManual || !sourceTest) {
            return;
        }
        const useTest = sourceTest.checked;
        manualPanel.classList.toggle("d-none", useTest);
        testPanel.classList.toggle("d-none", !useTest);
    }

    sourceManual?.addEventListener("change", togglePatientPanels);
    sourceTest?.addEventListener("change", togglePatientPanels);
    togglePatientPanels();
});
