/**
 * Toggle model-specific hyperparameter panels on the training form.
 */

document.addEventListener("DOMContentLoaded", () => {
    const modelSelect = document.getElementById("modelTypeSelect");
    if (!modelSelect) {
        return;
    }

    const panels = document.querySelectorAll(".param-panel");

    const showPanel = () => {
        const selected = modelSelect.value;
        panels.forEach((panel) => {
            panel.classList.toggle("d-none", panel.id !== `params-${selected}`);
        });
    };

    modelSelect.addEventListener("change", showPanel);
    showPanel();
});
