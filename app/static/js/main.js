/**
 * Shared UI interactions — loading states, form feedback, navigation.
 */

const PageLoader = {
    overlay: null,

    init() {
        this.overlay = document.getElementById("pageLoader");
    },

    show(label = "Loading...") {
        if (!this.overlay) {
            return;
        }
        const labelEl = this.overlay.querySelector(".hc-loading-overlay__label");
        if (labelEl) {
            labelEl.textContent = label;
        }
        this.overlay.classList.add("is-active");
        this.overlay.setAttribute("aria-hidden", "false");
    },

    hide() {
        if (!this.overlay) {
            return;
        }
        this.overlay.classList.remove("is-active");
        this.overlay.setAttribute("aria-hidden", "true");
    },
};

const bindFormLoading = () => {
    document.querySelectorAll("form").forEach((form) => {
        form.addEventListener("submit", (event) => {
            if (event.defaultPrevented) {
                return;
            }
            if (form.dataset.noLoader === "true") {
                return;
            }

            const submitter = form.querySelector('[type="submit"]');
            if (submitter && !submitter.disabled) {
                submitter.classList.add("is-loading");
                submitter.disabled = true;
            }
        });
    });
};

const bindNavigationLoader = () => {
    document.addEventListener("click", (event) => {
        const link = event.target.closest("a[href]");
        if (!link) {
            return;
        }

        const href = link.getAttribute("href");
        if (!href || href.startsWith("#") || href.startsWith("javascript:")) {
            return;
        }
        if (link.target === "_blank" || link.hasAttribute("download")) {
            return;
        }
        if (link.origin !== window.location.origin) {
            return;
        }
        if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) {
            return;
        }

        PageLoader.show();
    });
};

document.addEventListener("DOMContentLoaded", () => {
    PageLoader.init();
    PageLoader.hide();
    bindFormLoading();
    bindNavigationLoader();

    window.addEventListener("pageshow", (event) => {
        if (event.persisted) {
            PageLoader.hide();
            document.querySelectorAll(".btn.is-loading").forEach((btn) => {
                btn.classList.remove("is-loading");
                btn.disabled = false;
            });
        }
    });
});

window.HC = { PageLoader };
