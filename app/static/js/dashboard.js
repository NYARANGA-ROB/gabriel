/**
 * Dashboard sidebar interactions and animations.
 */

document.addEventListener("DOMContentLoaded", () => {
    const sidebar = document.getElementById("dashboardSidebar");
    const toggle = document.getElementById("sidebarToggle");
    const backdrop = document.getElementById("sidebarBackdrop");

    if (!sidebar) {
        return;
    }

    sidebar.classList.add("is-animating");
    setTimeout(() => sidebar.classList.remove("is-animating"), 600);

    if (!toggle || !backdrop) {
        return;
    }

    const closeSidebar = () => {
        sidebar.classList.remove("show");
        backdrop.classList.remove("is-visible");
        backdrop.hidden = true;
        toggle.setAttribute("aria-expanded", "false");
        document.body.classList.remove("sidebar-open");
    };

    const openSidebar = () => {
        sidebar.classList.add("show");
        backdrop.hidden = false;
        requestAnimationFrame(() => backdrop.classList.add("is-visible"));
        toggle.setAttribute("aria-expanded", "true");
        document.body.classList.add("sidebar-open");
    };

    toggle.addEventListener("click", () => {
        if (sidebar.classList.contains("show")) {
            closeSidebar();
        } else {
            openSidebar();
        }
    });

    backdrop.addEventListener("click", closeSidebar);

    sidebar.querySelectorAll(".nav-link").forEach((link) => {
        link.addEventListener("click", () => {
            if (window.innerWidth < 992 && !link.classList.contains("disabled")) {
                closeSidebar();
            }
        });
    });

    window.addEventListener("resize", () => {
        if (window.innerWidth >= 992) {
            closeSidebar();
        }
    });

    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
            closeSidebar();
        }
    });
});
