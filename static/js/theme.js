/* Dashboard theme persistence and lightweight dashboard enhancements. */
(() => {
    const storageKey = "theme";
    const validThemes = new Set(["light", "dark", "system"]);
    const systemQuery = window.matchMedia("(prefers-color-scheme: dark)");

    const getPreference = () => {
        const savedTheme = window.localStorage.getItem(storageKey);
        return validThemes.has(savedTheme) ? savedTheme : "system";
    };

    const resolvedTheme = (preference) => (
        preference === "system"
            ? (systemQuery.matches ? "dark" : "light")
            : preference
    );

    const updateThemeControls = (preference) => {
        document.querySelectorAll("[data-theme-option]").forEach((option) => {
            const isActive = option.dataset.themeOption === preference;
            option.classList.toggle("active", isActive);
            option.setAttribute("aria-pressed", String(isActive));
        });
        document.querySelectorAll("[data-theme-check]").forEach((checkmark) => {
            checkmark.classList.toggle(
                "d-none",
                checkmark.dataset.themeCheck !== preference,
            );
        });

        const icon = document.querySelector("[data-theme-icon]");
        if (icon) {
            const iconClass = preference === "light"
                ? "bi-sun"
                : preference === "dark"
                    ? "bi-moon-stars"
                    : "bi-pc-display";
            icon.className = `bi ${iconClass}`;
        }
    };

    const applyTheme = (preference, persist = false) => {
        const root = document.documentElement;
        root.dataset.theme = resolvedTheme(preference);
        root.dataset.themePreference = preference;

        if (persist) {
            window.localStorage.setItem(storageKey, preference);
        }
        updateThemeControls(preference);
    };

    applyTheme(getPreference());

    document.addEventListener("DOMContentLoaded", () => {
        applyTheme(getPreference());
        document.querySelectorAll("[data-theme-option]").forEach((option) => {
            option.addEventListener("click", () => {
                applyTheme(option.dataset.themeOption || "system", true);
            });
        });

        const greeting = document.querySelector("[data-dashboard-greeting]");
        if (greeting) {
            const hour = new Date().getHours();
            greeting.textContent = hour < 12
                ? "Good Morning"
                : hour < 18
                    ? "Good Afternoon"
                    : "Good Evening";
        }
    });

    systemQuery.addEventListener("change", () => {
        if (getPreference() === "system") {
            applyTheme("system");
        }
    });
})();


