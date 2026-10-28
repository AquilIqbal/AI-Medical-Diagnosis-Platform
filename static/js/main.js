/* Small progressive enhancement for shared templates and form controls. */
document.addEventListener("DOMContentLoaded", () => {
    const yearElement = document.querySelector("[data-current-year]");
    if (yearElement) {
        yearElement.textContent = String(new Date().getFullYear());
    }

    const passwordToggle = document.querySelector("[data-password-toggle]");
    if (passwordToggle instanceof HTMLButtonElement) {
        const passwordInput = document.getElementById(
            passwordToggle.dataset.passwordTarget || "",
        );

        if (passwordInput instanceof HTMLInputElement) {
            passwordToggle.addEventListener("click", () => {
                const isHidden = passwordInput.type === "password";
                passwordInput.type = isHidden ? "text" : "password";
                passwordToggle.setAttribute("aria-pressed", String(isHidden));
                passwordToggle.setAttribute(
                    "aria-label",
                    isHidden ? "Hide password" : "Show password",
                );

                const icon = passwordToggle.querySelector("i");
                icon?.classList.toggle("bi-eye", !isHidden);
                icon?.classList.toggle("bi-eye-slash", isHidden);
            });
        }
    }

    const validationForms = document.querySelectorAll("[data-validate-form]");
    validationForms.forEach((form) => {
        form.addEventListener("submit", (event) => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add("was-validated");
        });
    });
});
