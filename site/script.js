const root = document.documentElement;
const themeToggle = document.querySelector(".theme-toggle");
const storedTheme = localStorage.getItem("theme");

const setTheme = (theme) => {
  root.dataset.theme = theme;
  localStorage.setItem("theme", theme);

  if (themeToggle) {
    const isDark = theme === "dark";
    themeToggle.setAttribute("aria-pressed", String(isDark));
    themeToggle.querySelector("span").textContent = isDark ? "Dark" : "Light";
  }
};

setTheme(storedTheme || "dark");

themeToggle?.addEventListener("click", () => {
  setTheme(root.dataset.theme === "dark" ? "light" : "dark");
});

document.querySelectorAll(".stagger-group").forEach((group) => {
  group.querySelectorAll(".reveal").forEach((item, index) => {
    item.style.setProperty("--stagger-delay", `${index * 110}ms`);
  });
});

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.2 },
);

document.querySelectorAll(".reveal").forEach((item) => observer.observe(item));
