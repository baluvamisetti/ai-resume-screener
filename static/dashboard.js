// ============================================================
// dashboard.js — AI Resume Screener
// Shared JS utilities across all pages
// ============================================================

document.addEventListener("DOMContentLoaded", function () {

  // ── Animate progress bars that have data-width ───────────
  document.querySelectorAll(".progress-bar[data-width]")
          .forEach(bar => {
    const width = bar.getAttribute("data-width");
    setTimeout(() => {
      bar.style.transition = "width 1s ease-in-out";
      bar.style.width      = width + "%";
    }, 400);
  });

  // ── Animate score circle on results page ─────────────────
  const scoreFill = document.querySelector(".score-fill");
  if (scoreFill) {
    const circumference = 2 * Math.PI * 85;
    const dashVal       = parseFloat(
      scoreFill.getAttribute("stroke-dasharray")
    );
    scoreFill.style.strokeDasharray = "0 " + circumference;
    setTimeout(() => {
      scoreFill.style.transition =
        "stroke-dasharray 1.4s ease-in-out";
      scoreFill.style.strokeDasharray =
        dashVal + " " + circumference;
    }, 500);
  }

});

// ── Auto-dismiss flash messages after 5 s ─────────────────
setTimeout(function () {
  document.querySelectorAll(".alert").forEach(a => {
    a.style.transition = "opacity 0.5s";
    a.style.opacity    = "0";
    setTimeout(() => a.remove(), 500);
  });
}, 5000);
