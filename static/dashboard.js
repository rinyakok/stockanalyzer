(() => {
  const favoritesKey = "stock-dashboard-favorites-v1";
  const form = document.getElementById("searchForm");
  const queryInput = document.getElementById("query");
  const lookupMode = document.getElementById("lookupMode");
  const favoritesSelect = document.getElementById("favorites");
  const favoriteButton = document.getElementById("favoriteButton");
  const analyzeButton = document.getElementById("analyzeButton");
  const status = document.getElementById("status");
  const emptyState = document.getElementById("emptyState");
  const reportFrame = document.getElementById("reportFrame");
  let favorites = [];
  let currentInstrument = null;
  let frameResizeObserver = null;

  try {
    const saved = JSON.parse(localStorage.getItem(favoritesKey) || "[]");
    if (Array.isArray(saved)) favorites = saved.filter((item) => item && item.symbol && item.label);
  } catch {}

  function renderFavorites() {
    favoritesSelect.replaceChildren(new Option(favorites.length ? "Choose a favorite" : "No saved favorites", ""));
    favorites.forEach((instrument) => {
      favoritesSelect.add(new Option(`${instrument.label} · ${instrument.symbol}`, instrument.symbol));
    });
  }

  function updateFavoriteButton() {
    favoriteButton.disabled = !currentInstrument;
    favoriteButton.textContent = currentInstrument && favorites.some((item) => item.symbol === currentInstrument.symbol)
      ? "Remove favorite"
      : "Save favorite";
  }

  function setStatus(message, state = "") {
    status.textContent = message;
    status.dataset.state = state;
  }

  function detectType(value) {
    if (lookupMode.value !== "auto") return lookupMode.value;
    if (/^[A-Z]{2}[A-Z0-9]{10}$/i.test(value)) return "isin";
    if (/\s/.test(value)) return "name";
    return "ticker";
  }

  function resizeReportFrame() {
    const reportDocument = reportFrame.contentDocument;
    if (!reportDocument?.documentElement) return;
    const height = Math.max(900, reportDocument.documentElement.scrollHeight + 24);
    reportFrame.style.height = `${height}px`;
  }

  async function analyzeInstrument(type, value) {
    analyzeButton.disabled = true;
    setStatus(`Looking up ${value}…`);
    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({type, value}),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "Analysis failed.");

      currentInstrument = {symbol: result.symbol, label: result.label};
      favoritesSelect.value = favorites.some((item) => item.symbol === result.symbol) ? result.symbol : "";
      updateFavoriteButton();
      if (frameResizeObserver) frameResizeObserver.disconnect();
      reportFrame.onload = () => {
        resizeReportFrame();
        if (reportFrame.contentDocument?.body && "ResizeObserver" in window) {
          frameResizeObserver = new ResizeObserver(resizeReportFrame);
          frameResizeObserver.observe(reportFrame.contentDocument.body);
        }
      };
      reportFrame.srcdoc = result.report_html;
      reportFrame.hidden = false;
      emptyState.hidden = true;
      setStatus(`${result.label} · ${result.symbol}`, "success");
    } catch (error) {
      setStatus(error.message || "Could not load this instrument.", "error");
    } finally {
      analyzeButton.disabled = false;
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const value = queryInput.value.trim();
    if (value) analyzeInstrument(detectType(value), value);
  });

  favoritesSelect.addEventListener("change", () => {
    const favorite = favorites.find((item) => item.symbol === favoritesSelect.value);
    if (!favorite) return;
    queryInput.value = favorite.symbol;
    lookupMode.value = "ticker";
    analyzeInstrument("ticker", favorite.symbol);
  });

  favoriteButton.addEventListener("click", () => {
    if (!currentInstrument) return;
    const existingIndex = favorites.findIndex((item) => item.symbol === currentInstrument.symbol);
    if (existingIndex >= 0) favorites.splice(existingIndex, 1);
    else favorites.unshift(currentInstrument);
    try { localStorage.setItem(favoritesKey, JSON.stringify(favorites)); } catch {}
    renderFavorites();
    favoritesSelect.value = favorites.some((item) => item.symbol === currentInstrument.symbol)
      ? currentInstrument.symbol
      : "";
    updateFavoriteButton();
  });

  renderFavorites();
})();