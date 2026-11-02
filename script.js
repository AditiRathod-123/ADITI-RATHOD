const newsText = document.getElementById("newsText");
const charCount = document.getElementById("charCount");
const analyseBtn = document.getElementById("analyseBtn");
const errorBox = document.getElementById("errorBox");
const resultPanel = document.getElementById("resultPanel");

const samples = {
  real: "The city council approved its annual public transport budget during a meeting on Tuesday. The published budget document lists funding for bus maintenance, route improvements and accessibility upgrades.",
  fake: "BREAKING: A secret miracle drink makes every person instantly fluent in every language overnight, researchers say. Officials are hiding the discovery and everyone must share this message immediately."
};

newsText.addEventListener("input", () => {
  charCount.textContent = `${newsText.value.length.toLocaleString()} / 10,000 characters`;
});

document.getElementById("clearBtn").addEventListener("click", () => {
  newsText.value = "";
  newsText.dispatchEvent(new Event("input"));
  resultPanel.style.display = "none";
  errorBox.style.display = "none";
});

document.querySelectorAll(".sample-btn").forEach(button => {
  button.addEventListener("click", () => {
    newsText.value = samples[button.dataset.sample];
    newsText.dispatchEvent(new Event("input"));
    newsText.focus();
  });
});

analyseBtn.addEventListener("click", async () => {
  errorBox.style.display = "none";
  resultPanel.style.display = "none";
  const text = newsText.value.trim();

  if (text.length < 25) {
    showError("Please enter at least 25 characters so the model has enough text to analyse.");
    return;
  }

  analyseBtn.disabled = true;
  analyseBtn.querySelector("span:first-child").textContent = "Analysing text...";
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({news_text: text})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not analyse this text.");

    const prediction = document.getElementById("predictionText");
    prediction.textContent = data.prediction;
    prediction.className = data.prediction.toLowerCase();
    document.getElementById("confidenceText").textContent = `${data.confidence}%`;
    document.getElementById("scoreRing").style.setProperty("--score", `${data.confidence}%`);
    document.getElementById("resultMessage").textContent = data.message;
    document.getElementById("disclaimer").textContent = data.disclaimer;
    resultPanel.style.display = "block";
    resultPanel.scrollIntoView({behavior: "smooth", block: "nearest"});
  } catch (error) {
    showError(error.message);
  } finally {
    analyseBtn.disabled = false;
    analyseBtn.querySelector("span:first-child").textContent = "Analyse news text";
  }
});

function showError(message) {
  errorBox.textContent = message;
  errorBox.style.display = "block";
}
