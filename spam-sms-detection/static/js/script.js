const messageBox = document.getElementById("message");
const checkBtn = document.getElementById("checkBtn");
const clearBtn = document.getElementById("clearBtn");
const resultBox = document.getElementById("result");
const labelEl = document.getElementById("label");
const confidenceEl = document.getElementById("confidence");
const barFill = document.getElementById("barFill");
const errorEl = document.getElementById("error");

function hideResults() {
  resultBox.classList.add("hidden");
  errorEl.classList.add("hidden");
}

async function checkMessage() {
  hideResults();
  const message = messageBox.value.trim();
  if (!message) {
    errorEl.textContent = "Please enter an SMS message.";
    errorEl.classList.remove("hidden");
    return;
  }

  checkBtn.disabled = true;
  checkBtn.textContent = "Checking...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: message }),
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Something went wrong.");
    }

    const isSpam = data.prediction === "SPAM";
    labelEl.textContent = data.prediction;
    confidenceEl.textContent = data.confidence + "%";
    barFill.style.width = data.confidence + "%";
    resultBox.className = "result " + (isSpam ? "spam" : "safe");
  } catch (err) {
    errorEl.textContent = err.message;
    errorEl.classList.remove("hidden");
  } finally {
    checkBtn.disabled = false;
    checkBtn.textContent = "Check Message";
  }
}

function resetForm() {
  messageBox.value = "";
  barFill.style.width = "0";
  hideResults();
  messageBox.focus();
}

checkBtn.addEventListener("click", checkMessage);
clearBtn.addEventListener("click", resetForm);
