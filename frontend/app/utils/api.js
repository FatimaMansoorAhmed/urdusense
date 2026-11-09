const API_BASE_URL = "http://127.0.0.1:8000";

export async function analyzeSentiment(text) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/predict`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      throw new Error("Failed to fetch sentiment analysis.");
    }

    return await response.json();
  } catch (error) {
    console.error("API Error:", error);
    throw error;
  }
}