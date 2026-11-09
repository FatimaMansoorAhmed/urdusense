"use client";
import { analyzeSentiment } from "@/app/utils/api";
import { useState } from "react";
import { Sparkles, Loader2, ThumbsUp, ThumbsDown, MinusCircle } from "lucide-react";

interface PredictionResult {
  text: string;
  prediction: string;
  confidence: number;
  probabilities: {
    [key: string]: number;
  };
}

export default function Home() {
  const [text, setText] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await analyzeSentiment(text);
      setResult(data);
    } catch (err) {
      setError("Backend server se connect karne mein masla aa raha hai. Please check karein ke FastAPI server chal raha hai.");
    } finally {
      setLoading(false);
    }
  };

  const getSentimentBadge = (prediction: string) => {
    switch (prediction) {
      case "Positive":
        return (
          <span className="flex items-center gap-1.5 text-emerald-600 bg-emerald-50 px-3 py-1 rounded-full font-medium text-sm border border-emerald-200">
            <ThumbsUp className="w-4 h-4" /> Positive
          </span>
        );
      case "Negative":
        return (
          <span className="flex items-center gap-1.5 text-rose-600 bg-rose-50 px-3 py-1 rounded-full font-medium text-sm border border-rose-200">
            <ThumbsDown className="w-4 h-4" /> Negative
          </span>
        );
      default:
        return (
          <span className="flex items-center gap-1.5 text-amber-600 bg-amber-50 px-3 py-1 rounded-full font-medium text-sm border border-amber-200">
            <MinusCircle className="w-4 h-4" /> Neutral
          </span>
        );
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-50 via-indigo-50/30 to-slate-100 flex flex-col items-center justify-center p-4 sm:p-8">
      <div className="w-full max-w-xl bg-white rounded-2xl shadow-xl border border-slate-100 p-6 sm:p-8 transition-all">
        
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center bg-indigo-100 text-indigo-600 p-3 rounded-xl mb-3 shadow-sm">
            <Sparkles className="w-6 h-6" />
          </div>
          <h1 className="text-2xl sm:text-3xl font-bold text-slate-800 tracking-tight">
            UrduSense Sentiment Analyzer
          </h1>
          <p className="text-slate-500 text-sm mt-1">
            Analyze emotions and sentiments in Roman Urdu text instantly using AI.
          </p>
        </div>

        {/* Form */}
        <form onSubmit={handlePredict} className="space-y-4">
          <div>
            <label htmlFor="text-input" className="block text-xs font-semibold text-slate-600 uppercase tracking-wider mb-2">
              Enter Roman Urdu Text
            </label>
            <textarea
              id="text-input"
              rows={4}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Misal ke tor par: Yeh movie waqai boht achi thi, maza aa gaya!"
              className="w-full rounded-xl border border-slate-200 p-4 text-slate-700 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all shadow-sm text-sm resize-none"
            />
          </div>

          <button
            type="submit"
            disabled={loading || !text.trim()}
            className="w-full bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-300 text-white font-medium py-3 px-4 rounded-xl shadow-lg shadow-indigo-200 transition-all flex items-center justify-center gap-2 cursor-pointer disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" /> Analyzing...
              </>
            ) : (
              "Analyze Sentiment"
            )}
          </button>
        </form>

        {/* Error message */}
        {error && (
          <div className="mt-6 p-4 bg-rose-50 border border-rose-200 text-rose-700 rounded-xl text-sm">
            {error}
          </div>
        )}

        {/* Result Card */}
        {result && (
          <div className="mt-8 pt-6 border-t border-slate-100 space-y-5 animate-in fade-in slide-in-from-bottom-2 duration-300">
            <div className="flex items-center justify-between">
              <span className="text-sm font-semibold text-slate-500 uppercase tracking-wider">Result</span>
              {getSentimentBadge(result.prediction)}
            </div>

            <div className="bg-slate-50 rounded-xl p-4 border border-slate-100">
              <div className="flex justify-between text-sm mb-2">
                <span className="text-slate-500">Confidence Score</span>
                <span className="font-semibold text-slate-700">{(result.confidence * 100).toFixed(1)}%</span>
              </div>
              <div className="w-full bg-slate-200 h-2 rounded-full overflow-hidden">
                <div
                  className="bg-indigo-600 h-full rounded-full transition-all duration-500"
                  style={{ width: `${result.confidence * 100}%` }}
                />
              </div>
            </div>

            {/* Probabilities Breakdown */}
            <div className="space-y-2">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Class Probabilities</span>
              <div className="grid grid-cols-3 gap-2">
                {Object.entries(result.probabilities).map(([label, prob]) => (
                  <div key={label} className="bg-slate-50 border border-slate-100 p-3 rounded-xl text-center">
                    <div className="text-xs text-slate-500 mb-1 font-medium">{label}</div>
                    <div className="text-sm font-bold text-slate-700">{(Number(prob) * 100).toFixed(1)}%</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

      </div>
    </main>
  );
}