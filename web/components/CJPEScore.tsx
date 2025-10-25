"use client";

import { useState, useEffect } from "react";

interface CJPEScoreProps {
  score?: number;
  loading?: boolean;
}

export default function CJPEScore({ score, loading = false }: CJPEScoreProps) {
  const [displayScore, setDisplayScore] = useState<number | null>(null);

  useEffect(() => {
    if (score !== undefined) {
      setDisplayScore(score);
    }
  }, [score]);

  const getScoreColor = (value: number) => {
    if (value >= 0.8) return "text-green-500";
    if (value >= 0.6) return "text-primary-500";
    return "text-red-500";
  };

  const getScoreLabel = (value: number) => {
    if (value >= 0.8) return "Highly Favorable";
    if (value >= 0.6) return "Likely Favorable";
    if (value >= 0.4) return "Uncertain";
    return "Likely Unfavorable";
  };

  const getGradientColor = (value: number) => {
    if (value >= 0.8) return "from-green-500 to-green-600";
    if (value >= 0.6) return "from-primary-500 to-primary-600";
    return "from-red-500 to-red-600";
  };

  return (
    <div className="bg-card-light dark:bg-card-dark rounded-xl shadow-2xl p-6 border border-border-light dark:border-border-dark">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-bold text-gray-900 dark:text-white">
          Court Judgment Prediction
        </h2>
        <div className="w-2 h-2 rounded-full bg-primary-500 animate-pulse" />
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-32">
          <div className="relative">
            <div className="absolute inset-0 bg-primary-500 blur-xl opacity-50" />
            <div className="relative animate-spin rounded-full h-12 w-12 border-4 border-border-dark border-t-primary-500"></div>
          </div>
        </div>
      ) : displayScore !== null ? (
        <div className="space-y-4 animate-fadeIn">
          <div className="flex items-end justify-between">
            <div>
              <span
                className={`text-5xl font-bold ${getScoreColor(displayScore)}`}
              >
                {(displayScore * 100).toFixed(1)}
              </span>
              <span className="text-2xl text-gray-600 dark:text-gray-400">
                %
              </span>
            </div>
            <div className="text-right">
              <span
                className={`text-sm font-semibold ${getScoreColor(displayScore)}`}
              >
                {getScoreLabel(displayScore)}
              </span>
            </div>
          </div>

          <div className="relative w-full h-3 bg-surface-light dark:bg-surface-dark rounded-full overflow-hidden">
            <div
              className={`h-full bg-gradient-to-r ${getGradientColor(displayScore)} transition-all duration-1000 ease-out rounded-full shadow-lg`}
              style={{ width: `${displayScore * 100}%` }}
            >
              <div className="absolute inset-0 bg-white opacity-20 animate-pulse" />
            </div>
          </div>

          <div className="pt-2 border-t border-border-light dark:border-border-dark">
            <p className="text-xs text-gray-600 dark:text-gray-400">
              Court Judgment Prediction and Explanation (CJPE)
            </p>
          </div>
        </div>
      ) : (
        <div className="text-center py-10">
          <div className="relative mb-4 inline-block">
            <div className="absolute inset-0 bg-primary-500 blur-2xl opacity-20" />
            <div className="relative w-16 h-16 rounded-full border-4 border-border-light dark:border-border-dark flex items-center justify-center">
              <svg
                className="w-8 h-8 text-gray-400 dark:text-gray-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                />
              </svg>
            </div>
          </div>
          <p className="text-gray-600 dark:text-gray-400 font-medium">
            No active query
          </p>
          <p className="text-xs text-gray-500 dark:text-gray-500 mt-2">
            Send a message to see metrics
          </p>
        </div>
      )}
    </div>
  );
}
