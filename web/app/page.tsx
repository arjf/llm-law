"use client";

import { useState } from "react";
import ChatInterface from "@/components/ChatInterface";
import CJPEScore from "@/components/CJPEScore";
import Navbar from "@/components/Navbar";

export default function Home() {
  const [cjpeScore, setCjpeScore] = useState<number | undefined>(undefined);

  const handleScoreUpdate = (score: number) => {
    setCjpeScore(score);
  };

  return (
    <>
      <Navbar />
      <main className="min-h-screen bg-background-light dark:bg-background-dark pt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* Main Chat Area */}
            <div className="lg:col-span-3 h-[calc(100vh-140px)]">
              <ChatInterface onScoreUpdate={handleScoreUpdate} />
            </div>

            {/* Sidebar */}
            <div className="lg:col-span-1">
              <div className="sticky top-24 space-y-6">
                {/* CJPE Score */}
                <CJPEScore score={cjpeScore} loading={false} />

                {/* Quick Stats */}
                <div className="bg-card-light dark:bg-card-dark rounded-xl shadow-2xl p-6 border border-border-light dark:border-border-dark">
                  <h3 className="text-sm font-bold text-gray-900 dark:text-white mb-4">
                    System Stats
                  </h3>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-gray-600 dark:text-gray-400">
                        Model
                      </span>
                      <span className="text-xs font-semibold text-gray-900 dark:text-white">
                        Mistral-7B
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-gray-600 dark:text-gray-400">
                        Retrieval
                      </span>
                      <span className="text-xs font-semibold text-gray-900 dark:text-white">
                        InLegalBERT
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-gray-600 dark:text-gray-400">
                        Status
                      </span>
                      <div className="flex items-center gap-1.5">
                        <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                        <span className="text-xs font-semibold text-green-500">
                          Online
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Features */}
                <div className="bg-card-light dark:bg-card-dark rounded-xl shadow-2xl p-6 border border-border-light dark:border-border-dark">
                  <h3 className="text-sm font-bold text-gray-900 dark:text-white mb-4">
                    Features
                  </h3>
                  <div className="space-y-3">
                    {[
                      {
                        icon: "M13 10V3L4 14h7v7l9-11h-7z",
                        text: "Instant Answers",
                      },
                      {
                        icon: "M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z",
                        text: "Document Analysis",
                      },
                      {
                        icon: "M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2",
                        text: "Case Precedents",
                      },
                      {
                        icon: "M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4",
                        text: "Smart Retrieval",
                      },
                    ].map((feature, idx) => (
                      <div
                        key={idx}
                        className="flex items-center gap-3 text-gray-700 dark:text-gray-300"
                      >
                        <svg
                          className="w-4 h-4 text-primary-500 flex-shrink-0"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d={feature.icon}
                          />
                        </svg>
                        <span className="text-xs">{feature.text}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </>
  );
}
