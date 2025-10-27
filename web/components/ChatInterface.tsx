"use client";

import { useState, useRef, useEffect } from "react";

interface UploadedDocument {
  id: string;
  name: string;
  size: number;
  type: string;
  file: File;
  uploadedAt: Date;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
  documentIds?: string[];
}

interface ChatInterfaceProps {
  onScoreUpdate?: (score: number) => void;
  onCJPELoading?: (loading: boolean) => void;
}

export default function ChatInterface({
  onScoreUpdate,
  onCJPELoading,
}: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [documents, setDocuments] = useState<UploadedDocument[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    setUploadedFiles((prev) => [...prev, ...files]);
  };

  const removeFile = (index: number) => {
    setUploadedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleDocumentClick = (doc: UploadedDocument) => {
    // Create a download link for the document
    const url = URL.createObjectURL(doc.file);
    const a = document.createElement("a");
    a.href = url;
    a.download = doc.name;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const removeDocument = (docId: string) => {
    setDocuments((prev) => prev.filter((doc) => doc.id !== docId));
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if ((!input.trim() && uploadedFiles.length === 0) || isLoading) return;

    // Create document objects from uploaded files
    const newDocs: UploadedDocument[] = uploadedFiles.map((file) => ({
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      name: file.name,
      size: file.size,
      type: file.type,
      file: file,
      uploadedAt: new Date(),
    }));

    const docIds = newDocs.map((doc) => doc.id);

    // Add documents to persistent state
    if (newDocs.length > 0) {
      setDocuments((prev) => [...prev, ...newDocs]);
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim() || `Analyzing ${uploadedFiles.length} document(s)`,
      timestamp: new Date(),
      documentIds: docIds.length > 0 ? docIds : undefined,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    // Create assistant message for streaming
    const assistantMessageId = (Date.now() + 1).toString();
    const assistantMessage: Message = {
      id: assistantMessageId,
      role: "assistant",
      content: "",
      timestamp: new Date(),
      isStreaming: true,
    };

    setMessages((prev) => [...prev, assistantMessage]);

    try {
      // Prepare conversation history
      const conversationHistory = messages.map((msg) => ({
        role: msg.role,
        content: msg.content,
        documentIds: msg.documentIds,
      }));

      // Add current user message to history
      conversationHistory.push({
        role: userMessage.role,
        content: userMessage.content,
        documentIds: userMessage.documentIds,
      });

      // Prepare document metadata
      const documentMetadata = documents.map((doc) => ({
        id: doc.id,
        name: doc.name,
        size: doc.size,
        type: doc.type,
      }));

      // Call CJPE API in parallel (don't await)
      if (onCJPELoading) onCJPELoading(true);

      fetch("/api/cjpe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: userMessage.content }),
      })
        .then((res) => res.json())
        .then((cjpeData) => {
          // Update CJPE score immediately when available
          if (onScoreUpdate && cjpeData.probabilities) {
            onScoreUpdate(cjpeData.probabilities.favorable);
          }
          if (onCJPELoading) onCJPELoading(false);
        })
        .catch((err) => {
          console.error("CJPE prediction error:", err);
          if (onCJPELoading) onCJPELoading(false);
        });

      // Call Gemma streaming API
      const response = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: userMessage.content,
          files: uploadedFiles.map((f) => f.name),
          conversation_history: conversationHistory,
          documents: documentMetadata,
        }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      // Read streaming response
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error("No response body");
      }

      let fullText = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });

        // Check if this is metadata
        if (chunk.includes("__METADATA__")) {
          const parts = chunk.split("__METADATA__");
          const textPart = parts[0];
          const metadataPart = parts[1];

          if (textPart) {
            fullText += textPart;
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMessageId
                  ? { ...msg, content: fullText }
                  : msg,
              ),
            );
          }

          // Parse and handle metadata (legacy support)
          try {
            const metadata = JSON.parse(metadataPart.trim());
            if (onScoreUpdate && metadata.cjpe_score) {
              onScoreUpdate(metadata.cjpe_score);
            }
          } catch (e) {
            console.error("Failed to parse metadata:", e);
          }
        } else {
          fullText += chunk;
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMessageId
                ? { ...msg, content: fullText }
                : msg,
            ),
          );
        }
      }

      // Mark streaming as complete
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId ? { ...msg, isStreaming: false } : msg,
        ),
      );

      setIsLoading(false);
      setUploadedFiles([]); // Clear temporary upload state
    } catch (error) {
      console.error("Error querying RAG system:", error);

      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === assistantMessageId
            ? {
                ...msg,
                content: "Error connecting to the system. Please try again.",
                isStreaming: false,
              }
            : msg,
        ),
      );
      setIsLoading(false);
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  return (
    <div className="flex flex-col h-full bg-card-light dark:bg-card-dark rounded-xl shadow-2xl border border-border-light dark:border-border-dark overflow-hidden">
      <div className="border-b border-border-light dark:border-border-dark px-6 py-4 bg-surface-light dark:bg-surface-dark">
        <h2 className="text-xl font-bold text-gray-900 dark:text-white">
          Legal Assistant
        </h2>
        <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
          Ask questions about Indian employment and workplace law
        </p>
      </div>

      {/* Persistent Documents Area */}
      {documents.length > 0 && (
        <div className="border-b border-border-light dark:border-border-dark px-6 py-3 bg-surface-light dark:bg-surface-dark">
          <div className="flex items-center gap-2 mb-2">
            <svg
              className="w-4 h-4 text-primary-500"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
              />
            </svg>
            <h3 className="text-xs font-semibold text-gray-700 dark:text-gray-300">
              Uploaded Documents ({documents.length})
            </h3>
          </div>
          <div className="flex flex-wrap gap-2">
            {documents.map((doc) => (
              <div
                key={doc.id}
                className="group relative flex items-center gap-2 px-3 py-2 bg-card-light dark:bg-card-dark border border-border-light dark:border-border-dark rounded-lg hover:border-primary-500 dark:hover:border-primary-500 transition-all duration-200 cursor-pointer"
                onClick={() => handleDocumentClick(doc)}
                title={`Click to download ${doc.name}`}
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
                    d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"
                  />
                </svg>
                <div className="flex flex-col min-w-0">
                  <span className="text-xs font-medium text-gray-900 dark:text-gray-100 truncate max-w-[200px]">
                    {doc.name}
                  </span>
                  <span className="text-[10px] text-gray-500 dark:text-gray-500">
                    {formatFileSize(doc.size)} •{" "}
                    {doc.uploadedAt.toLocaleDateString()}
                  </span>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    removeDocument(doc.id);
                  }}
                  className="ml-2 text-gray-400 hover:text-red-500 transition-colors opacity-0 group-hover:opacity-100"
                  title="Remove document"
                >
                  <svg
                    className="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                </button>
                <div className="absolute inset-0 rounded-lg ring-2 ring-primary-500 opacity-0 group-hover:opacity-20 transition-opacity pointer-events-none" />
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center animate-fadeIn">
            <div className="relative mb-6">
              <div className="absolute inset-0 bg-primary-500 blur-3xl opacity-20" />
              <div className="relative w-20 h-20 rounded-full bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center">
                <svg
                  className="w-10 h-10 text-white"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
                  />
                </svg>
              </div>
            </div>
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-3">
              Start Your Legal Consultation
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 max-w-md mb-8">
              Get instant answers powered by advanced AI trained on Indian legal
              documents. Upload documents or ask questions directly.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 w-full max-w-2xl">
              {[
                {
                  icon: "M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z",
                  title: "Upload Documents",
                  desc: "Analyze contracts and agreements",
                },
                {
                  icon: "M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z",
                  title: "Ask Questions",
                  desc: "Get instant legal guidance",
                },
                {
                  icon: "M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01",
                  title: "Case Analysis",
                  desc: "Review relevant precedents",
                },
              ].map((feature, idx) => (
                <div
                  key={idx}
                  className="p-4 rounded-lg bg-surface-light dark:bg-surface-dark border border-border-light dark:border-border-dark hover:border-primary-500 transition-all duration-300 cursor-pointer group"
                >
                  <svg
                    className="w-8 h-8 text-primary-500 mb-2 group-hover:scale-110 transition-transform"
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
                  <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-1">
                    {feature.title}
                  </h4>
                  <p className="text-xs text-gray-600 dark:text-gray-400">
                    {feature.desc}
                  </p>
                </div>
              ))}
            </div>
          </div>
        ) : (
          <>
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${
                  message.role === "user" ? "justify-end" : "justify-start"
                } animate-fadeIn`}
              >
                <div
                  className={`max-w-3xl rounded-2xl px-6 py-4 ${
                    message.role === "user"
                      ? "bg-gradient-to-br from-primary-500 to-primary-600 text-white shadow-lg"
                      : "bg-surface-light dark:bg-surface-dark text-gray-900 dark:text-gray-100 border border-border-light dark:border-border-dark"
                  }`}
                >
                  <div className="flex items-start gap-3">
                    {message.role === "assistant" && (
                      <div className="flex-shrink-0 mt-1">
                        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center shadow-lg">
                          <svg
                            className="w-5 h-5 text-white"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                          >
                            <path
                              strokeLinecap="round"
                              strokeLinejoin="round"
                              strokeWidth={2}
                              d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                            />
                          </svg>
                        </div>
                      </div>
                    )}
                    <div className="flex-1 min-w-0">
                      <p className="text-sm whitespace-pre-wrap break-words leading-relaxed">
                        {message.content}
                        {message.isStreaming && (
                          <span className="inline-block w-2 h-4 ml-1 bg-primary-500 animate-pulse" />
                        )}
                      </p>
                      {message.documentIds &&
                        message.documentIds.length > 0 && (
                          <div className="flex flex-wrap gap-2 mt-3">
                            {message.documentIds.map((docId) => {
                              const doc = documents.find((d) => d.id === docId);
                              if (!doc) return null;
                              return (
                                <button
                                  key={docId}
                                  onClick={() => handleDocumentClick(doc)}
                                  className="flex items-center gap-1.5 px-2 py-1 bg-white/10 hover:bg-white/20 dark:bg-black/10 dark:hover:bg-black/20 rounded text-xs transition-colors"
                                  title={`Click to download ${doc.name}`}
                                >
                                  <svg
                                    className="w-3 h-3"
                                    fill="none"
                                    stroke="currentColor"
                                    viewBox="0 0 24 24"
                                  >
                                    <path
                                      strokeLinecap="round"
                                      strokeLinejoin="round"
                                      strokeWidth={2}
                                      d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13"
                                    />
                                  </svg>
                                  <span className="truncate max-w-[120px]">
                                    {doc.name}
                                  </span>
                                </button>
                              );
                            })}
                          </div>
                        )}
                      <p
                        className={`text-xs mt-2 ${
                          message.role === "user"
                            ? "text-primary-100"
                            : "text-gray-500 dark:text-gray-500"
                        }`}
                      >
                        {formatTime(message.timestamp)}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </>
        )}
      </div>

      <div className="border-t border-border-light dark:border-border-dark p-4 bg-surface-light dark:bg-surface-dark">
        {uploadedFiles.length > 0 && (
          <div className="mb-3 flex flex-wrap gap-2">
            {uploadedFiles.map((file, index) => (
              <div
                key={index}
                className="flex items-center gap-2 px-3 py-1.5 bg-card-light dark:bg-card-dark border border-border-light dark:border-border-dark rounded-lg text-sm"
              >
                <svg
                  className="w-4 h-4 text-primary-500"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                  />
                </svg>
                <span className="text-gray-700 dark:text-gray-300 truncate max-w-xs">
                  {file.name}
                </span>
                <button
                  onClick={() => removeFile(index)}
                  className="text-gray-400 hover:text-red-500 transition-colors"
                >
                  <svg
                    className="w-4 h-4"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M6 18L18 6M6 6l12 12"
                    />
                  </svg>
                </button>
              </div>
            ))}
          </div>
        )}

        <form onSubmit={handleSubmit} className="flex gap-2">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileUpload}
            multiple
            accept=".pdf,.doc,.docx,.txt"
            className="hidden"
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={isLoading}
            className="p-3 rounded-xl bg-card-light dark:bg-card-dark border border-border-light dark:border-border-dark hover:border-primary-500 dark:hover:border-primary-500 text-gray-700 dark:text-gray-300 hover:text-primary-500 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-primary-500"
            title="Upload document"
          >
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M15.172 7l-6.586 6.586a2 2 0 102.828 2.828l6.414-6.586a4 4 0 00-5.656-5.656l-6.415 6.585a6 6 0 108.486 8.486L20.5 13"
              />
            </svg>
          </button>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your legal question..."
            disabled={isLoading}
            className="flex-1 px-4 py-3 bg-card-light dark:bg-card-dark border border-border-light dark:border-border-dark rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
          />
          <button
            type="submit"
            disabled={
              isLoading || (!input.trim() && uploadedFiles.length === 0)
            }
            className="px-6 py-3 bg-gradient-to-r from-primary-500 to-primary-600 text-white rounded-xl hover:from-primary-600 hover:to-primary-700 focus:outline-none focus:ring-2 focus:ring-primary-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 font-medium shadow-lg hover:shadow-xl disabled:shadow-none flex items-center gap-2"
          >
            {isLoading ? (
              <>
                <svg
                  className="animate-spin h-5 w-5"
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                >
                  <circle
                    className="opacity-25"
                    cx="12"
                    cy="12"
                    r="10"
                    stroke="currentColor"
                    strokeWidth="4"
                  ></circle>
                  <path
                    className="opacity-75"
                    fill="currentColor"
                    d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                  ></path>
                </svg>
                <span>Thinking...</span>
              </>
            ) : (
              <>
                <span>Send</span>
                <svg
                  className="w-5 h-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M14 5l7 7m0 0l-7 7m7-7H3"
                  />
                </svg>
              </>
            )}
          </button>
        </form>
      </div>
    </div>
  );
}
