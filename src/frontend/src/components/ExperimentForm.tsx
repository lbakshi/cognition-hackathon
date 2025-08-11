'use client'

import { useState } from 'react'
import { Send, Sparkles } from 'lucide-react'
import { cn } from '../lib/utils'

interface ExperimentFormProps {
  onSubmit: (prompt: string) => void
  isLoading: boolean
}

export default function ExperimentForm({ onSubmit, isLoading }: ExperimentFormProps) {
  const [prompt, setPrompt] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (prompt.trim() && !isLoading) {
      onSubmit(prompt.trim())
    }
  }

  const examplePrompts = [
    "Compare CNN with GELU activation vs ReLU on image classification",
    "Test transformer vs LSTM on text summarization task",
    "Evaluate ResNet with different normalization techniques"
  ]

  return (
    <div className="w-full max-w-4xl mx-auto">
      <div className="glass-card rounded-2xl p-8 animate-fade-in">
        <div className="text-center mb-8">
          <div className="flex justify-center mb-4">
            <div className="p-3 bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-full">
              <Sparkles className="w-8 h-8 text-blue-600" />
            </div>
          </div>
          <h2 className="text-2xl font-bold gradient-text mb-2">
            Research Experiment Generator
          </h2>
          <p className="text-muted-foreground">
            Describe your research idea and we'll run the experiments for you
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="relative">
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Describe your research hypothesis or experiment idea..."
              className={cn(
                "w-full min-h-[120px] px-6 py-4 rounded-xl",
                "bg-white/50 dark:bg-black/50 backdrop-blur-sm",
                "border border-gray-200/50 dark:border-gray-700/50",
                "focus:ring-2 focus:ring-blue-500/50 focus:border-transparent",
                "placeholder:text-gray-400 text-gray-900 dark:text-gray-100",
                "transition-all duration-200",
                "resize-none"
              )}
              disabled={isLoading}
            />
            <div className="absolute bottom-4 right-4 text-xs text-gray-400">
              {prompt.length}/500
            </div>
          </div>

          <button
            type="submit"
            disabled={!prompt.trim() || isLoading}
            className={cn(
              "w-full flex items-center justify-center gap-3 px-8 py-4",
              "bg-gradient-to-r from-blue-600 to-purple-600",
              "text-white font-medium rounded-xl",
              "hover:from-blue-700 hover:to-purple-700",
              "disabled:opacity-50 disabled:cursor-not-allowed",
              "transition-all duration-200",
              "shadow-lg shadow-blue-500/25"
            )}
          >
            {isLoading ? (
              <>
                <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white" />
                <span>Running Experiment...</span>
              </>
            ) : (
              <>
                <Send className="w-5 h-5" />
                <span>Start Experiment</span>
              </>
            )}
          </button>
        </form>

        {!isLoading && (
          <div className="mt-8 animate-slide-up">
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
              Need inspiration? Try these examples:
            </p>
            <div className="space-y-2">
              {examplePrompts.map((example, index) => (
                <button
                  key={index}
                  onClick={() => setPrompt(example)}
                  className={cn(
                    "w-full text-left px-4 py-3 rounded-lg",
                    "bg-gray-50/50 dark:bg-gray-800/50",
                    "border border-gray-200/30 dark:border-gray-700/30",
                    "hover:bg-gray-100/50 dark:hover:bg-gray-700/50",
                    "text-sm text-gray-700 dark:text-gray-300",
                    "transition-all duration-200"
                  )}
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
