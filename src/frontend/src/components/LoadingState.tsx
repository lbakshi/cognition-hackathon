'use client'

import { Brain, Cpu, Zap } from 'lucide-react'

interface LoadingStateProps {
  message?: string
  progress?: number
}

export default function LoadingState({ message = "Running experiment...", progress }: LoadingStateProps) {
  const steps = [
    { icon: Brain, label: "Planning experiment", delay: 0 },
    { icon: Cpu, label: "Generating models", delay: 200 },
    { icon: Zap, label: "Training & evaluation", delay: 400 },
  ]

  return (
    <div className="w-full max-w-2xl mx-auto">
      <div className="glass-card rounded-2xl p-12 text-center">
        <div className="mb-8">
          <div className="relative mx-auto w-24 h-24 mb-6">
            <div className="absolute inset-0 rounded-full border-4 border-blue-200 dark:border-blue-800"></div>
            <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-blue-600 animate-spin"></div>
            <div className="absolute inset-2 rounded-full border-4 border-transparent border-r-purple-600 animate-spin" style={{ animationDirection: 'reverse', animationDuration: '1.5s' }}></div>
          </div>
          
          <h2 className="text-2xl font-bold gradient-text mb-2">
            AI Research in Progress
          </h2>
          <p className="text-gray-600 dark:text-gray-400">
            {message}
          </p>
        </div>

        {progress !== undefined && (
          <div className="mb-8">
            <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
              <div 
                className="bg-gradient-to-r from-blue-600 to-purple-600 h-2 rounded-full transition-all duration-500"
                style={{ width: `${progress}%` }}
              ></div>
            </div>
            <p className="text-sm text-gray-500 mt-2">{progress}% complete</p>
          </div>
        )}

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
          {steps.map(({ icon: Icon, label, delay }, index) => (
            <div
              key={index}
              className={`flex flex-col items-center gap-3 animate-pulse-slow`}
              style={{ animationDelay: `${delay}ms` }}
            >
              <div className="p-4 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-full">
                <Icon className="w-6 h-6 text-blue-600" />
              </div>
              <span className="text-sm text-gray-600 dark:text-gray-400">
                {label}
              </span>
            </div>
          ))}
        </div>

        <div className="mt-8 text-xs text-gray-500 dark:text-gray-400">
          This may take a few minutes depending on the complexity of your experiment
        </div>
      </div>
    </div>
  )
}
