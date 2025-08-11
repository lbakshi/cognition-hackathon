'use client'

import { useState } from 'react'
import ExperimentForm from '../components/ExperimentForm'
import ResultsDashboard from '../components/ResultsDashboard'
import LoadingState from '../components/LoadingState'
import { ExperimentResult, ExperimentStatus } from '../types/experiment'
import { demoExperimentResults } from '../lib/demo-data'
import { simulateAPICall } from '../lib/utils'

export default function Home() {
  const [experimentStatus, setExperimentStatus] = useState<ExperimentStatus>({
    status: 'idle'
  })
  const [results, setResults] = useState<ExperimentResult | null>(null)
  const [experimentPrompt, setExperimentPrompt] = useState('')

  const handleExperimentSubmit = async (prompt: string) => {
    try {
      setExperimentPrompt(prompt)
      setExperimentStatus({ status: 'running', progress: 0 })
      
      // Simulate progress updates
      const progressSteps = [
        { progress: 20, message: 'Planning experiment architecture...' },
        { progress: 40, message: 'Generating PyTorch models...' },
        { progress: 60, message: 'Training candidate model...' },
        { progress: 80, message: 'Training baseline model...' },
        { progress: 90, message: 'Computing metrics...' },
        { progress: 100, message: 'Experiment completed!' }
      ]

      for (const step of progressSteps) {
        await new Promise(resolve => setTimeout(resolve, 800))
        setExperimentStatus({ 
          status: 'running', 
          progress: step.progress,
          message: step.message
        })
      }

      // TODO: Replace with actual API call to Railway service
      // const response = await fetch('/api/experiment', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify({ prompt })
      // })
      // const results = await response.json()

      // For now, use demo data
      const experimentResults = await simulateAPICall(demoExperimentResults, 500)
      
      setResults(experimentResults)
      setExperimentStatus({ status: 'completed' })
      
    } catch (error) {
      console.error('Experiment failed:', error)
      setExperimentStatus({ 
        status: 'error', 
        message: 'Experiment failed. Please try again.' 
      })
    }
  }

  const handleReset = () => {
    setExperimentStatus({ status: 'idle' })
    setResults(null)
    setExperimentPrompt('')
  }

  return (
    <main className="min-h-screen py-8 px-4">
      <div className="container mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl md:text-6xl font-bold gradient-text mb-4">
            Cognition Research
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
            AI-powered research platform that designs, runs, and analyzes 
            machine learning experiments automatically
          </p>
        </div>

        {/* Main Content */}
        <div className="space-y-12">
          {experimentStatus.status === 'idle' && (
            <ExperimentForm 
              onSubmit={handleExperimentSubmit}
              isLoading={false}
            />
          )}

          {experimentStatus.status === 'running' && (
            <LoadingState 
              message={experimentStatus.message}
              progress={experimentStatus.progress}
            />
          )}

          {experimentStatus.status === 'completed' && results && (
            <div className="space-y-8">
              <ResultsDashboard 
                data={results}
                experimentPrompt={experimentPrompt}
              />
              
              {/* Reset Button */}
              <div className="text-center">
                <button
                  onClick={handleReset}
                  className="px-8 py-3 bg-gradient-to-r from-gray-600 to-gray-700 text-white font-medium rounded-xl hover:from-gray-700 hover:to-gray-800 transition-all duration-200 shadow-lg"
                >
                  Run New Experiment
                </button>
              </div>
            </div>
          )}

          {experimentStatus.status === 'error' && (
            <div className="text-center">
              <div className="glass-card rounded-2xl p-8 max-w-md mx-auto">
                <div className="text-red-500 mb-4">
                  <svg className="w-16 h-16 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <h3 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
                  Experiment Failed
                </h3>
                <p className="text-gray-600 dark:text-gray-400 mb-6">
                  {experimentStatus.message}
                </p>
                <button
                  onClick={handleReset}
                  className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  Try Again
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="text-center mt-16 text-sm text-gray-500 dark:text-gray-400">
          <p>Powered by Next.js 15, Claude 4.1, and Modal serverless infrastructure</p>
        </div>
      </div>
    </main>
  )
}
