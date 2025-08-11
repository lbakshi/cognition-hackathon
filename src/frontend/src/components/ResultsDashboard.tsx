'use client'

import { BarChart3, TrendingUp, Target, Award } from 'lucide-react'
import MetricChart from './MetricChart'
import { ExperimentResult } from '../types/experiment'
import { experimentLabels, metricColors } from '../lib/demo-data'
import { formatMetricValue } from '../lib/utils'

interface ResultsDashboardProps {
  data: ExperimentResult
  experimentPrompt: string
}

export default function ResultsDashboard({ data, experimentPrompt }: ResultsDashboardProps) {
  const experimentIds = Object.keys(data)
  
  // Safe accessor functions for experiment data
  const getExperimentLabel = (key: string): string => {
    return experimentLabels[key] || key
  }

  const getMetricColor = (key: string, fallback: string = '#6B7280'): string => {
    return metricColors[key] || fallback
  }
  
  // Calculate final metric values for summary
  const getFinalValue = (expId: string, metric: 'accuracy' | 'precision' | 'recall' | 'f1-score') => {
    const metricData = data[expId][metric]
    return metricData.Y[metricData.Y.length - 1] || 0
  }

  const metrics = [
    { key: 'accuracy' as const, title: 'Accuracy', icon: Target },
    { key: 'precision' as const, title: 'Precision', icon: BarChart3 },
    { key: 'recall' as const, title: 'Recall', icon: TrendingUp },
    { key: 'f1-score' as const, title: 'F1-Score', icon: Award },
  ]

  return (
    <div className="w-full max-w-7xl mx-auto space-y-8 animate-fade-in">
      {/* Header */}
      <div className="glass-card rounded-2xl p-8">
        <h2 className="text-2xl font-bold gradient-text mb-4">
          Experiment Results
        </h2>
        <div className="p-4 bg-blue-50/50 dark:bg-blue-900/20 rounded-xl border border-blue-200/30 dark:border-blue-700/30">
          <p className="text-sm text-blue-800 dark:text-blue-200 font-medium mb-1">
            Research Query:
          </p>
          <p className="text-blue-900 dark:text-blue-100 italic">
            "{experimentPrompt}"
          </p>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {metrics.map(({ key, title, icon: Icon }) => (
          <div key={key} className="glass-card rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <Icon className="w-8 h-8 text-blue-600" />
              <span className="text-sm font-medium text-gray-600 dark:text-gray-400">
                {title}
              </span>
            </div>
            <div className="space-y-3">
              {experimentIds.map((expId) => (
                <div key={expId} className="flex justify-between items-center">
                  <div className="flex items-center gap-2">
                    <div 
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: getMetricColor(expId) }}
                    />
                    <span className="text-xs text-gray-600 dark:text-gray-400 truncate max-w-[100px]">
                      {getExperimentLabel(expId).split('(')[0] || expId}
                    </span>
                  </div>
                  <span className="font-bold text-gray-900 dark:text-gray-100">
                    {formatMetricValue(getFinalValue(expId, key))}
                  </span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {metrics.map(({ key, title }) => (
          <div key={key} className="animate-slide-up">
            <MetricChart
              data={data}
              metric={key}
              title={title}
            />
          </div>
        ))}
      </div>

      {/* Model Comparison */}
      <div className="glass-card rounded-2xl p-8">
        <h3 className="text-xl font-bold gradient-text mb-6">
          Model Comparison Summary
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {experimentIds.map((expId) => (
            <div key={expId} className="space-y-4">
              <div className="flex items-center gap-3">
                <div 
                  className="w-4 h-4 rounded-full"
                  style={{ backgroundColor: getMetricColor(expId) }}
                />
                <h4 className="font-semibold text-gray-900 dark:text-gray-100">
                  {getExperimentLabel(expId)}
                </h4>
              </div>
              <div className="grid grid-cols-2 gap-4">
                {metrics.map(({ key, title }) => (
                  <div key={key} className="text-center p-3 bg-gray-50/50 dark:bg-gray-800/50 rounded-lg">
                    <div className="text-lg font-bold text-gray-900 dark:text-gray-100">
                      {formatMetricValue(getFinalValue(expId, key))}
                    </div>
                    <div className="text-xs text-gray-600 dark:text-gray-400">
                      {title}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
