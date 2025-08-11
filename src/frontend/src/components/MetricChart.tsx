'use client'

import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { ExperimentResult } from '@/types/experiment'
import { experimentLabels, metricColors } from '@/lib/demo-data'
import { formatMetricValue, formatEpoch } from '@/lib/utils'

interface MetricChartProps {
  data: ExperimentResult
  metric: 'accuracy' | 'precision' | 'recall' | 'f1-score'
  title: string
}

export default function MetricChart({ data, metric, title }: MetricChartProps) {
  // Transform data for recharts
  const chartData: any[] = []
  
  // Get all experiment IDs
  const experimentIds = Object.keys(data)
  
  if (experimentIds.length === 0) return null
  
  // Get the maximum number of epochs from all experiments
  const maxEpochs = Math.max(...experimentIds.map(id => data[id][metric].X.length))
  
  // Create data points for each epoch
  for (let i = 0; i < maxEpochs; i++) {
    const dataPoint: any = { epoch: i }
    
    experimentIds.forEach(expId => {
      const expData = data[expId][metric]
      if (i < expData.X.length) {
        dataPoint[expId] = expData.Y[i]
      }
    })
    
    chartData.push(dataPoint)
  }

  // Safe accessor for experiment labels
  const getExperimentLabel = (key: string): string => {
    return experimentLabels[key] || key
  }

  const getMetricColor = (key: string, fallbackIndex: number): string => {
    return metricColors[key] || `hsl(${fallbackIndex * 137.5}, 70%, 50%)`
  }

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white/90 dark:bg-black/90 backdrop-blur-md p-4 rounded-lg border border-gray-200/50 dark:border-gray-700/50 shadow-lg">
          <p className="font-medium text-gray-900 dark:text-gray-100 mb-2">
            {formatEpoch(label)}
          </p>
          {payload.map((entry: any, index: number) => (
            <p key={index} style={{ color: entry.color }} className="text-sm">
              {getExperimentLabel(entry.dataKey)}: {formatMetricValue(entry.value)}
            </p>
          ))}
        </div>
      )
    }
    return null
  }

  return (
    <div className="glass-card rounded-xl p-6">
      <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4 capitalize">
        {title}
      </h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" opacity={0.3} />
            <XAxis 
              dataKey="epoch" 
              stroke="#6B7280"
              fontSize={12}
              tickFormatter={formatEpoch}
            />
            <YAxis 
              stroke="#6B7280"
              fontSize={12}
              domain={[0, 1]}
              tickFormatter={formatMetricValue}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend 
              formatter={(value) => getExperimentLabel(value)}
              wrapperStyle={{ fontSize: '12px' }}
            />
            {experimentIds.map((expId, index) => (
              <Line
                key={expId}
                type="monotone"
                dataKey={expId}
                stroke={getMetricColor(expId, index)}
                strokeWidth={2}
                dot={{ r: 3, strokeWidth: 2 }}
                activeDot={{ r: 5, strokeWidth: 0 }}
                connectNulls={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
