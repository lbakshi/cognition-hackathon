export interface MetricData {
  X: number[]
  Y: number[]
}

export interface ExperimentMetrics {
  accuracy: MetricData
  precision: MetricData
  recall: MetricData
  'f1-score': MetricData
}

export interface ExperimentResult {
  [experiment_id: string]: ExperimentMetrics
}

export interface ExperimentRequest {
  prompt: string
}

export interface ExperimentStatus {
  status: 'idle' | 'running' | 'completed' | 'error'
  progress?: number
  message?: string
}
