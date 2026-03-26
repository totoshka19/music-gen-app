import type { Status } from './types'

export const BASE_URL = 'http://localhost:8000'
export const API = `${BASE_URL}/api/v1`

export const STATUS_LABELS: Record<Status, string> = {
  idle: 'Generate',
  queued: 'In queue...',
  generating: 'Generating...',
  done: 'Generate again',
  error: 'Error — retry',
}
