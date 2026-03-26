export type Status = 'idle' | 'queued' | 'generating' | 'done' | 'error'

export interface Track {
  task_id: string
  prompt: string
  file_url: string
  duration: number
  created_at: string
}
