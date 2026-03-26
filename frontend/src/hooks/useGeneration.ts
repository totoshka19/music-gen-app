import { useState, useRef, useCallback, useEffect } from 'react'
import axios from 'axios'
import { API } from '../constants'
import type { Status, Track } from '../types'

interface GenerationParams {
  prompt: string
  duration: number
  temperature: number
  cfgCoef: number
}

export function useGeneration(onSuccess: (track: Track) => void) {
  const [status, setStatus] = useState<Status>('idle')
  const [elapsed, setElapsed] = useState(0)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  // Ref чтобы polling-интервал всегда видел актуальный onSuccess (избегаем stale closure)
  const onSuccessRef = useRef(onSuccess)
  useEffect(() => { onSuccessRef.current = onSuccess }, [onSuccess])

  const clearTimers = useCallback(() => {
    if (pollRef.current) clearInterval(pollRef.current)
    if (timerRef.current) clearInterval(timerRef.current)
  }, [])

  useEffect(() => () => clearTimers(), [clearTimers])

  const generate = useCallback(async ({ prompt, duration, temperature, cfgCoef }: GenerationParams) => {
    if (!prompt.trim()) return
    setStatus('queued')
    setElapsed(0)
    timerRef.current = setInterval(() => setElapsed(s => s + 1), 1000)

    try {
      const { data } = await axios.post(`${API}/generate`, {
        prompt,
        duration,
        temperature,
        cfg_coef: cfgCoef,
      })
      const taskId: string = data.task_id

      pollRef.current = setInterval(async () => {
        try {
          const { data: s } = await axios.get(`${API}/status/${taskId}`)
          setStatus(s.status)

          if (s.status === 'done') {
            clearTimers()
            const { data: r } = await axios.get(`${API}/result/${taskId}`)
            onSuccessRef.current({
              task_id: taskId,
              prompt,
              file_url: r.file_url,
              duration: r.duration,
              created_at: r.created_at,
            })
          }

          if (s.status === 'failed') {
            clearTimers()
            setStatus('error')
          }
        } catch {
          clearTimers()
          setStatus('error')
        }
      }, 5000)
    } catch {
      clearTimers()
      setStatus('error')
    }
  }, [clearTimers])

  return { status, elapsed, generate }
}
