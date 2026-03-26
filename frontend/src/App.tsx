import { useState, useRef, useEffect, useCallback } from 'react'
import axios from 'axios'
import WaveSurfer from 'wavesurfer.js'
import './App.css'

const API = 'http://localhost:8000/api/v1'

type Status = 'idle' | 'queued' | 'generating' | 'done' | 'error'

interface Track {
  task_id: string
  prompt: string
  file_url: string
  duration: number
  created_at: string
}

const STATUS_LABELS: Record<Status, string> = {
  idle: 'Generate',
  queued: 'In queue...',
  generating: 'Generating...',
  done: 'Generate again',
  error: 'Error — retry',
}

export default function App() {
  const [prompt, setPrompt] = useState('')
  const [duration, setDuration] = useState(20)
  const [status, setStatus] = useState<Status>('idle')
  const [tracks, setTracks] = useState<Track[]>([])
  const [currentUrl, setCurrentUrl] = useState('')
  const [currentPrompt, setCurrentPrompt] = useState('')
  const [isPlaying, setIsPlaying] = useState(false)
  const [activeTrackId, setActiveTrackId] = useState<string | null>(null)
  const [elapsed, setElapsed] = useState(0)
  const [temperature, setTemperature] = useState(0.9)
  const [cfgCoef, setCfgCoef] = useState(3.0)
  const [showAdvanced, setShowAdvanced] = useState(false)

  const waveRef = useRef<HTMLDivElement>(null)
  const wsRef = useRef<WaveSurfer | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const downloadTrack = useCallback(async (url: string) => {
    const resp = await fetch(`http://localhost:8000${url}`)
    const blob = await resp.blob()
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = url.split('/').pop() ?? 'track.wav'
    a.click()
    URL.revokeObjectURL(a.href)
  }, [])

  const loadTrack = useCallback((url: string, trackPrompt: string, trackId: string) => {
    if (!waveRef.current) return
    wsRef.current?.destroy()
    setIsPlaying(false)
    setActiveTrackId(trackId)
    setCurrentPrompt(trackPrompt)

    const ws = WaveSurfer.create({
      container: waveRef.current,
      waveColor: '#6366f1',
      progressColor: '#22d3ee',
      height: 72,
      barWidth: 2,
      barGap: 1,
      barRadius: 3,
      cursorColor: 'transparent',
    })
    ws.on('play', () => setIsPlaying(true))
    ws.on('pause', () => setIsPlaying(false))
    ws.on('finish', () => setIsPlaying(false))
    ws.load(`http://localhost:8000${url}`)
    wsRef.current = ws
  }, [])

  useEffect(() => {
    if (currentUrl) loadTrack(currentUrl, currentPrompt, activeTrackId ?? '')
  }, [currentUrl]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    axios.get(`${API}/tracks`).then(({ data }) => {
      if (data.tracks?.length) setTracks(data.tracks)
    }).catch(() => {})
  }, [])

  useEffect(() => () => {
    wsRef.current?.destroy()
    if (pollRef.current) clearInterval(pollRef.current)
    if (timerRef.current) clearInterval(timerRef.current)
  }, [])

  const generate = async () => {
    if (!prompt.trim() || ['queued', 'generating'].includes(status)) return
    setStatus('queued')
    setElapsed(0)
    timerRef.current = setInterval(() => setElapsed(s => s + 1), 1000)

    try {
      const { data } = await axios.post(`${API}/generate`, { prompt, duration, temperature, cfg_coef: cfgCoef })
      const taskId: string = data.task_id

      pollRef.current = setInterval(async () => {
        try {
          const { data: s } = await axios.get(`${API}/status/${taskId}`)
          setStatus(s.status)

          if (s.status === 'done') {
            clearInterval(pollRef.current!)
            clearInterval(timerRef.current!)
            const { data: r } = await axios.get(`${API}/result/${taskId}`)
            const track: Track = {
              task_id: taskId,
              prompt,
              file_url: r.file_url,
              duration: r.duration,
              created_at: r.created_at,
            }
            setCurrentUrl(r.file_url)
            setCurrentPrompt(prompt)
            setActiveTrackId(taskId)
            setTracks(prev => [track, ...prev])
          }

          if (s.status === 'failed') {
            clearInterval(pollRef.current!)
            clearInterval(timerRef.current!)
            setStatus('error')
          }
        } catch {
          clearInterval(pollRef.current!)
          clearInterval(timerRef.current!)
          setStatus('error')
        }
      }, 5000)
    } catch {
      clearInterval(timerRef.current!)
      setStatus('error')
    }
  }

  const isLoading = status === 'queued' || status === 'generating'

  return (
    <div className="app">
      <div className="container">

        <header className="header">
          <h1 className="title">Music Generator</h1>
          <p className="subtitle">Powered by MusicGen Medium</p>
        </header>

        <div className="card">
          <label className="field-label">PROMPT</label>
          <textarea
            className="textarea"
            value={prompt}
            onChange={e => setPrompt(e.target.value)}
            placeholder="Describe the music in English: lofi hip hop, relaxed piano, 85 bpm, vinyl crackle..."
            onKeyDown={e => { if (e.key === 'Enter' && e.ctrlKey) generate() }}
          />

          <div className="duration-row">
            <label className="field-label">DURATION</label>
            <span className="duration-value">{duration}s</span>
            <input
              type="range"
              className="slider"
              min={5}
              max={120}
              step={5}
              value={duration}
              onChange={e => setDuration(+e.target.value)}
            />
          </div>

          <button
            className="btn-advanced"
            onClick={() => setShowAdvanced(v => !v)}
          >
            Advanced
            <svg
              className={`chevron-icon${showAdvanced ? ' open' : ''}`}
              width="12"
              height="12"
              viewBox="0 0 12 12"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path d="M2 4L6 8L10 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>

          <div className={`advanced-panel${showAdvanced ? ' open' : ''}`}>
            <div className="advanced-inner">
              <div className="duration-row">
                <label className="field-label">TEMPERATURE</label>
                <span className="duration-value">{temperature.toFixed(1)}</span>
                <input
                  type="range"
                  className="slider"
                  min={0.5}
                  max={1.5}
                  step={0.1}
                  value={temperature}
                  onChange={e => setTemperature(+e.target.value)}
                />
              </div>
              <div className="duration-row">
                <label className="field-label">CFG COEF</label>
                <span className="duration-value">{cfgCoef.toFixed(1)}</span>
                <input
                  type="range"
                  className="slider"
                  min={1.0}
                  max={6.0}
                  step={0.5}
                  value={cfgCoef}
                  onChange={e => setCfgCoef(+e.target.value)}
                />
              </div>
            </div>
          </div>

          <button
            className={`btn-primary${isLoading ? ' loading' : ''}${status === 'error' ? ' error' : ''}`}
            onClick={generate}
            disabled={isLoading}
          >
            {isLoading && <span className="spinner" />}
            {STATUS_LABELS[status]}
            {isLoading && <span className="timer">{elapsed}s</span>}
          </button>
        </div>

        {currentUrl && (
          <div className="card player-card">
            <div className="player-prompt">{currentPrompt}</div>
            <div ref={waveRef} className="waveform" />
            <div className="player-controls">
              <button
                className="btn-play"
                onClick={() => wsRef.current?.playPause()}
              >
                {isPlaying ? '⏸' : '▶'}
              </button>
              <button
                className="btn-download"
                onClick={() => downloadTrack(currentUrl)}
              >
                Download .wav
              </button>
            </div>
          </div>
        )}

        {tracks.length > 0 && (
          <div className="history">
            <div className="section-label">HISTORY</div>
            {tracks.map(t => (
              <div
                key={t.task_id}
                className={`track-item${t.task_id === activeTrackId ? ' active' : ''}`}
                onClick={() => loadTrack(t.file_url, t.prompt, t.task_id)}
              >
                <span className="track-icon">{t.task_id === activeTrackId ? '▶' : '○'}</span>
                <span className="track-prompt">{t.prompt}</span>
                <span className="track-duration">{t.duration}s</span>
              </div>
            ))}
          </div>
        )}

      </div>
    </div>
  )
}
