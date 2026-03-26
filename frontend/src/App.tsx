import { useState, useCallback } from 'react'
import './App.css'
import { STATUS_LABELS } from './constants'
import { useWaveSurfer } from './hooks/useWaveSurfer'
import { useGeneration } from './hooks/useGeneration'
import { useTrackHistory } from './hooks/useTrackHistory'
import { downloadTrack } from './utils/downloadTrack'
import type { Track } from './types'

export default function App() {
  const [prompt, setPrompt] = useState('')
  const [duration, setDuration] = useState(20)
  const [temperature, setTemperature] = useState(0.9)
  const [cfgCoef, setCfgCoef] = useState(3.0)
  const [showAdvanced, setShowAdvanced] = useState(false)

  const { waveRef, isPlaying, currentUrl, currentPrompt, activeTrackId, loadTrack, playPause } = useWaveSurfer()
  const { tracks, addTrack } = useTrackHistory()

  const onSuccess = useCallback((track: Track) => {
    loadTrack(track.file_url, track.prompt, track.task_id)
    addTrack(track)
  }, [loadTrack, addTrack])

  const { status, elapsed, generate } = useGeneration(onSuccess)

  const handleGenerate = () => {
    if (status === 'queued' || status === 'generating') return
    generate({ prompt, duration, temperature, cfgCoef })
  }

  const isLoading = status === 'queued' || status === 'generating'

  return (
    <main className="app">
      <div className="container">

        <header className="header">
          <h1 className="title">Music Generator</h1>
          <p className="subtitle">Powered by MusicGen Medium</p>
        </header>

        <div className="card">
          <label htmlFor="prompt" className="field-label">PROMPT</label>
          <textarea
            id="prompt"
            className="textarea"
            value={prompt}
            onChange={e => setPrompt(e.target.value)}
            placeholder="Describe the music in English: lofi hip hop, relaxed piano, 85 bpm, vinyl crackle..."
            onKeyDown={e => { if (e.key === 'Enter' && e.ctrlKey) handleGenerate() }}
          />

          <div className="duration-row">
            <label htmlFor="duration" className="field-label">DURATION</label>
            <span className="duration-value">{duration}s</span>
            <input
              id="duration"
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
            aria-expanded={showAdvanced}
          >
            Advanced
            <svg
              className={`chevron-icon${showAdvanced ? ' open' : ''}`}
              width="12"
              height="12"
              viewBox="0 0 12 12"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
              aria-hidden="true"
            >
              <path d="M2 4L6 8L10 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>

          <div className={`advanced-panel${showAdvanced ? ' open' : ''}`}>
            <div className="advanced-inner">
              <div className="duration-row">
                <label htmlFor="temperature" className="field-label">TEMPERATURE</label>
                <span className="duration-value">{temperature.toFixed(1)}</span>
                <input
                  id="temperature"
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
                <label htmlFor="cfg-coef" className="field-label">CFG COEF</label>
                <span className="duration-value">{cfgCoef.toFixed(1)}</span>
                <input
                  id="cfg-coef"
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
            onClick={handleGenerate}
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
                onClick={playPause}
                aria-label={isPlaying ? 'Pause' : 'Play'}
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
                role="button"
                tabIndex={0}
                onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') loadTrack(t.file_url, t.prompt, t.task_id) }}
                aria-label={`Play: ${t.prompt}`}
              >
                <span className="track-icon" aria-hidden="true">{t.task_id === activeTrackId ? '▶' : '○'}</span>
                <span className="track-prompt">{t.prompt}</span>
                <span className="track-duration">{t.duration}s</span>
              </div>
            ))}
          </div>
        )}

      </div>
    </main>
  )
}
