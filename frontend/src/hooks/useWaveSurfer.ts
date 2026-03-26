import { useRef, useState, useCallback, useEffect } from 'react'
import WaveSurfer from 'wavesurfer.js'
import { BASE_URL } from '../constants'

export function useWaveSurfer() {
  const waveRef = useRef<HTMLDivElement>(null)
  const wsRef = useRef<WaveSurfer | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentUrl, setCurrentUrl] = useState('')
  const [currentPrompt, setCurrentPrompt] = useState('')
  const [activeTrackId, setActiveTrackId] = useState<string | null>(null)

  const loadTrack = useCallback((url: string, trackPrompt: string, trackId: string) => {
    if (!waveRef.current) return
    wsRef.current?.destroy()
    setIsPlaying(false)
    setCurrentUrl(url)
    setCurrentPrompt(trackPrompt)
    setActiveTrackId(trackId)

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
    ws.load(`${BASE_URL}${url}`)
    wsRef.current = ws
  }, [])

  const playPause = useCallback(() => {
    wsRef.current?.playPause()
  }, [])

  useEffect(() => {
    return () => { wsRef.current?.destroy() }
  }, [])

  return { waveRef, isPlaying, currentUrl, currentPrompt, activeTrackId, loadTrack, playPause }
}
