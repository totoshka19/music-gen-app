import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'
import { API } from '../constants'
import type { Track } from '../types'

export function useTrackHistory() {
  const [tracks, setTracks] = useState<Track[]>([])

  useEffect(() => {
    axios.get(`${API}/tracks`).then(({ data }) => {
      if (data.tracks?.length) setTracks(data.tracks)
    }).catch(() => {})
  }, [])

  const addTrack = useCallback((track: Track) => {
    setTracks(prev => [track, ...prev])
  }, [])

  return { tracks, addTrack }
}
