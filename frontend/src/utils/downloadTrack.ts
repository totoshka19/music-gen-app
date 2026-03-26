import { BASE_URL } from '../constants'

export async function downloadTrack(url: string): Promise<void> {
  const resp = await fetch(`${BASE_URL}${url}`)
  const blob = await resp.blob()
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = url.split('/').pop() ?? 'track.wav'
  a.click()
  URL.revokeObjectURL(a.href)
}
