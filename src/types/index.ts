export type Tool = 'select' | 'text' | 'image' | 'rectangle' | 'circle' | 'signature'

export interface Annotation {
  id: string
  type: Tool
  page: number
  x: number
  y: number
  width?: number
  height?: number
  text?: string
  color?: string
  fontSize?: number
  imageUrl?: string
}
