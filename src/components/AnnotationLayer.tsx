import { Trash2 } from 'lucide-react'
import { Annotation } from '../types'

interface AnnotationLayerProps {
  annotations: Annotation[]
  onDelete: (id: string) => void
  onUpdate: (id: string, updates: Partial<Annotation>) => void
}

function AnnotationLayer({ annotations, onDelete, onUpdate }: AnnotationLayerProps) {
  return (
    <div className="absolute inset-0 pointer-events-none">
      {annotations.map((annotation) => (
        <div
          key={annotation.id}
          className="absolute pointer-events-auto group"
          style={{
            left: annotation.x,
            top: annotation.y,
            width: annotation.width,
            height: annotation.height,
          }}
        >
          {annotation.type === 'text' && (
            <div
              style={{
                color: annotation.color,
                fontSize: `${annotation.fontSize}px`,
                fontWeight: 'bold',
              }}
            >
              {annotation.text}
            </div>
          )}

          {annotation.type === 'image' && annotation.imageUrl && (
            <img
              src={annotation.imageUrl}
              alt="Annotation"
              className="w-full h-full object-contain"
            />
          )}

          {annotation.type === 'rectangle' && (
            <div
              className="w-full h-full border-2"
              style={{ borderColor: annotation.color }}
            />
          )}

          {annotation.type === 'circle' && (
            <div
              className="w-full h-full border-2 rounded-full"
              style={{ borderColor: annotation.color }}
            />
          )}

          {annotation.type === 'signature' && (
            <div
              style={{
                color: annotation.color,
                fontSize: `${annotation.fontSize}px`,
                fontFamily: 'cursive',
                fontStyle: 'italic',
              }}
            >
              {annotation.text}
            </div>
          )}

          <button
            onClick={() => onDelete(annotation.id)}
            className="absolute -top-2 -right-2 bg-red-500 text-white p-1 rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
          >
            <Trash2 size={16} />
          </button>
        </div>
      ))}
    </div>
  )
}

export default AnnotationLayer
