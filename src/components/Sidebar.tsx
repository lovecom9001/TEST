import { MousePointer, Type, Image, Square, Circle, PenTool } from 'lucide-react'
import { Tool } from '../types'

interface SidebarProps {
  currentTool: Tool
  onToolChange: (tool: Tool) => void
  color: string
  onColorChange: (color: string) => void
  fontSize: number
  onFontSizeChange: (size: number) => void
}

function Sidebar({ currentTool, onToolChange, color, onColorChange, fontSize, onFontSizeChange }: SidebarProps) {
  const tools: { id: Tool; icon: any; label: string }[] = [
    { id: 'select', icon: MousePointer, label: '선택' },
    { id: 'text', icon: Type, label: '텍스트' },
    { id: 'image', icon: Image, label: '이미지' },
    { id: 'rectangle', icon: Square, label: '사각형' },
    { id: 'circle', icon: Circle, label: '원' },
    { id: 'signature', icon: PenTool, label: '서명' },
  ]

  return (
    <div className="w-64 bg-gray-800 border-r border-gray-700 p-4 flex flex-col gap-6">
      <div>
        <h2 className="text-sm font-semibold text-gray-400 mb-3">도구</h2>
        <div className="space-y-2">
          {tools.map((tool) => {
            const Icon = tool.icon
            return (
              <button
                key={tool.id}
                onClick={() => onToolChange(tool.id)}
                className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg transition-colors ${
                  currentTool === tool.id
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                <Icon size={20} />
                <span>{tool.label}</span>
              </button>
            )
          })}
        </div>
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-semibold text-gray-400 mb-2">
            색상
          </label>
          <input
            type="color"
            value={color}
            onChange={(e) => onColorChange(e.target.value)}
            className="w-full h-10 rounded cursor-pointer"
          />
        </div>

        <div>
          <label className="block text-sm font-semibold text-gray-400 mb-2">
            글자 크기: {fontSize}px
          </label>
          <input
            type="range"
            min="10"
            max="72"
            value={fontSize}
            onChange={(e) => onFontSizeChange(Number(e.target.value))}
            className="w-full"
          />
        </div>
      </div>
    </div>
  )
}

export default Sidebar
