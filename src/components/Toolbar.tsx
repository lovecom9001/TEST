import { FileUp } from 'lucide-react'

interface ToolbarProps {
  onFileUpload: (file: File) => void
}

function Toolbar({ onFileUpload }: ToolbarProps) {
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file && file.type === 'application/pdf') {
      onFileUpload(file)
    }
  }

  return (
    <div className="bg-gray-800 border-b border-gray-700 p-4 flex items-center justify-between">
      <h1 className="text-xl font-bold text-white">PDF 편집기</h1>

      <div className="flex gap-4">
        <label className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg cursor-pointer transition-colors">
          <FileUp size={20} />
          <span>PDF 파일 선택</span>
          <input
            type="file"
            accept="application/pdf"
            onChange={handleFileChange}
            className="hidden"
          />
        </label>
      </div>
    </div>
  )
}

export default Toolbar
