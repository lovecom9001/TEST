import { useState } from 'react'
import PDFViewer from './components/PDFViewer'
import Toolbar from './components/Toolbar'
import Sidebar from './components/Sidebar'
import { Tool, Annotation } from './types'

function App() {
  const [pdfFile, setPdfFile] = useState<File | null>(null)
  const [currentTool, setCurrentTool] = useState<Tool>('select')
  const [annotations, setAnnotations] = useState<Annotation[]>([])
  const [color, setColor] = useState('#000000')
  const [fontSize, setFontSize] = useState(16)

  const handleFileUpload = (file: File) => {
    setPdfFile(file)
    setAnnotations([])
  }

  const handleAddAnnotation = (annotation: Annotation) => {
    setAnnotations([...annotations, annotation])
  }

  const handleDeleteAnnotation = (id: string) => {
    setAnnotations(annotations.filter(a => a.id !== id))
  }

  const handleUpdateAnnotation = (id: string, updates: Partial<Annotation>) => {
    setAnnotations(annotations.map(a =>
      a.id === id ? { ...a, ...updates } : a
    ))
  }

  return (
    <div className="flex h-screen bg-gray-900">
      <Sidebar
        currentTool={currentTool}
        onToolChange={setCurrentTool}
        color={color}
        onColorChange={setColor}
        fontSize={fontSize}
        onFontSizeChange={setFontSize}
      />

      <div className="flex-1 flex flex-col">
        <Toolbar onFileUpload={handleFileUpload} />

        <div className="flex-1 overflow-auto">
          {pdfFile ? (
            <PDFViewer
              file={pdfFile}
              currentTool={currentTool}
              annotations={annotations}
              onAddAnnotation={handleAddAnnotation}
              onDeleteAnnotation={handleDeleteAnnotation}
              onUpdateAnnotation={handleUpdateAnnotation}
              color={color}
              fontSize={fontSize}
            />
          ) : (
            <div className="flex items-center justify-center h-full">
              <div className="text-center text-gray-400">
                <p className="text-xl mb-2">PDF 파일을 업로드하세요</p>
                <p className="text-sm">상단의 "PDF 파일 선택" 버튼을 클릭하세요</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
