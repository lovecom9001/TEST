import { useEffect, useRef, useState } from 'react'
import { getDocument, GlobalWorkerOptions } from 'pdfjs-dist'
import { Tool, Annotation } from '../types'
import AnnotationLayer from './AnnotationLayer'

// PDF.js worker 설정
GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js`

interface PDFViewerProps {
  file: File
  currentTool: Tool
  annotations: Annotation[]
  onAddAnnotation: (annotation: Annotation) => void
  onDeleteAnnotation: (id: string) => void
  onUpdateAnnotation: (id: string, updates: Partial<Annotation>) => void
  color: string
  fontSize: number
}

function PDFViewer({
  file,
  currentTool,
  annotations,
  onAddAnnotation,
  onDeleteAnnotation,
  onUpdateAnnotation,
  color,
  fontSize,
}: PDFViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [numPages, setNumPages] = useState(0)
  const [currentPage, setCurrentPage] = useState(1)
  const [pdfDoc, setPdfDoc] = useState<any>(null)

  useEffect(() => {
    const loadPDF = async () => {
      const fileReader = new FileReader()

      fileReader.onload = async function () {
        const typedArray = new Uint8Array(this.result as ArrayBuffer)
        const loadingTask = getDocument(typedArray)
        const pdf = await loadingTask.promise

        setPdfDoc(pdf)
        setNumPages(pdf.numPages)
        setCurrentPage(1)
      }

      fileReader.readAsArrayBuffer(file)
    }

    loadPDF()
  }, [file])

  useEffect(() => {
    if (!pdfDoc || !canvasRef.current) return

    const renderPage = async () => {
      const page = await pdfDoc.getPage(currentPage)
      const canvas = canvasRef.current!
      const context = canvas.getContext('2d')!

      const viewport = page.getViewport({ scale: 1.5 })
      canvas.height = viewport.height
      canvas.width = viewport.width

      const renderContext = {
        canvasContext: context,
        viewport: viewport,
      }

      await page.render(renderContext).promise
    }

    renderPage()
  }, [pdfDoc, currentPage])

  const handleCanvasClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (currentTool === 'select') return

    const rect = e.currentTarget.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    const newAnnotation: Annotation = {
      id: Date.now().toString(),
      type: currentTool,
      page: currentPage,
      x,
      y,
      color,
      fontSize,
    }

    if (currentTool === 'text') {
      const text = prompt('텍스트를 입력하세요:')
      if (text) {
        newAnnotation.text = text
        onAddAnnotation(newAnnotation)
      }
    } else if (currentTool === 'image') {
      const input = document.createElement('input')
      input.type = 'file'
      input.accept = 'image/*'
      input.onchange = (e: any) => {
        const file = e.target.files[0]
        if (file) {
          const reader = new FileReader()
          reader.onload = (e) => {
            newAnnotation.imageUrl = e.target?.result as string
            newAnnotation.width = 100
            newAnnotation.height = 100
            onAddAnnotation(newAnnotation)
          }
          reader.readAsDataURL(file)
        }
      }
      input.click()
    } else if (currentTool === 'rectangle' || currentTool === 'circle') {
      newAnnotation.width = 100
      newAnnotation.height = 100
      onAddAnnotation(newAnnotation)
    } else if (currentTool === 'signature') {
      const text = prompt('서명을 입력하세요:')
      if (text) {
        newAnnotation.text = text
        newAnnotation.fontSize = 24
        onAddAnnotation(newAnnotation)
      }
    }
  }

  return (
    <div className="flex flex-col items-center p-8 bg-gray-900">
      <div className="relative mb-4" onClick={handleCanvasClick}>
        <canvas ref={canvasRef} className="border border-gray-700 shadow-lg" />
        <AnnotationLayer
          annotations={annotations.filter(a => a.page === currentPage)}
          onDelete={onDeleteAnnotation}
          onUpdate={onUpdateAnnotation}
        />
      </div>

      {numPages > 0 && (
        <div className="flex items-center gap-4 bg-gray-800 px-6 py-3 rounded-lg">
          <button
            onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
            disabled={currentPage === 1}
            className="px-4 py-2 bg-blue-600 text-white rounded disabled:bg-gray-600 disabled:cursor-not-allowed hover:bg-blue-700 transition-colors"
          >
            이전
          </button>
          <span className="text-white">
            {currentPage} / {numPages}
          </span>
          <button
            onClick={() => setCurrentPage(Math.min(numPages, currentPage + 1))}
            disabled={currentPage === numPages}
            className="px-4 py-2 bg-blue-600 text-white rounded disabled:bg-gray-600 disabled:cursor-not-allowed hover:bg-blue-700 transition-colors"
          >
            다음
          </button>
        </div>
      )}
    </div>
  )
}

export default PDFViewer
