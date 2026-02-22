using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using PDFtoImage;
using iText.Kernel.Pdf;
using iText.Kernel.Pdf.Canvas;
using iText.Kernel.Colors;
using iText.Kernel.Font;
using iText.IO.Font.Constants;
using PDFEditor.Models;
using SkiaSharp;

namespace PDFEditor.Services
{
    public class PdfService
    {
        private PdfDocument? _pdfDocument;
        private string? _currentPath;

        public int PageCount => _pdfDocument?.GetNumberOfPages() ?? 0;

        public async Task LoadPdfAsync(string filePath)
        {
            await Task.Run(() =>
            {
                _currentPath = filePath;
                _pdfDocument = new PdfDocument(new PdfReader(filePath));
            });
        }

        public BitmapImage RenderPage(int pageIndex)
        {
            if (_currentPath == null)
                throw new InvalidOperationException("PDF가 로드되지 않았습니다.");

            // PDFtoImage를 사용하여 페이지 렌더링 (Stream 방식)
            using var stream = System.IO.File.OpenRead(_currentPath);
            using var bitmap = PDFtoImage.Conversion.ToImage(stream, (System.Index)pageIndex, new RenderOptions(Dpi: 150));

            var bitmapImage = new BitmapImage();
            using (var memory = new MemoryStream())
            {
                bitmap.Encode(memory, SKEncodedImageFormat.Png, 100);
                memory.Position = 0;

                bitmapImage.BeginInit();
                bitmapImage.StreamSource = memory;
                bitmapImage.CacheOption = BitmapCacheOption.OnLoad;
                bitmapImage.EndInit();
                bitmapImage.Freeze();
            }

            return bitmapImage;
        }

        public async Task<byte[]> RenderPageToBytesAsync(int pageIndex)
        {
            if (_currentPath == null)
                throw new InvalidOperationException("PDF가 로드되지 않았습니다.");

            return await Task.Run(() =>
            {
                using var fileStream = System.IO.File.OpenRead(_currentPath);
                using var bitmap = PDFtoImage.Conversion.ToImage(fileStream, (System.Index)pageIndex, new RenderOptions(Dpi: 150));
                using var ms = new MemoryStream();
                bitmap.Encode(ms, SKEncodedImageFormat.Png, 100);
                return ms.ToArray();
            });
        }

        public async Task SavePdfAsync(string inputPath, string outputPath, IEnumerable<Annotation> annotations)
        {
            await Task.Run(() =>
            {
                using var reader = new PdfReader(inputPath);
                using var writer = new PdfWriter(outputPath);
                using var pdfDoc = new PdfDocument(reader, writer);

                foreach (var annotation in annotations)
                {
                    var page = pdfDoc.GetPage(annotation.Page);
                    var canvas = new PdfCanvas(page);

                    switch (annotation.Type)
                    {
                        case "Text":
                        case "Signature":
                            DrawText(canvas, annotation, page);
                            break;

                        case "Rectangle":
                            DrawRectangle(canvas, annotation, page);
                            break;

                        case "Circle":
                            DrawCircle(canvas, annotation, page);
                            break;
                    }
                }

                pdfDoc.Close();
            });
        }

        private void DrawText(PdfCanvas canvas, Annotation annotation, PdfPage page)
        {
            if (string.IsNullOrEmpty(annotation.Text))
                return;

            var pageHeight = page.GetPageSize().GetHeight();
            var font = PdfFontFactory.CreateFont(StandardFonts.HELVETICA_BOLD);
            var fontSize = (float)(annotation.FontSize ?? 16);

            var color = ParseColor(annotation.Color ?? "#000000");

            canvas.SaveState();
            canvas.SetFontAndSize(font, fontSize);
            canvas.SetColor(color, true);
            canvas.BeginText();
            canvas.MoveText(annotation.X, pageHeight - annotation.Y - fontSize);
            canvas.ShowText(annotation.Text);
            canvas.EndText();
            canvas.RestoreState();
        }

        private void DrawRectangle(PdfCanvas canvas, Annotation annotation, PdfPage page)
        {
            var pageHeight = page.GetPageSize().GetHeight();
            var color = ParseColor(annotation.Color ?? "#000000");

            canvas.SaveState();
            canvas.SetStrokeColor(color);
            canvas.SetLineWidth(2);
            canvas.Rectangle(
                annotation.X,
                pageHeight - annotation.Y - (annotation.Height ?? 100),
                annotation.Width ?? 100,
                annotation.Height ?? 100
            );
            canvas.Stroke();
            canvas.RestoreState();
        }

        private void DrawCircle(PdfCanvas canvas, Annotation annotation, PdfPage page)
        {
            var pageHeight = page.GetPageSize().GetHeight();
            var color = ParseColor(annotation.Color ?? "#000000");
            var radius = (annotation.Width ?? 100) / 2;
            var centerX = annotation.X + radius;
            var centerY = pageHeight - annotation.Y - radius;

            canvas.SaveState();
            canvas.SetStrokeColor(color);
            canvas.SetLineWidth(2);
            canvas.Circle(centerX, centerY, radius);
            canvas.Stroke();
            canvas.RestoreState();
        }

        private DeviceRgb ParseColor(string hexColor)
        {
            // #RRGGBB 형식을 RGB로 변환
            hexColor = hexColor.TrimStart('#');
            var r = Convert.ToInt32(hexColor.Substring(0, 2), 16);
            var g = Convert.ToInt32(hexColor.Substring(2, 2), 16);
            var b = Convert.ToInt32(hexColor.Substring(4, 2), 16);

            return new DeviceRgb(r, g, b);
        }

        public void Dispose()
        {
            _pdfDocument?.Close();
        }
    }
}
