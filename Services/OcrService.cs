using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Windows.Graphics.Imaging;
using Windows.Media.Ocr;

namespace PDFEditor.Services
{
    public class OcrTextBlock
    {
        public string Text { get; set; } = "";
        public double X { get; set; }
        public double Y { get; set; }
        public double Width { get; set; }
        public double Height { get; set; }
    }

    public class OcrService
    {
        public async Task<List<OcrTextBlock>> RecognizeWithPositionsAsync(byte[] imageBytes)
        {
            using var ms = new MemoryStream(imageBytes);
            var randomAccessStream = ms.AsRandomAccessStream();
            var decoder = await BitmapDecoder.CreateAsync(randomAccessStream);

            using var softwareBitmap = await decoder.GetSoftwareBitmapAsync();
            using var convertedBitmap = SoftwareBitmap.Convert(
                softwareBitmap,
                BitmapPixelFormat.Bgra8,
                BitmapAlphaMode.Premultiplied);

            OcrEngine? engine = null;
            try
            {
                var korean = new Windows.Globalization.Language("ko");
                if (OcrEngine.IsLanguageSupported(korean))
                    engine = OcrEngine.TryCreateFromLanguage(korean);
            }
            catch { }
            engine ??= OcrEngine.TryCreateFromUserProfileLanguages();

            if (engine == null)
                throw new InvalidOperationException(
                    "OCR 엔진을 초기화할 수 없습니다.\n" +
                    "Windows 설정 > 시간 및 언어 > 언어에서 언어팩을 설치해주세요.");

            var result = await engine.RecognizeAsync(convertedBitmap);

            var blocks = new List<OcrTextBlock>();
            foreach (var line in result.Lines)
            {
                blocks.Add(new OcrTextBlock
                {
                    Text = line.Text,
                    X = line.BoundingRect.X,
                    Y = line.BoundingRect.Y,
                    Width = line.BoundingRect.Width,
                    Height = line.BoundingRect.Height
                });
            }
            return blocks;
        }
    }
}
