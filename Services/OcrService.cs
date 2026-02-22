using System;
using System.IO;
using System.Threading.Tasks;
using Windows.Graphics.Imaging;
using Windows.Media.Ocr;

namespace PDFEditor.Services
{
    public class OcrService
    {
        public async Task<string> RecognizeTextAsync(byte[] imageBytes)
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

            // 한국어 우선 시도
            try
            {
                var korean = new Windows.Globalization.Language("ko");
                if (OcrEngine.IsLanguageSupported(korean))
                    engine = OcrEngine.TryCreateFromLanguage(korean);
            }
            catch { }

            // 시스템 언어로 대체
            engine ??= OcrEngine.TryCreateFromUserProfileLanguages();

            if (engine == null)
                throw new InvalidOperationException(
                    "OCR 엔진을 초기화할 수 없습니다.\n" +
                    "Windows 설정 > 시간 및 언어 > 언어에서 언어팩을 설치해주세요.");

            var result = await engine.RecognizeAsync(convertedBitmap);
            return result.Text;
        }
    }
}
