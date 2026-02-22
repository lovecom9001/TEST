using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Shapes;
using Microsoft.Win32;
using PDFEditor.Models;
using PDFEditor.Services;

namespace PDFEditor
{
    public partial class MainWindow : Window
    {
        private MainViewModel _viewModel;
        private PdfService _pdfService;
        private OcrService _ocrService;
        private readonly List<TextBox> _ocrOverlays = new();
        private double _pdfScale = 1.0;
        private string? _currentTool = "Select";
        private string _selectedColor = "#000000";

        public MainWindow()
        {
            InitializeComponent();
            _viewModel = (MainViewModel)DataContext;
            _pdfService = new PdfService();
            _ocrService = new OcrService();
        }

        private async void OpenPDF_Click(object sender, RoutedEventArgs e)
        {
            var openFileDialog = new OpenFileDialog
            {
                Filter = "PDF Files (*.pdf)|*.pdf",
                Title = "PDF 파일 선택"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                try
                {
                    _viewModel.CurrentPdfPath = openFileDialog.FileName;
                    await _pdfService.LoadPdfAsync(openFileDialog.FileName);

                    _viewModel.TotalPages = _pdfService.PageCount;
                    _viewModel.CurrentPage = 1;
                    _viewModel.IsPdfLoaded = true;

                    RenderCurrentPage();
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"PDF 파일을 열 수 없습니다: {ex.Message}", "오류", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }

        private async void SavePDF_Click(object sender, RoutedEventArgs e)
        {
            if (!_viewModel.IsPdfLoaded || string.IsNullOrEmpty(_viewModel.CurrentPdfPath))
            {
                MessageBox.Show("저장할 PDF가 없습니다.", "알림", MessageBoxButton.OK, MessageBoxImage.Information);
                return;
            }

            var saveFileDialog = new SaveFileDialog
            {
                Filter = "PDF Files (*.pdf)|*.pdf",
                Title = "PDF 저장",
                FileName = "edited_" + System.IO.Path.GetFileName(_viewModel.CurrentPdfPath)
            };

            if (saveFileDialog.ShowDialog() == true)
            {
                try
                {
                    await _pdfService.SavePdfAsync(_viewModel.CurrentPdfPath, saveFileDialog.FileName, _viewModel.Annotations);
                    MessageBox.Show("PDF가 성공적으로 저장되었습니다!", "성공", MessageBoxButton.OK, MessageBoxImage.Information);
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"PDF를 저장할 수 없습니다: {ex.Message}", "오류", MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }

        private void PdfViewer_PreviewMouseWheel(object sender, MouseWheelEventArgs e)
        {
            if (Keyboard.Modifiers != ModifierKeys.Control) return;
            e.Handled = true;

            _pdfScale = Math.Clamp(_pdfScale + (e.Delta > 0 ? 0.1 : -0.1), 0.25, 4.0);
            PdfScaleTransform.ScaleX = _pdfScale;
            PdfScaleTransform.ScaleY = _pdfScale;
        }

        private void Tool_Click(object sender, RoutedEventArgs e)
        {
            if (sender is Button button)
            {
                _currentTool = button.Tag?.ToString();

                // 모든 도구 버튼의 선택 상태 초기화
                var parent = (StackPanel)button.Parent;
                foreach (var child in parent.Children)
                {
                    if (child is Button btn && btn.Tag != null)
                    {
                        btn.SetValue(Button.BackgroundProperty, FindResource("SurfaceBrush"));
                        btn.SetValue(Button.ForegroundProperty, new SolidColorBrush((Color)ColorConverter.ConvertFromString("#9CA3AF")));
                    }
                }

                // 선택된 버튼 강조
                button.SetValue(Button.BackgroundProperty, FindResource("PrimaryBrush"));
                button.SetValue(Button.ForegroundProperty, FindResource("TextBrush"));
            }
        }

        private void Canvas_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
        {
            if (!_viewModel.IsPdfLoaded || _currentTool == "Select")
                return;

            var position = e.GetPosition(AnnotationCanvas);

            switch (_currentTool)
            {
                case "Text":
                    AddTextAnnotation(position);
                    break;
                case "Rectangle":
                    AddRectangleAnnotation(position);
                    break;
                case "Circle":
                    AddCircleAnnotation(position);
                    break;
                case "Signature":
                    AddSignatureAnnotation(position);
                    break;
            }
        }

        private void AddTextAnnotation(Point position)
        {
            var inputDialog = new TextInputDialog();
            if (inputDialog.ShowDialog() == true && !string.IsNullOrEmpty(inputDialog.InputText))
            {
                var textBlock = new TextBlock
                {
                    Text = inputDialog.InputText,
                    FontSize = FontSizeSlider.Value,
                    Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString(_selectedColor)),
                    FontWeight = FontWeights.Bold
                };

                Canvas.SetLeft(textBlock, position.X);
                Canvas.SetTop(textBlock, position.Y);
                AnnotationCanvas.Children.Add(textBlock);

                _viewModel.Annotations.Add(new Annotation
                {
                    Type = "Text",
                    Page = _viewModel.CurrentPage,
                    X = position.X,
                    Y = position.Y,
                    Text = inputDialog.InputText,
                    Color = _selectedColor,
                    FontSize = FontSizeSlider.Value
                });
            }
        }

        private void AddRectangleAnnotation(Point position)
        {
            var rectangle = new Rectangle
            {
                Width = 100,
                Height = 100,
                Stroke = new SolidColorBrush((Color)ColorConverter.ConvertFromString(_selectedColor)),
                StrokeThickness = 2
            };

            Canvas.SetLeft(rectangle, position.X);
            Canvas.SetTop(rectangle, position.Y);
            AnnotationCanvas.Children.Add(rectangle);

            _viewModel.Annotations.Add(new Annotation
            {
                Type = "Rectangle",
                Page = _viewModel.CurrentPage,
                X = position.X,
                Y = position.Y,
                Width = 100,
                Height = 100,
                Color = _selectedColor
            });
        }

        private void AddCircleAnnotation(Point position)
        {
            var ellipse = new Ellipse
            {
                Width = 100,
                Height = 100,
                Stroke = new SolidColorBrush((Color)ColorConverter.ConvertFromString(_selectedColor)),
                StrokeThickness = 2
            };

            Canvas.SetLeft(ellipse, position.X);
            Canvas.SetTop(ellipse, position.Y);
            AnnotationCanvas.Children.Add(ellipse);

            _viewModel.Annotations.Add(new Annotation
            {
                Type = "Circle",
                Page = _viewModel.CurrentPage,
                X = position.X,
                Y = position.Y,
                Width = 100,
                Height = 100,
                Color = _selectedColor
            });
        }

        private void AddSignatureAnnotation(Point position)
        {
            var inputDialog = new TextInputDialog { Title = "서명 입력" };
            if (inputDialog.ShowDialog() == true && !string.IsNullOrEmpty(inputDialog.InputText))
            {
                var textBlock = new TextBlock
                {
                    Text = inputDialog.InputText,
                    FontSize = 24,
                    Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString(_selectedColor)),
                    FontFamily = new FontFamily("Segoe Script"),
                    FontStyle = FontStyles.Italic
                };

                Canvas.SetLeft(textBlock, position.X);
                Canvas.SetTop(textBlock, position.Y);
                AnnotationCanvas.Children.Add(textBlock);

                _viewModel.Annotations.Add(new Annotation
                {
                    Type = "Signature",
                    Page = _viewModel.CurrentPage,
                    X = position.X,
                    Y = position.Y,
                    Text = inputDialog.InputText,
                    Color = _selectedColor,
                    FontSize = 24
                });
            }
        }

        private async void OCR_Click(object sender, RoutedEventArgs e)
        {
            if (!_viewModel.IsPdfLoaded) return;

            var button = (Button)sender;
            var originalContent = button.Content;
            button.IsEnabled = false;
            button.Content = "🔍 인식 중...";

            try
            {
                ClearOcrOverlays();

                var imageBytes = await _pdfService.RenderPageToBytesAsync(_viewModel.CurrentPage - 1);
                var blocks = await _ocrService.RecognizeWithPositionsAsync(imageBytes);

                if (blocks.Count == 0)
                {
                    MessageBox.Show("인식된 텍스트가 없습니다.", "OCR 결과", MessageBoxButton.OK, MessageBoxImage.Information);
                    return;
                }

                // OCR 이미지 실제 크기로 좌표 변환 (DPI 차이 보정)
                var ocrBmp = new System.Windows.Media.Imaging.BitmapImage();
                ocrBmp.BeginInit();
                ocrBmp.StreamSource = new System.IO.MemoryStream(imageBytes);
                ocrBmp.CacheOption = System.Windows.Media.Imaging.BitmapCacheOption.OnLoad;
                ocrBmp.EndInit();
                double imgW = ocrBmp.PixelWidth;
                double imgH = ocrBmp.PixelHeight;
                double canvasW = AnnotationCanvas.ActualWidth;
                double canvasH = AnnotationCanvas.ActualHeight;
                double scale = Math.Min(canvasW / imgW, canvasH / imgH);
                double offsetX = (canvasW - imgW * scale) / 2;
                double offsetY = (canvasH - imgH * scale) / 2;

                foreach (var block in blocks)
                {
                    double x = block.X * scale + offsetX;
                    double y = block.Y * scale + offsetY;
                    double w = block.Width * scale;
                    double h = block.Height * scale;
                    double fontSize = Math.Max(8, h * 0.72);

                    var textBox = new TextBox
                    {
                        Text = block.Text,
                        FontSize = fontSize,
                        Background = new SolidColorBrush(Color.FromArgb(120, 255, 255, 100)),
                        BorderBrush = new SolidColorBrush(Color.FromArgb(200, 255, 165, 0)),
                        BorderThickness = new Thickness(1),
                        Foreground = new SolidColorBrush(Colors.Black),
                        Padding = new Thickness(2),
                        MinWidth = w,
                        Height = h + 6,
                        Tag = fontSize
                    };

                    Canvas.SetLeft(textBox, x);
                    Canvas.SetTop(textBox, y);
                    AnnotationCanvas.Children.Add(textBox);
                    _ocrOverlays.Add(textBox);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"OCR 오류: {ex.Message}", "오류", MessageBoxButton.OK, MessageBoxImage.Error);
            }
            finally
            {
                button.Content = originalContent;
                button.IsEnabled = true;
            }
        }

        private void ApplyOcr_Click(object sender, RoutedEventArgs e)
        {
            foreach (var textBox in _ocrOverlays.ToList())
            {
                if (string.IsNullOrWhiteSpace(textBox.Text)) continue;

                var x = Canvas.GetLeft(textBox);
                var y = Canvas.GetTop(textBox);
                var fontSize = textBox.Tag is double fs ? fs : 14.0;

                var textBlock = new TextBlock
                {
                    Text = textBox.Text,
                    FontSize = fontSize,
                    Foreground = new SolidColorBrush(Colors.Black),
                    TextWrapping = TextWrapping.NoWrap
                };

                Canvas.SetLeft(textBlock, x);
                Canvas.SetTop(textBlock, y);
                AnnotationCanvas.Children.Remove(textBox);
                AnnotationCanvas.Children.Add(textBlock);

                _viewModel.Annotations.Add(new Annotation
                {
                    Type = "Text",
                    Page = _viewModel.CurrentPage,
                    X = x,
                    Y = y,
                    Text = textBox.Text,
                    Color = "#000000",
                    FontSize = fontSize
                });
            }
            _ocrOverlays.Clear();
        }

        private void ClearOcr_Click(object sender, RoutedEventArgs e) => ClearOcrOverlays();

        private void ClearOcrOverlays()
        {
            foreach (var tb in _ocrOverlays)
                AnnotationCanvas.Children.Remove(tb);
            _ocrOverlays.Clear();
        }

        private void Undo_Click(object sender, RoutedEventArgs e)
        {
            // 현재 페이지의 마지막 주석 제거
            var lastAnnotation = _viewModel.Annotations
                .Where(a => a.Page == _viewModel.CurrentPage)
                .LastOrDefault();

            if (lastAnnotation != null)
            {
                _viewModel.Annotations.Remove(lastAnnotation);

                // 캔버스 다시 그리기
                AnnotationCanvas.Children.Clear();
                foreach (var annotation in _viewModel.Annotations.Where(a => a.Page == _viewModel.CurrentPage))
                {
                    RenderAnnotation(annotation);
                }
            }
        }

        private void PreviousPage_Click(object sender, RoutedEventArgs e)
        {
            if (_viewModel.CurrentPage > 1)
            {
                ClearOcrOverlays();
                _viewModel.CurrentPage--;
                RenderCurrentPage();
            }
        }

        private void NextPage_Click(object sender, RoutedEventArgs e)
        {
            if (_viewModel.CurrentPage < _viewModel.TotalPages)
            {
                ClearOcrOverlays();
                _viewModel.CurrentPage++;
                RenderCurrentPage();
            }
        }

        private void RenderCurrentPage()
        {
            if (!_viewModel.IsPdfLoaded)
                return;

            var pageImage = _pdfService.RenderPage(_viewModel.CurrentPage - 1);
            PdfImageView.Source = pageImage;
            PageInfo.Text = $"{_viewModel.CurrentPage} / {_viewModel.TotalPages}";

            // 현재 페이지의 주석만 표시
            AnnotationCanvas.Children.Clear();
            foreach (var annotation in _viewModel.Annotations.Where(a => a.Page == _viewModel.CurrentPage))
            {
                RenderAnnotation(annotation);
            }
        }

        private void RenderAnnotation(Annotation annotation)
        {
            UIElement? element = null;

            switch (annotation.Type)
            {
                case "Text":
                    element = new TextBlock
                    {
                        Text = annotation.Text,
                        FontSize = annotation.FontSize ?? 16,
                        Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString(annotation.Color ?? "#000000")),
                        FontWeight = FontWeights.Bold
                    };
                    break;

                case "Rectangle":
                    element = new Rectangle
                    {
                        Width = annotation.Width ?? 100,
                        Height = annotation.Height ?? 100,
                        Stroke = new SolidColorBrush((Color)ColorConverter.ConvertFromString(annotation.Color ?? "#000000")),
                        StrokeThickness = 2
                    };
                    break;

                case "Circle":
                    element = new Ellipse
                    {
                        Width = annotation.Width ?? 100,
                        Height = annotation.Height ?? 100,
                        Stroke = new SolidColorBrush((Color)ColorConverter.ConvertFromString(annotation.Color ?? "#000000")),
                        StrokeThickness = 2
                    };
                    break;

                case "Signature":
                    element = new TextBlock
                    {
                        Text = annotation.Text,
                        FontSize = annotation.FontSize ?? 24,
                        Foreground = new SolidColorBrush((Color)ColorConverter.ConvertFromString(annotation.Color ?? "#000000")),
                        FontFamily = new FontFamily("Segoe Script"),
                        FontStyle = FontStyles.Italic
                    };
                    break;
            }

            if (element != null)
            {
                Canvas.SetLeft(element, annotation.X);
                Canvas.SetTop(element, annotation.Y);
                AnnotationCanvas.Children.Add(element);
            }
        }
    }
}
