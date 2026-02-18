using System;
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
        private string? _currentTool = "Select";
        private string _selectedColor = "#000000";

        public MainWindow()
        {
            InitializeComponent();
            _viewModel = (MainViewModel)DataContext;
            _pdfService = new PdfService();
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

        private void PreviousPage_Click(object sender, RoutedEventArgs e)
        {
            if (_viewModel.CurrentPage > 1)
            {
                _viewModel.CurrentPage--;
                RenderCurrentPage();
            }
        }

        private void NextPage_Click(object sender, RoutedEventArgs e)
        {
            if (_viewModel.CurrentPage < _viewModel.TotalPages)
            {
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
