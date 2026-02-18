using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Runtime.CompilerServices;
using PDFEditor.Models;

namespace PDFEditor
{
    public class MainViewModel : INotifyPropertyChanged
    {
        private bool _isPdfLoaded;
        private int _currentPage = 1;
        private int _totalPages = 1;
        private string? _currentPdfPath;

        public bool IsPdfLoaded
        {
            get => _isPdfLoaded;
            set
            {
                _isPdfLoaded = value;
                OnPropertyChanged();
                OnPropertyChanged(nameof(CanGoPrevious));
                OnPropertyChanged(nameof(CanGoNext));
            }
        }

        public int CurrentPage
        {
            get => _currentPage;
            set
            {
                _currentPage = value;
                OnPropertyChanged();
                OnPropertyChanged(nameof(CanGoPrevious));
                OnPropertyChanged(nameof(CanGoNext));
            }
        }

        public int TotalPages
        {
            get => _totalPages;
            set
            {
                _totalPages = value;
                OnPropertyChanged();
                OnPropertyChanged(nameof(CanGoPrevious));
                OnPropertyChanged(nameof(CanGoNext));
            }
        }

        public string? CurrentPdfPath
        {
            get => _currentPdfPath;
            set
            {
                _currentPdfPath = value;
                OnPropertyChanged();
            }
        }

        public bool CanGoPrevious => IsPdfLoaded && CurrentPage > 1;
        public bool CanGoNext => IsPdfLoaded && CurrentPage < TotalPages;

        public ObservableCollection<Annotation> Annotations { get; } = new();

        public event PropertyChangedEventHandler? PropertyChanged;

        protected virtual void OnPropertyChanged([CallerMemberName] string? propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }
    }
}
