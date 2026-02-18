namespace PDFEditor.Models
{
    public class Annotation
    {
        public string Type { get; set; } = string.Empty;
        public int Page { get; set; }
        public double X { get; set; }
        public double Y { get; set; }
        public double? Width { get; set; }
        public double? Height { get; set; }
        public string? Text { get; set; }
        public string? Color { get; set; }
        public double? FontSize { get; set; }
    }
}
