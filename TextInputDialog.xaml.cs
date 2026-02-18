using System.Windows;

namespace PDFEditor
{
    public partial class TextInputDialog : Window
    {
        public string InputText => InputTextBox.Text;

        public TextInputDialog()
        {
            InitializeComponent();
            InputTextBox.Focus();
        }

        private void OK_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = true;
            Close();
        }

        private void Cancel_Click(object sender, RoutedEventArgs e)
        {
            DialogResult = false;
            Close();
        }
    }
}
