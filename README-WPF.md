# PDF 편집기 - C# WPF

Windows 데스크톱용 PDF 편집 애플리케이션입니다.

## 기능

- ✅ PDF 파일 열기 및 보기
- ✅ 텍스트 추가
- ✅ 도형 그리기 (사각형, 원)
- ✅ 서명 추가
- ✅ 페이지 탐색
- ✅ 편집된 PDF 저장

## 기술 스택

- .NET 8.0
- WPF (Windows Presentation Foundation)
- XAML
- PdfiumViewer - PDF 렌더링
- iText7 - PDF 편집 및 저장
- MVVM 패턴

## 요구사항

- Windows 10 이상
- .NET 8.0 SDK

## 설치 및 실행

### 1. .NET SDK 설치

https://dotnet.microsoft.com/download 에서 .NET 8.0 SDK를 다운로드 및 설치하세요.

### 2. 프로젝트 빌드

```bash
dotnet restore
dotnet build
```

### 3. 실행

```bash
dotnet run
```

또는 Visual Studio에서 프로젝트를 열고 F5를 눌러 실행하세요.

## 사용 방법

1. **PDF 열기**: 상단의 "📁 PDF 열기" 버튼을 클릭하여 PDF 파일을 선택합니다.
2. **도구 선택**: 왼쪽 사이드바에서 원하는 편집 도구를 선택합니다.
   - 🖱️ 선택: 기본 선택 모드
   - ✏️ 텍스트: 텍스트 추가
   - 🖼️ 이미지: 이미지 추가 (예정)
   - ▭ 사각형: 사각형 그리기
   - ○ 원: 원 그리기
   - ✍️ 서명: 서명 추가
3. **편집**: PDF 페이지를 클릭하여 주석을 추가합니다.
4. **저장**: "💾 저장" 버튼을 클릭하여 편집된 PDF를 저장합니다.

## 프로젝트 구조

```
PDFEditor/
├── Models/
│   └── Annotation.cs          # 주석 데이터 모델
├── Services/
│   └── PdfService.cs          # PDF 처리 서비스
├── Converters/
│   └── BoolToVisibilityConverter.cs  # XAML 컨버터
├── MainWindow.xaml            # 메인 윈도우 UI
├── MainWindow.xaml.cs         # 메인 윈도우 로직
├── MainViewModel.cs           # 뷰모델
├── TextInputDialog.xaml       # 텍스트 입력 대화상자
├── App.xaml                   # 애플리케이션 리소스
└── PDFEditor.csproj          # 프로젝트 파일
```

## NuGet 패키지

- **PdfiumViewer**: PDF 렌더링 및 표시
- **itext7**: PDF 생성 및 편집
- **itext7.bouncy-castle-adapter**: iText7 암호화 지원
- **Microsoft.Xaml.Behaviors.Wpf**: WPF 동작 및 인터랙션

## 라이선스

MIT

## 문제 해결

### PDF가 표시되지 않는 경우

- .NET 8.0 SDK가 설치되어 있는지 확인하세요.
- `dotnet restore`를 실행하여 NuGet 패키지를 복원하세요.

### 빌드 오류가 발생하는 경우

```bash
dotnet clean
dotnet restore
dotnet build
```

위 명령어를 순서대로 실행하세요.
