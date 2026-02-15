package com.example;

import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;

import com.itextpdf.kernel.pdf.PdfDocument;
import com.itextpdf.kernel.pdf.PdfWriter;
import com.itextpdf.kernel.geom.PageSize;
import com.itextpdf.kernel.colors.ColorConstants;
import com.itextpdf.layout.Document;
import com.itextpdf.layout.element.Cell;
import com.itextpdf.layout.element.Paragraph;
import com.itextpdf.layout.element.Table;
import com.itextpdf.layout.properties.TextAlignment;
import com.itextpdf.layout.properties.UnitValue;

import java.io.*;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 엑셀 파일 자동화 클래스
 * - 엑셀 파일 열기
 * - 특정 시트 불러오기
 * - 특정 셀에 데이터 입력
 * - 특정 시트를 PDF로 저장
 */
public class ExcelToPdf {

    private Workbook workbook;
    private String filePath;

    // ──────────────────────────────────────────────
    // 1. 엑셀 파일 열기
    // ──────────────────────────────────────────────
    public Workbook openWorkbook(String filePath) throws IOException {
        this.filePath = filePath;
        File file = new File(filePath);

        if (!file.exists()) {
            System.out.println("파일이 없어 새로 생성합니다: " + filePath);
            this.workbook = new XSSFWorkbook();
            this.workbook.createSheet("Sheet1");
            try (FileOutputStream fos = new FileOutputStream(filePath)) {
                this.workbook.write(fos);
            }
            // 다시 열기 (읽기/쓰기 모드)
            this.workbook.close();
        }

        FileInputStream fis = new FileInputStream(filePath);
        this.workbook = new XSSFWorkbook(fis);
        fis.close();

        System.out.println("엑셀 파일 열기 완료: " + filePath);
        System.out.print("시트 목록: [");
        for (int i = 0; i < workbook.getNumberOfSheets(); i++) {
            if (i > 0) System.out.print(", ");
            System.out.print(workbook.getSheetName(i));
        }
        System.out.println("]");

        return this.workbook;
    }

    // ──────────────────────────────────────────────
    // 2. 특정 시트 불러오기
    // ──────────────────────────────────────────────
    public Sheet getSheet(String sheetName) {
        Sheet sheet = workbook.getSheet(sheetName);
        if (sheet == null) {
            throw new IllegalArgumentException(
                "시트 '" + sheetName + "'이(가) 존재하지 않습니다.");
        }
        System.out.println("시트 불러오기 완료: " + sheetName);
        return sheet;
    }

    // ──────────────────────────────────────────────
    // 3. 특정 셀에 데이터 입력
    // ──────────────────────────────────────────────
    public void writeCell(Sheet sheet, String cellAddress, Object value) {
        // "A1" -> row=0, col=0  /  "B3" -> row=2, col=1
        org.apache.poi.ss.util.CellReference ref =
                new org.apache.poi.ss.util.CellReference(cellAddress);
        int rowIdx = ref.getRow();
        int colIdx = ref.getCol();

        Row row = sheet.getRow(rowIdx);
        if (row == null) {
            row = sheet.createRow(rowIdx);
        }
        org.apache.poi.ss.usermodel.Cell cell = row.getCell(colIdx);
        if (cell == null) {
            cell = row.createCell(colIdx);
        }

        if (value instanceof Number) {
            cell.setCellValue(((Number) value).doubleValue());
        } else {
            cell.setCellValue(String.valueOf(value));
        }
        System.out.println("셀 " + cellAddress + "에 데이터 입력 완료: " + value);
    }

    public void writeCells(Sheet sheet, Map<String, Object> data) {
        for (Map.Entry<String, Object> entry : data.entrySet()) {
            writeCell(sheet, entry.getKey(), entry.getValue());
        }
    }

    // ──────────────────────────────────────────────
    // 4. 엑셀 파일 저장
    // ──────────────────────────────────────────────
    public void saveWorkbook(String outputPath) throws IOException {
        try (FileOutputStream fos = new FileOutputStream(outputPath)) {
            workbook.write(fos);
        }
        System.out.println("엑셀 파일 저장 완료: " + outputPath);
    }

    public void saveWorkbook() throws IOException {
        saveWorkbook(this.filePath);
    }

    // ──────────────────────────────────────────────
    // 5. 시트를 PDF로 저장
    // ──────────────────────────────────────────────
    public void sheetToPdf(Sheet sheet, String pdfPath) throws IOException {
        int lastRow = sheet.getLastRowNum();
        if (lastRow < 0) {
            System.out.println("시트에 데이터가 없습니다.");
            return;
        }

        // 최대 컬럼 수 계산
        int maxCol = 0;
        for (int i = 0; i <= lastRow; i++) {
            Row row = sheet.getRow(i);
            if (row != null && row.getLastCellNum() > maxCol) {
                maxCol = row.getLastCellNum();
            }
        }

        // PDF 생성 (가로 방향)
        PdfWriter writer = new PdfWriter(pdfPath);
        PdfDocument pdfDoc = new PdfDocument(writer);
        Document document = new Document(pdfDoc, PageSize.A4.rotate());
        document.setMargins(20, 20, 20, 20);

        // 테이블 생성
        Table table = new Table(UnitValue.createPercentArray(maxCol))
                .useAllAvailableWidth();

        for (int rowIdx = 0; rowIdx <= lastRow; rowIdx++) {
            Row row = sheet.getRow(rowIdx);
            for (int colIdx = 0; colIdx < maxCol; colIdx++) {
                String cellValue = "";
                if (row != null) {
                    org.apache.poi.ss.usermodel.Cell cell = row.getCell(colIdx);
                    if (cell != null) {
                        cellValue = getCellValueAsString(cell);
                    }
                }

                Cell pdfCell = new Cell()
                        .add(new Paragraph(cellValue))
                        .setTextAlignment(TextAlignment.CENTER)
                        .setPadding(5);

                // 헤더 행 스타일
                if (rowIdx == 0) {
                    pdfCell.setBackgroundColor(ColorConstants.GRAY)
                            .setFontColor(ColorConstants.WHITE)
                            .setBold();
                } else if (rowIdx % 2 == 0) {
                    pdfCell.setBackgroundColor(ColorConstants.LIGHT_GRAY);
                }

                table.addCell(pdfCell);
            }
        }

        document.add(table);
        document.close();
        System.out.println("PDF 저장 완료: " + pdfPath);
    }

    private String getCellValueAsString(org.apache.poi.ss.usermodel.Cell cell) {
        switch (cell.getCellType()) {
            case STRING:
                return cell.getStringCellValue();
            case NUMERIC:
                double num = cell.getNumericCellValue();
                if (num == Math.floor(num)) {
                    return String.valueOf((long) num);
                }
                return String.valueOf(num);
            case BOOLEAN:
                return String.valueOf(cell.getBooleanCellValue());
            case FORMULA:
                return cell.getCellFormula();
            default:
                return "";
        }
    }

    public void close() throws IOException {
        if (workbook != null) {
            workbook.close();
        }
    }

    // ──────────────────────────────────────────────
    // 사용 예시 (직접 실행 시)
    // ──────────────────────────────────────────────
    public static void main(String[] args) {
        // ── 설정값 (필요에 따라 수정) ──────────────
        String excelFile = "sample.xlsx";    // 엑셀 파일 경로
        String sheetName = "Sheet1";         // 작업할 시트 이름
        String pdfOutput = "output.pdf";     // PDF 저장 경로

        // 입력할 데이터 (셀 주소 -> 값)
        Map<String, Object> cellData = new LinkedHashMap<>();
        cellData.put("A1", "Name");
        cellData.put("B1", "Age");
        cellData.put("C1", "Email");
        cellData.put("A2", "Hong Gildong");
        cellData.put("B2", 30);
        cellData.put("C2", "hong@example.com");
        cellData.put("A3", "Kim Cheolsu");
        cellData.put("B3", 25);
        cellData.put("C3", "kim@example.com");
        // ──────────────────────────────────────────

        ExcelToPdf app = new ExcelToPdf();
        try {
            // 1. 엑셀 파일 열기
            app.openWorkbook(excelFile);

            // 2. 특정 시트 불러오기
            Sheet sheet = app.getSheet(sheetName);

            // 3. 특정 셀에 데이터 입력
            app.writeCells(sheet, cellData);

            // 4. 엑셀 파일 저장
            app.saveWorkbook();

            // 5. 시트를 PDF로 저장
            app.sheetToPdf(sheet, pdfOutput);

            app.close();
            System.out.println("\n모든 작업이 완료되었습니다!");

        } catch (Exception e) {
            System.err.println("오류 발생: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
