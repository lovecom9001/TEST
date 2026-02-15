package com.example.excelweb.service;

import com.itextpdf.kernel.colors.ColorConstants;
import com.itextpdf.kernel.geom.PageSize;
import com.itextpdf.kernel.pdf.PdfDocument;
import com.itextpdf.kernel.pdf.PdfWriter;
import com.itextpdf.layout.Document;
import com.itextpdf.layout.element.Cell;
import com.itextpdf.layout.element.Paragraph;
import com.itextpdf.layout.element.Table;
import com.itextpdf.layout.properties.TextAlignment;
import com.itextpdf.layout.properties.UnitValue;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.ss.util.CellReference;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.springframework.stereotype.Service;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Service
public class ExcelService {

    /**
     * 엑셀 파일에서 시트 목록 조회
     */
    public List<String> getSheetNames(InputStream inputStream) throws IOException {
        try (Workbook workbook = new XSSFWorkbook(inputStream)) {
            List<String> names = new ArrayList<>();
            for (int i = 0; i < workbook.getNumberOfSheets(); i++) {
                names.add(workbook.getSheetName(i));
            }
            return names;
        }
    }

    /**
     * 특정 시트의 데이터를 2차원 리스트로 반환
     */
    public Map<String, Object> getSheetData(byte[] fileBytes, String sheetName) throws IOException {
        try (Workbook workbook = new XSSFWorkbook(new java.io.ByteArrayInputStream(fileBytes))) {
            Sheet sheet = workbook.getSheet(sheetName);
            if (sheet == null) {
                throw new IllegalArgumentException("시트 '" + sheetName + "'이(가) 존재하지 않습니다.");
            }

            int lastRow = sheet.getLastRowNum();
            int maxCol = 0;
            for (int i = 0; i <= lastRow; i++) {
                Row row = sheet.getRow(i);
                if (row != null && row.getLastCellNum() > maxCol) {
                    maxCol = row.getLastCellNum();
                }
            }

            List<List<String>> rows = new ArrayList<>();
            for (int i = 0; i <= lastRow; i++) {
                Row row = sheet.getRow(i);
                List<String> rowData = new ArrayList<>();
                for (int j = 0; j < maxCol; j++) {
                    String value = "";
                    if (row != null) {
                        org.apache.poi.ss.usermodel.Cell cell = row.getCell(j);
                        if (cell != null) {
                            value = getCellValueAsString(cell);
                        }
                    }
                    rowData.add(value);
                }
                rows.add(rowData);
            }

            Map<String, Object> result = new LinkedHashMap<>();
            result.put("sheetName", sheetName);
            result.put("rows", rows);
            result.put("maxCol", maxCol);
            result.put("maxRow", lastRow + 1);
            return result;
        }
    }

    /**
     * 특정 셀에 데이터를 입력하고, 수정된 엑셀 바이트 배열 반환
     */
    public byte[] writeCells(byte[] fileBytes, String sheetName,
                             Map<String, String> cellData) throws IOException {
        try (Workbook workbook = new XSSFWorkbook(new java.io.ByteArrayInputStream(fileBytes))) {
            Sheet sheet = workbook.getSheet(sheetName);
            if (sheet == null) {
                throw new IllegalArgumentException("시트 '" + sheetName + "'이(가) 존재하지 않습니다.");
            }

            for (Map.Entry<String, String> entry : cellData.entrySet()) {
                CellReference ref = new CellReference(entry.getKey());
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

                String value = entry.getValue();
                try {
                    double numValue = Double.parseDouble(value);
                    cell.setCellValue(numValue);
                } catch (NumberFormatException e) {
                    cell.setCellValue(value);
                }
            }

            ByteArrayOutputStream bos = new ByteArrayOutputStream();
            workbook.write(bos);
            return bos.toByteArray();
        }
    }

    /**
     * 특정 시트를 PDF 바이트 배열로 변환
     */
    public byte[] sheetToPdf(byte[] fileBytes, String sheetName) throws IOException {
        try (Workbook workbook = new XSSFWorkbook(new java.io.ByteArrayInputStream(fileBytes))) {
            Sheet sheet = workbook.getSheet(sheetName);
            if (sheet == null) {
                throw new IllegalArgumentException("시트 '" + sheetName + "'이(가) 존재하지 않습니다.");
            }

            int lastRow = sheet.getLastRowNum();
            int maxCol = 0;
            for (int i = 0; i <= lastRow; i++) {
                Row row = sheet.getRow(i);
                if (row != null && row.getLastCellNum() > maxCol) {
                    maxCol = row.getLastCellNum();
                }
            }

            if (lastRow < 0 || maxCol == 0) {
                throw new IllegalStateException("시트에 데이터가 없습니다.");
            }

            ByteArrayOutputStream pdfBytes = new ByteArrayOutputStream();
            PdfWriter writer = new PdfWriter(pdfBytes);
            PdfDocument pdfDoc = new PdfDocument(writer);
            Document document = new Document(pdfDoc, PageSize.A4.rotate());
            document.setMargins(20, 20, 20, 20);

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
            return pdfBytes.toByteArray();
        }
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
}
