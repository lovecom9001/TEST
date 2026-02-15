package com.example.excelweb.service;

import com.itextpdf.kernel.colors.DeviceRgb;
import com.itextpdf.kernel.font.PdfFont;
import com.itextpdf.kernel.font.PdfFontFactory;
import com.itextpdf.kernel.geom.PageSize;
import com.itextpdf.kernel.pdf.PdfDocument;
import com.itextpdf.kernel.pdf.PdfWriter;
import com.itextpdf.layout.Document;
import com.itextpdf.layout.borders.Border;
import com.itextpdf.layout.borders.SolidBorder;
import com.itextpdf.layout.element.Cell;
import com.itextpdf.layout.element.Paragraph;
import com.itextpdf.layout.element.Table;
import com.itextpdf.layout.properties.TextAlignment;
import com.itextpdf.layout.properties.UnitValue;
import com.itextpdf.layout.properties.VerticalAlignment;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.ss.util.CellRangeAddress;
import org.apache.poi.ss.util.CellReference;
import org.apache.poi.xssf.usermodel.XSSFColor;
import org.springframework.stereotype.Service;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.*;

@Service
public class ExcelService {

    /**
     * 컬럼 인덱스를 엑셀 컬럼명으로 변환 (0→A, 25→Z, 26→AA, 27→AB ...)
     */
    public static String columnIndexToLetter(int index) {
        StringBuilder sb = new StringBuilder();
        while (index >= 0) {
            sb.insert(0, (char) ('A' + (index % 26)));
            index = index / 26 - 1;
        }
        return sb.toString();
    }

    /**
     * 엑셀 파일에서 시트 목록 조회
     */
    public List<String> getSheetNames(InputStream inputStream) throws IOException {
        try (Workbook workbook = WorkbookFactory.create(inputStream)) {
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
        try (Workbook workbook = WorkbookFactory.create(new java.io.ByteArrayInputStream(fileBytes))) {
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

            // 컬럼 헤더 문자열 목록 생성 (A, B, ... Z, AA, AB ...)
            List<String> colHeaders = new ArrayList<>();
            for (int j = 0; j < maxCol; j++) {
                colHeaders.add(columnIndexToLetter(j));
            }

            Map<String, Object> result = new LinkedHashMap<>();
            result.put("sheetName", sheetName);
            result.put("rows", rows);
            result.put("colHeaders", colHeaders);
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
        try (Workbook workbook = WorkbookFactory.create(new java.io.ByteArrayInputStream(fileBytes))) {
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
     * - 한글 폰트 지원
     * - 병합 셀 지원
     * - 실제 셀 스타일(정렬, 배경색, 볼드, 테두리) 반영
     */
    public byte[] sheetToPdf(byte[] fileBytes, String sheetName) throws IOException {
        try (Workbook workbook = WorkbookFactory.create(new java.io.ByteArrayInputStream(fileBytes))) {
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

            // ── 병합 셀 정보 수집 ──
            Map<String, int[]> mergeStartMap = new HashMap<>();  // "row,col" → [rowSpan, colSpan]
            Set<String> coveredCells = new HashSet<>();
            for (int i = 0; i < sheet.getNumMergedRegions(); i++) {
                CellRangeAddress region = sheet.getMergedRegion(i);
                int fr = region.getFirstRow(), lr = region.getLastRow();
                int fc = region.getFirstColumn(), lc = region.getLastColumn();
                mergeStartMap.put(fr + "," + fc, new int[]{lr - fr + 1, lc - fc + 1});
                for (int r = fr; r <= lr; r++) {
                    for (int c = fc; c <= lc; c++) {
                        if (r != fr || c != fc) coveredCells.add(r + "," + c);
                    }
                }
            }

            // ── 컬럼 너비 비율 계산 ──
            float[] colWidths = new float[maxCol];
            for (int i = 0; i < maxCol; i++) {
                colWidths[i] = Math.max(sheet.getColumnWidth(i) / 256f, 2f);
            }

            // ── 한글 폰트 ──
            PdfFont font;
            try {
                font = PdfFontFactory.createFont("HYGoThic-Medium", "UniKS-UCS2-H");
            } catch (Exception e) {
                font = PdfFontFactory.createFont();
            }

            // ── PDF 문서 생성 ──
            ByteArrayOutputStream pdfBytes = new ByteArrayOutputStream();
            PdfWriter writer = new PdfWriter(pdfBytes);
            PdfDocument pdfDoc = new PdfDocument(writer);
            Document document = new Document(pdfDoc, PageSize.A4);
            document.setMargins(30, 30, 30, 30);
            document.setFont(font);
            document.setFontSize(9);

            Table table = new Table(UnitValue.createPercentArray(colWidths))
                    .useAllAvailableWidth();

            for (int rowIdx = 0; rowIdx <= lastRow; rowIdx++) {
                Row row = sheet.getRow(rowIdx);
                for (int colIdx = 0; colIdx < maxCol; colIdx++) {
                    String key = rowIdx + "," + colIdx;

                    // 병합 셀에 의해 가려지는 셀은 건너뛰기
                    if (coveredCells.contains(key)) continue;

                    String cellValue = "";
                    org.apache.poi.ss.usermodel.Cell excelCell = null;
                    if (row != null) {
                        excelCell = row.getCell(colIdx);
                        if (excelCell != null) {
                            cellValue = getCellValueAsString(excelCell);
                        }
                    }

                    // 병합 셀이면 rowSpan/colSpan 적용
                    int[] spans = mergeStartMap.get(key);
                    Cell pdfCell = (spans != null) ? new Cell(spans[0], spans[1]) : new Cell();

                    pdfCell.add(new Paragraph(cellValue != null ? cellValue : ""))
                           .setPadding(4)
                           .setFontSize(9)
                           .setBorder(new SolidBorder(new DeviceRgb(180, 180, 180), 0.5f));

                    // 엑셀 셀 스타일 반영
                    if (excelCell != null) {
                        applyCellStyle(pdfCell, excelCell, workbook, font);
                    }

                    table.addCell(pdfCell);
                }
            }

            document.add(table);
            document.close();
            return pdfBytes.toByteArray();
        }
    }

    /**
     * 엑셀 셀 스타일을 PDF 셀에 적용
     */
    private void applyCellStyle(Cell pdfCell, org.apache.poi.ss.usermodel.Cell excelCell,
                                Workbook workbook, PdfFont font) {
        CellStyle style = excelCell.getCellStyle();

        // 텍스트 정렬
        switch (style.getAlignment()) {
            case CENTER: pdfCell.setTextAlignment(TextAlignment.CENTER); break;
            case RIGHT: pdfCell.setTextAlignment(TextAlignment.RIGHT); break;
            default: pdfCell.setTextAlignment(TextAlignment.LEFT); break;
        }

        // 세로 정렬
        switch (style.getVerticalAlignment()) {
            case CENTER: pdfCell.setVerticalAlignment(VerticalAlignment.MIDDLE); break;
            case BOTTOM: pdfCell.setVerticalAlignment(VerticalAlignment.BOTTOM); break;
            default: pdfCell.setVerticalAlignment(VerticalAlignment.TOP); break;
        }

        // 볼드, 폰트 크기
        Font excelFont = workbook.getFontAt(style.getFontIndexAsInt());
        if (excelFont.getBold()) {
            pdfCell.setBold();
        }
        if (excelFont.getFontHeightInPoints() > 0) {
            pdfCell.setFontSize(Math.min(excelFont.getFontHeightInPoints(), 14));
        }

        // 폰트 색상
        DeviceRgb fontColor = getFontColor(excelFont, workbook);
        if (fontColor != null) {
            pdfCell.setFontColor(fontColor);
        }

        // 배경색
        if (style.getFillPattern() == FillPatternType.SOLID_FOREGROUND) {
            DeviceRgb bgColor = poiColorToDeviceRgb(style.getFillForegroundColorColor());
            if (bgColor != null) {
                pdfCell.setBackgroundColor(bgColor);
            }
        }

        // 테두리 (엑셀에 테두리가 있으면 검은 실선 적용)
        if (style.getBorderTop() != BorderStyle.NONE ||
            style.getBorderBottom() != BorderStyle.NONE ||
            style.getBorderLeft() != BorderStyle.NONE ||
            style.getBorderRight() != BorderStyle.NONE) {
            Border border = new SolidBorder(new DeviceRgb(0, 0, 0), 0.5f);
            if (style.getBorderTop() != BorderStyle.NONE) pdfCell.setBorderTop(border);
            if (style.getBorderBottom() != BorderStyle.NONE) pdfCell.setBorderBottom(border);
            if (style.getBorderLeft() != BorderStyle.NONE) pdfCell.setBorderLeft(border);
            if (style.getBorderRight() != BorderStyle.NONE) pdfCell.setBorderRight(border);
        }
    }

    /**
     * POI Color → iText DeviceRgb 변환
     */
    private DeviceRgb poiColorToDeviceRgb(org.apache.poi.ss.usermodel.Color color) {
        if (color == null) return null;
        try {
            if (color instanceof XSSFColor) {
                byte[] rgb = ((XSSFColor) color).getRGB();
                if (rgb != null) {
                    int offset = rgb.length == 4 ? 1 : 0;
                    return new DeviceRgb(rgb[offset] & 0xFF, rgb[offset + 1] & 0xFF, rgb[offset + 2] & 0xFF);
                }
            }
        } catch (Exception ignored) {}
        return null;
    }

    /**
     * 폰트 색상 추출
     */
    private DeviceRgb getFontColor(Font excelFont, Workbook workbook) {
        try {
            if (excelFont instanceof org.apache.poi.xssf.usermodel.XSSFFont) {
                XSSFColor xssfColor = ((org.apache.poi.xssf.usermodel.XSSFFont) excelFont).getXSSFColor();
                if (xssfColor != null) {
                    byte[] rgb = xssfColor.getRGB();
                    if (rgb != null) {
                        int offset = rgb.length == 4 ? 1 : 0;
                        int r = rgb[offset] & 0xFF, g = rgb[offset + 1] & 0xFF, b = rgb[offset + 2] & 0xFF;
                        if (r == 0 && g == 0 && b == 0) return null; // 검정은 기본값이므로 생략
                        return new DeviceRgb(r, g, b);
                    }
                }
            }
        } catch (Exception ignored) {}
        return null;
    }

    private String getCellValueAsString(org.apache.poi.ss.usermodel.Cell cell) {
        try {
            switch (cell.getCellType()) {
                case STRING:
                    return cell.getStringCellValue();
                case NUMERIC:
                    if (DateUtil.isCellDateFormatted(cell)) {
                        return cell.getLocalDateTimeCellValue().toString();
                    }
                    double num = cell.getNumericCellValue();
                    if (num == Math.floor(num) && !Double.isInfinite(num)) {
                        return String.valueOf((long) num);
                    }
                    return String.valueOf(num);
                case BOOLEAN:
                    return String.valueOf(cell.getBooleanCellValue());
                case FORMULA:
                    try {
                        // 수식 결과값을 가져오기 시도
                        CellType cachedType = cell.getCachedFormulaResultType();
                        if (cachedType == CellType.NUMERIC) {
                            double fnum = cell.getNumericCellValue();
                            if (fnum == Math.floor(fnum) && !Double.isInfinite(fnum)) {
                                return String.valueOf((long) fnum);
                            }
                            return String.valueOf(fnum);
                        } else if (cachedType == CellType.STRING) {
                            return cell.getStringCellValue();
                        } else if (cachedType == CellType.BOOLEAN) {
                            return String.valueOf(cell.getBooleanCellValue());
                        }
                        return cell.getCellFormula();
                    } catch (Exception e) {
                        return cell.getCellFormula();
                    }
                case ERROR:
                    return "#ERROR";
                default:
                    return "";
            }
        } catch (Exception e) {
            return "";
        }
    }
}
