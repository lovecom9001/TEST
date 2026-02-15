package com.example.excelweb.controller;

import com.example.excelweb.service.ExcelService;
import jakarta.servlet.http.HttpSession;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Controller
public class ExcelController {

    private static final Logger log = LoggerFactory.getLogger(ExcelController.class);

    private final ExcelService excelService;

    public ExcelController(ExcelService excelService) {
        this.excelService = excelService;
    }

    /**
     * 메인 페이지
     */
    @GetMapping("/")
    public String index() {
        return "index";
    }

    /**
     * 1단계: 엑셀 파일 업로드 → 시트 목록 반환
     */
    @PostMapping("/upload")
    public String upload(@RequestParam("file") MultipartFile file,
                         HttpSession session, Model model) throws IOException {
        byte[] fileBytes = file.getBytes();
        String fileName = file.getOriginalFilename();

        // 세션에 파일 데이터 저장
        session.setAttribute("fileBytes", fileBytes);
        session.setAttribute("fileName", fileName);

        List<String> sheetNames = excelService.getSheetNames(file.getInputStream());
        model.addAttribute("sheetNames", sheetNames);
        model.addAttribute("fileName", fileName);

        return "sheets";
    }

    /**
     * 2단계: 시트 선택 → 셀 데이터 표시
     */
    @PostMapping("/sheet")
    public String selectSheet(@RequestParam("sheetName") String sheetName,
                              HttpSession session, Model model) {
        byte[] fileBytes = (byte[]) session.getAttribute("fileBytes");
        String fileName = (String) session.getAttribute("fileName");

        if (fileBytes == null) {
            return "redirect:/";
        }

        try {
            session.setAttribute("selectedSheet", sheetName);
            Map<String, Object> sheetData = excelService.getSheetData(fileBytes, sheetName);

            model.addAttribute("sheetData", sheetData);
            model.addAttribute("fileName", fileName);
            model.addAttribute("sheetName", sheetName);

            return "editor";
        } catch (Exception e) {
            log.error("시트 로딩 실패: {}", e.getMessage(), e);
            model.addAttribute("error", "시트를 불러오는 중 오류가 발생했습니다: " + e.getMessage());
            // 시트 목록 페이지로 돌아가기
            try {
                List<String> sheetNames = excelService.getSheetNames(
                        new java.io.ByteArrayInputStream(fileBytes));
                model.addAttribute("sheetNames", sheetNames);
                model.addAttribute("fileName", fileName);
            } catch (IOException ex) {
                return "redirect:/";
            }
            return "sheets";
        }
    }

    /**
     * 3단계: 셀 데이터 수정 → 수정된 시트 표시
     */
    @PostMapping("/edit")
    public String editCells(@RequestParam Map<String, String> allParams,
                            HttpSession session, Model model) throws IOException {
        byte[] fileBytes = (byte[]) session.getAttribute("fileBytes");
        String sheetName = (String) session.getAttribute("selectedSheet");
        String fileName = (String) session.getAttribute("fileName");

        if (fileBytes == null || sheetName == null) {
            return "redirect:/";
        }

        // "cell_" 접두사로 시작하는 파라미터만 추출
        Map<String, String> cellData = new LinkedHashMap<>();
        for (Map.Entry<String, String> entry : allParams.entrySet()) {
            if (entry.getKey().startsWith("cell_") && !entry.getValue().isEmpty()) {
                String cellAddress = entry.getKey().substring(5); // "cell_A1" → "A1"
                cellData.put(cellAddress, entry.getValue());
            }
        }

        if (!cellData.isEmpty()) {
            fileBytes = excelService.writeCells(fileBytes, sheetName, cellData);
            session.setAttribute("fileBytes", fileBytes);
        }

        Map<String, Object> sheetData = excelService.getSheetData(fileBytes, sheetName);
        model.addAttribute("sheetData", sheetData);
        model.addAttribute("fileName", fileName);
        model.addAttribute("sheetName", sheetName);
        model.addAttribute("message", "셀 데이터가 수정되었습니다. (" + cellData.size() + "개 셀)");

        return "editor";
    }

    /**
     * 4단계: PDF 다운로드
     */
    @PostMapping("/download-pdf")
    public ResponseEntity<byte[]> downloadPdf(HttpSession session) throws IOException {
        byte[] fileBytes = (byte[]) session.getAttribute("fileBytes");
        String sheetName = (String) session.getAttribute("selectedSheet");
        String fileName = (String) session.getAttribute("fileName");

        if (fileBytes == null || sheetName == null) {
            return ResponseEntity.badRequest().build();
        }

        byte[] pdfBytes = excelService.sheetToPdf(fileBytes, sheetName);

        String pdfFileName = fileName != null
                ? fileName.replaceAll("\\.[^.]+$", "") + "_" + sheetName + ".pdf"
                : "output.pdf";

        return ResponseEntity.ok()
                .header(HttpHeaders.CONTENT_DISPOSITION,
                        "attachment; filename=\"" + pdfFileName + "\"")
                .contentType(MediaType.APPLICATION_PDF)
                .body(pdfBytes);
    }

    /**
     * 수정된 엑셀 파일 다운로드
     */
    @PostMapping("/download-excel")
    public ResponseEntity<byte[]> downloadExcel(HttpSession session) {
        byte[] fileBytes = (byte[]) session.getAttribute("fileBytes");
        String fileName = (String) session.getAttribute("fileName");

        if (fileBytes == null) {
            return ResponseEntity.badRequest().build();
        }

        return ResponseEntity.ok()
                .header(HttpHeaders.CONTENT_DISPOSITION,
                        "attachment; filename=\"" + (fileName != null ? fileName : "output.xlsx") + "\"")
                .contentType(MediaType.parseMediaType(
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))
                .body(fileBytes);
    }
}
