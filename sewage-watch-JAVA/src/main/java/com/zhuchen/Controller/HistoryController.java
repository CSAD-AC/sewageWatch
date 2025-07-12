package com.zhuchen.Controller;

import com.zhuchen.Service.HistoryService;
import com.zhuchen.project.History;
import com.zhuchen.project.Result;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@Slf4j
@RequestMapping("/history")
public class HistoryController {
    HistoryService historyService;
    @Autowired
    public void setHistoryService(HistoryService historyService) {
        this.historyService = historyService;
    }
    @GetMapping
    public Result getAllHistory() {
        return historyService.findAllHistory();
    }
}
