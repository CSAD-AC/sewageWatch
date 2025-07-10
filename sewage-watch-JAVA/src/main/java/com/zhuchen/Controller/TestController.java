package com.zhuchen.Controller;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.ResponseBody;

@Slf4j
@Controller
@RequestMapping("/hello")
public class TestController {
    @GetMapping
    @ResponseBody
    public String hello() {
        log.info("hello world");
        return "hello world"; // 返回字符串作为响应
    }
}