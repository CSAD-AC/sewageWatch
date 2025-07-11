package com.zhuchen.Exception;

import com.zhuchen.project.Result;
import lombok.extern.slf4j.Slf4j;
import org.springframework.dao.DuplicateKeyException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@Slf4j
@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler
    public Result exceptionHandler(Exception e) {
        log.error("服务器异常：",e);
        return Result.error("服务器异常");
    }


}
