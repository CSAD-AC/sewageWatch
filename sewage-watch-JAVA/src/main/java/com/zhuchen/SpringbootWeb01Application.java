package com.zhuchen;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.security.servlet.SecurityAutoConfiguration;
import org.springframework.boot.web.servlet.ServletComponentScan;


@SpringBootApplication(exclude = {SecurityAutoConfiguration.class})
public class SpringbootWeb01Application {

    public static void main(String[] args) {
        SpringApplication.run(SpringbootWeb01Application.class, args);
    }

}
