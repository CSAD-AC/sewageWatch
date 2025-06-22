package com.zhuchen.project;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

/**
 * 用户信息
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class User {
    private Integer id;
    private String userName;
    private String password;
    private String name;
    private String role;
    private boolean enabled = true;
    private LocalDateTime updateTime;
}
