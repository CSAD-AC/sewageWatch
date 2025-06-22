package com.zhuchen.project;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@AllArgsConstructor
@NoArgsConstructor
@Data
public class UserInfo {
    private String userName;
    private String userRole;
    private Integer userId;
}
