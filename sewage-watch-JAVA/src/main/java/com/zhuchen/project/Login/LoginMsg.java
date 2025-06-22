package com.zhuchen.project.Login;

import com.zhuchen.project.UserInfo;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class LoginMsg {
    private String token;
    private UserInfo UserInfo;
}
