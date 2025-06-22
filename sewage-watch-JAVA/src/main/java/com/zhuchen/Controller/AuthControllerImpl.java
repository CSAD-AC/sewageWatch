package com.zhuchen.Controller;

import com.zhuchen.Service.AuthService;
import com.zhuchen.project.Result;
import com.zhuchen.project.User;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.context.request.RequestAttributes;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

@Slf4j
@RestController
public class AuthControllerImpl {
    private AuthService AuthService;
    @Autowired
    public void setAuthService(AuthService AuthService) {
        this.AuthService = AuthService;
    }

    @PostMapping("/login")
    public Result login(@RequestBody User user) {
        log.info("尝试登录： 用户名：{}，密码：{}", user.getUserName(), user.getPassword());
        return AuthService.login(user.getUserName(), user.getPassword());
    }
    @PostMapping("/register")
    public Result register(@RequestBody User user) {
        log.info("尝试注册： 用户名：{}，密码：{} 昵称: {}", user.getUserName(),
                user.getPassword(),
                user.getName());
        return AuthService.register(user.getUserName(),
                user.getPassword(),
                user.getName());
    }

    @PostMapping("/token")
    public Result verifyToken() {
        log.info("尝试token验证");
        String token = null;
        RequestAttributes requestAttributes = RequestContextHolder.getRequestAttributes();
        if (requestAttributes instanceof ServletRequestAttributes) {
            String tokenFromHeader = ((ServletRequestAttributes) requestAttributes).getRequest().getHeader("Authorization");
            if (tokenFromHeader != null && tokenFromHeader.startsWith("Bearer ")) {
                token = tokenFromHeader.substring(7); // 去掉 "Bearer " 前缀
            }
        }
        return AuthService.verifyToken(token);
    }
}
