package com.zhuchen.Service;

import com.zhuchen.project.Result;
import org.springframework.http.ResponseEntity;

public interface AuthService {
    public Result login(String userName, String password);
    public Result register(String userName, String password, String name);
    public Result verifyToken(String token);
}
