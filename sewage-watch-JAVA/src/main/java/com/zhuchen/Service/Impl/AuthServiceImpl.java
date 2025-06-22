package com.zhuchen.Service.Impl;

import com.zhuchen.Dao.UserDao;
import com.zhuchen.Service.AuthService;
import com.zhuchen.Utils.JwtUtil;
import com.zhuchen.project.Login.LoginMsg;
import com.zhuchen.project.User;
import com.zhuchen.project.UserInfo;
import com.zhuchen.project.Result;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import java.util.HashMap;
import java.util.List;
import java.util.Map;


@Slf4j
@Service
public class AuthServiceImpl implements AuthService {
    private final UserDao userDao;
    private final BCryptPasswordEncoder passwordEncoder = new BCryptPasswordEncoder();

    @Autowired
    public AuthServiceImpl(UserDao userDao) {
        this.userDao = userDao;
    }

    @Transactional(rollbackFor = Exception.class, propagation = Propagation.REQUIRES_NEW)
    @Override
    public Result login(String userName, String password) {
        List<User> users = userDao.findUserByUsername(userName);
        if(!users.isEmpty()) {
            User user = users.get(0);
            if(!user.isEnabled()) {
                log.info("用户{}被禁用", user.getUserName());
                return Result.error(401, "用户被禁用");
            }
            if(passwordEncoder.matches(password, user.getPassword())) {
                log.info("密码校验成功");
                UserInfo UserInfo = new UserInfo(user.getUserName(), user.getRole(), user.getId());
                // 生成token
                Map<String, Object> claims = new HashMap<>();
                claims.put("userName", user.getUserName());
                claims.put("role", user.getRole());
                claims.put("name", user.getName());
                claims.put("id", user.getId());
                claims.put("enabled", user.isEnabled());
                String token = JwtUtil.generateToken(claims);
                LoginMsg loginMsg = new LoginMsg(token, UserInfo);
                log.info("成功登录");
                return Result.success(loginMsg);
            }
        }
        log.info("用户名或密码错误,登录失败");
        return Result.error(401, "用户名或密码错误");

    }

    @Override
    public Result register(String userName, String password, String name) {
        //  对密码进行加密
        password = passwordEncoder.encode(password);
        User user = new User(null, userName, password, name,null, true, null);
        try {
            userDao.addUser(user);
        } catch (Exception e) {
            log.warn("用户名已存在");
            return Result.error(500,"用户名已存在");
        }
        log.info("成功注册");
        UserInfo userInfo = new UserInfo(user.getUserName(), "PUBLIC", user.getId());
        return Result.success(userInfo);
    }

    @Override
    public Result verifyToken(String token) {

        // 校验 token 是否有效
        if (JwtUtil.validateToken(token)) {
            Map<String, Object> claims = JwtUtil.parseToken(token);
            return Result.success(claims);
        } else {
            return Result.error(401, "无效的token");
        }
    }
}
