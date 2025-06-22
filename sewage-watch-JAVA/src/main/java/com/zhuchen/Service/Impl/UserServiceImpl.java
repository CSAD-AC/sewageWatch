package com.zhuchen.Service.Impl;

import com.zhuchen.Dao.UserDao;
import com.zhuchen.Service.UserService;
import com.zhuchen.project.Result;
import com.zhuchen.project.User;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Service;

import java.util.List;

@Slf4j
@Service
public class UserServiceImpl implements UserService {
    private final UserDao userDao;
    private final BCryptPasswordEncoder passwordEncoder = new BCryptPasswordEncoder();
    @Autowired
    public UserServiceImpl(UserDao userDao) {
        this.userDao = userDao;
    }

    @Override
    public Result getAllUser() {
        List<User> users = userDao.findAllUser();
        return Result.success(users);
    }

    @Override
    public Result deleteUserById(Integer id) {
        userDao.deleteUserById(id);
        return Result.success("删除成功");
    }

    @Override
    public Result addUser(User user) {
        user.setPassword(passwordEncoder.encode(user.getPassword()));
        userDao.addUser(user);
        return Result.success("添加成功");
    }

    @Override
    public Result updateUser(User user) {
        userDao.updateUser(user);
        return Result.success("更新成功");
    }
}
