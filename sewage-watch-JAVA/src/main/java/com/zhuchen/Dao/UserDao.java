package com.zhuchen.Dao;

import com.zhuchen.project.User;

import java.util.List;

public interface UserDao {
    public List<User> findUserByUsername(String username);
    public void addUser(User user);
    public void deleteUserById(int id);
    public void updateUser(User user);
    public List<User> findAllUser();
}
