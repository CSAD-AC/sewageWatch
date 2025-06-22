package com.zhuchen.Dao.Impl;
import com.zhuchen.Dao.UserDao;
import com.zhuchen.Mapper.UserMapper;
import com.zhuchen.project.User;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
public class UserDaoImpl implements UserDao {

    @Autowired
    private UserMapper userMapper;


    @Override
    public List<User> findUserByUsername(String username) {
        return userMapper.findUser(username, null, null, null, null, null);
    }
    @Override
    public void addUser(User user) {
        userMapper.addUser(user);
    }
    @Override
    public void deleteUserById(int id) {
        userMapper.deleteUserById(id);
    }
    @Override
    public void updateUser(User user) {
        userMapper.updateUser(user);
    }

    @Override
    public List<User> findAllUser() {
        return userMapper.findUser(null, null, null, null, null, null);
    }


}
