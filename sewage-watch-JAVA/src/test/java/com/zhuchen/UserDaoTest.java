package com.zhuchen;

import com.zhuchen.Dao.Impl.UserDaoImpl;
import com.zhuchen.Dao.UserDao;
import com.zhuchen.project.User;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;


import java.time.LocalDateTime;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;
@SpringBootTest
public class UserDaoTest {
    @Autowired
    private UserDao userDao;
    private User user;

    @BeforeEach
    public void setUp() {
        user = new User();
        user.setUserName("testuser");
        user.setPassword("$2a$10$7QoNuH0waGWfvDD.HIGZFO1o7EefHfAfwxM2nQanXkyL6/7H2gxyW");
        user.setName("testuser");
        user.setRole("PUBLIC");
        user.setEnabled(true);
        user.setUpdateTime(LocalDateTime.now());
    }
    @AfterEach
    public void tearDown() {
        if(userDao.findUserByUsername(user.getUserName()) != null)
            userDao.deleteUserById(user.getId());
    }

    @Test
    public void testFindUserByUsername() {
        // 初始状态，用户不存在
        assertTrue(userDao.findUserByUsername(user.getUserName()).isEmpty());

        // 添加用户后查找
        userDao.addUser(user);
        List<User> foundUsers = userDao.findUserByUsername(user.getUserName());

        // 验证结果不为空且用户名一致
        assertFalse(foundUsers.isEmpty());
        assertEquals(user.getUserName(), foundUsers.get(0).getUserName());
    }

    @Test
    public void testAddUser() {
        // 添加用户
        userDao.addUser(user);

        // 查询用户是否存在
        List<User> foundUsers = userDao.findUserByUsername(user.getUserName());

        // 验证用户是否成功添加
        assertFalse(foundUsers.isEmpty());
        assertEquals(user.getUserName(), foundUsers.get(0).getUserName());
    }

    @Test
    public void testDeleteUserById() {
        // 添加用户
        userDao.addUser(user);

        // 获取添加后的用户列表
        List<User> addedUserList = userDao.findUserByUsername(user.getUserName());
        int userId = addedUserList.get(0).getId();

        // 删除用户
        userDao.deleteUserById(userId);

        // 验证用户已被删除
        assertTrue(userDao.findUserByUsername(user.getUserName()).isEmpty());
    }

    @Test
    public void testUpdateUserById() {
        // 添加初始用户
        userDao.addUser(user);

        // 获取添加后的用户列表并修改信息
        List<User> addedUserList = userDao.findUserByUsername(user.getUserName());
        User updatedUser = addedUserList.get(0);
        String newUsername = "updateduser15as64d65";
        updatedUser.setUserName(newUsername);

        // 执行更新
        userDao.updateUser(updatedUser);

        // 查询更新后的用户
        List<User> updatedUserList = userDao.findUserByUsername(newUsername);

        // 验证用户名已更新
        assertFalse(updatedUserList.isEmpty());
        assertEquals(newUsername, updatedUserList.get(0).getUserName());
    }

}
