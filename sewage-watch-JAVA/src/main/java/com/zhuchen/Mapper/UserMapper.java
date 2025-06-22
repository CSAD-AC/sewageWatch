package com.zhuchen.Mapper;

import com.zhuchen.project.User;
import org.apache.ibatis.annotations.*;

import java.util.Date;
import java.util.List;

@Mapper //应用程序运行时，会自动创建UserMapper接口的实现类对象(代理对象)，并注入到IOC容器中
public interface UserMapper {
    //MySQL具体语句映射于 resources/com/zhuchen/Mapper/UserMapper.xml

//    public List<User> FindAll();
//
//    @Select("SELECT id, username, password, name, age, update_time as updateTime FROM user WHERE username = #{username}")
//    public List<User> findUser(String username);

    List<User> findUser(
            String userName,
            String password,
            String name,
            String role,
            Boolean enabled,
            Date updateTime
    );
    void addUser(User user);

    void deleteUserById(int id);

    void updateUser(User user);
}