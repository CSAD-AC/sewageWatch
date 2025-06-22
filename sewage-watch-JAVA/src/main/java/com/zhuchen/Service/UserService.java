package com.zhuchen.Service;

import com.zhuchen.project.Result;
import com.zhuchen.project.User;

public interface UserService {
    Result getAllUser();

    Result deleteUserById(Integer id);

    Result addUser(User user);

    Result updateUser(User user);

}
