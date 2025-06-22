package com.zhuchen.Controller;

import com.zhuchen.Service.UserService;
import com.zhuchen.project.Result;
import com.zhuchen.project.User;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@Slf4j
@RestController
@RequestMapping("/admin")
public class AdminController {
    UserService userService;
    @Autowired
    public AdminController(UserService userService) {
        this.userService = userService;
    }

    @PostMapping("/users")
    public Result addUser (@RequestBody User  user) {
        return userService.addUser(user);
    }

    @PutMapping("/users/{id}")
    public Result updateUser (@RequestBody User  user,@PathVariable Integer id) {
        user.setId(id);
        return userService.updateUser(user);
    }
    @DeleteMapping("/users/{id}")
    public Result deleteUser (@PathVariable Integer id) {
        return userService.deleteUserById(id);
    }
    @GetMapping("/users")
    public Result users() {
        return userService.getAllUser();
    }

}
