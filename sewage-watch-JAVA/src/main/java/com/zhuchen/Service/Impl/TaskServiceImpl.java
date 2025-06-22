package com.zhuchen.Service.Impl;

import com.zhuchen.Dao.TaskDao;
import com.zhuchen.Service.TaskService;
import com.zhuchen.project.Result;
import com.zhuchen.project.task.Task;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
@Slf4j
@Service
public class TaskServiceImpl implements TaskService {
    TaskDao taskDao;
    @Autowired
    public TaskServiceImpl(TaskDao taskDao) {
        this.taskDao = taskDao;
    }
    @Override
    public Result getAllTask() {
        List<Task> tasks = taskDao.findAllTask();
        return Result.success(tasks);
    }

    @Override
    public Result deleteTaskById(Integer id) {
        taskDao.deleteTaskById(id);
        return Result.success("删除成功");
    }

    @Override
    public Result addTask(Task task) {
        taskDao.addTask(task);
        return Result.success("添加成功");
    }

    @Override
    public Result updateTask(Task task) {
        taskDao.updateTask(task);
        return Result.success("更新成功");
    }
}
