package com.zhuchen.Dao.Impl;

import com.zhuchen.Dao.TaskDao;
import com.zhuchen.Mapper.TaskMapper;
import com.zhuchen.project.task.Task;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
public class TaskDaoImpl implements TaskDao {
    private TaskMapper taskMapper;
    @Autowired
    public void setTaskMapper(TaskMapper taskMapper) {
        this.taskMapper = taskMapper;
    }
    @Override
    public List<Task> findAllTask() {
        Task task = new Task();
        return taskMapper.findTask(task);
    }

    @Override
    public void deleteTaskById(int id) {
        taskMapper.deleteTaskById(id);
    }

    @Override
    public List<Task> findTask(Task task) {
        return taskMapper.findTask(task);
    }

    @Override
    public void addTask(Task task) {
        taskMapper.addTask(task);
    }

    @Override
    public void updateTask(Task task) {
        taskMapper.updateTaskBy(task);
    }
}
