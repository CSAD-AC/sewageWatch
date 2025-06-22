package com.zhuchen.Dao;

import com.zhuchen.project.task.Task;

import java.util.List;

public interface TaskDao {
    public List<Task> findAllTask();
    public void deleteTaskById(int id);
    public List<Task> findTask(Task task);
    public void addTask(Task task);
    public void updateTask(Task task);
}
