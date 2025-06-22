package com.zhuchen.Service;

import com.zhuchen.project.Result;
import com.zhuchen.project.task.Task;
import org.springframework.http.ResponseEntity;

public interface TaskService {
    Result getAllTask();

    Result deleteTaskById(Integer id);

    Result addTask(Task task);

    Result updateTask(Task task);
}
