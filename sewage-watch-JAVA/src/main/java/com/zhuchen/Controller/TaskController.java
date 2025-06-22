package com.zhuchen.Controller;

import com.zhuchen.Service.TaskService;
import com.zhuchen.project.Result;
import com.zhuchen.project.task.Task;
import lombok.extern.slf4j.Slf4j;
import org.apache.ibatis.annotations.Delete;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@Slf4j
@RestController()
@RequestMapping("/tasks")
public class TaskController {
    private TaskService taskService;
    @Autowired
    public TaskController(TaskService taskService) {
        this.taskService = taskService;
    }
    @GetMapping
    public Result tasks()
    {
        log.info("查询所有任务");
        return taskService.getAllTask();
    }
    @DeleteMapping("/{id}")
    public Result deleteTask(@PathVariable Integer id) {
        log.info("删除任务，id为 {}", id);
        return taskService.deleteTaskById(id);
    }
    @PostMapping
    public Result addTask(@RequestBody Task task) {
        log.info("添加任务 " + task);
        return taskService.addTask(task);
    }
    @PutMapping("/{id}")
    public Result updateTask(@RequestBody Task task, @PathVariable Integer id) {
        task.setId(id);
        log.info("更新任务 " + task);
        return taskService.updateTask(task);
    }
}
