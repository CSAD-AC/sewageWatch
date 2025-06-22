package com.zhuchen;

import com.zhuchen.Mapper.TaskMapper;
import com.zhuchen.project.task.Task;
import com.zhuchen.project.task.TaskPriority;
import com.zhuchen.project.task.TaskStatus;
import org.junit.jupiter.api.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
public class TaskDaoTest {

    @Autowired
    private TaskMapper taskMapper;

    private Task task;

    @BeforeEach
    public void setUp() {
        task = new Task();
        LocalDateTime now = LocalDateTime.now().truncatedTo(ChronoUnit.SECONDS);
        task.setTitle("Test Task");
        task.setDescription("This is a test task.");
        task.setStatus(TaskStatus.TODO);
        task.setPriority(TaskPriority.MEDIUM);
        task.setDeadline(now);
        task.setCreatedTime(now);
        task.setUpdatedTime(now);
    }

    @AfterEach
    public void tearDown() {
        if (task.getId() != 0) {
            taskMapper.deleteTaskById(task.getId());
        }
    }

    @Test
    public void testAddTask() {
        taskMapper.addTask(task);

        List<Task> foundTasks = taskMapper.findTask(new Task());
        assertFalse(foundTasks.isEmpty());
        assertTrue(foundTasks.stream().anyMatch(t -> t.getTitle().equals(task.getTitle())));
    }

    @Test
    public void testFindTask() {
        List<Task> initialTasks = taskMapper.findTask(task);
        assertFalse(initialTasks.contains(task));

        taskMapper.addTask(task);
        task.setCreatedTimeEnd(LocalDateTime.now());
        List<Task> foundTasks = taskMapper.findTask(task);
        assertFalse(foundTasks.isEmpty());
        assertEquals(task.getTitle(), foundTasks.get(0).getTitle());
    }

    @Test
    public void testUpdateTaskBy() {
        taskMapper.addTask(task);

        List<Task> addedTasks = taskMapper.findTask(task);
        Task updatedTask = addedTasks.get(0);
        String newTitle = "Updated Task Title";
        updatedTask.setTitle(newTitle);
        updatedTask.setStatus(TaskStatus.DOING);
        taskMapper.updateTaskBy(updatedTask);

        List<Task> updatedTasks = taskMapper.findTask(new Task());
        assertTrue(updatedTasks.stream().anyMatch(t -> t.getTitle().equals(newTitle)));
    }

    @Test
    public void testDeleteTaskById() {
        taskMapper.addTask(task);
        int taskId = taskMapper.findTask(task).get(0).getId();
        taskMapper.deleteTaskById(taskId);

        List<Task> remainingTasks = taskMapper.findTask(task);
        assertFalse(remainingTasks.stream().anyMatch(t -> t.getId() == taskId));
    }
}
