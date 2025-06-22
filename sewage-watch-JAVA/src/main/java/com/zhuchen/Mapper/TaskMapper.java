package com.zhuchen.Mapper;

import com.zhuchen.project.task.Task;
import org.apache.ibatis.annotations.Mapper;

import java.util.List;

@Mapper
public interface TaskMapper {
    public void addTask(Task task);
    public void updateTaskBy(Task task);  // 通过id更新任务
    public void deleteTaskById(int id);
    public List<Task> findTask(Task task);  //updatedTime存储createTime的截止查询时间
}
