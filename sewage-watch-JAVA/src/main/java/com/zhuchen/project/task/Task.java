package com.zhuchen.project.task;

import com.fasterxml.jackson.annotation.JsonFormat;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class Task {
    private Integer id;                   // 任务ID
    private String title;                 // 任务标题
    private String description;           // 任务描述
    private TaskStatus status;            // 任务状态
    private TaskPriority priority;        // 任务优先级
    private LocalDateTime deadline;       // 任务截止时间
    private LocalDateTime createdTime;    // 任务创建时间
    private LocalDateTime updatedTime;    // 任务更新时间

    private LocalDateTime createdTimeEnd; // 用于查找创建时间
}
