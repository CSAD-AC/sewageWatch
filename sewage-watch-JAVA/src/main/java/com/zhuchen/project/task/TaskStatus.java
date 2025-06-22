package com.zhuchen.project.task;

import lombok.Getter;

/**
 * 任务状态枚举类
 */
@Getter
public enum TaskStatus {
    TODO("待处理"),
    DOING("处理中"),
    DONE("已完成");

    private final String label;
    TaskStatus(String label) {
        this.label = label;
    }
}
