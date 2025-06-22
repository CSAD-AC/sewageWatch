package com.zhuchen.project.task;

import lombok.Getter;
/**
 * 任务优先级枚举类
 */
@Getter
public enum TaskPriority {
    LOWEST("最低", 0),
    LOW("低", 1),
    MEDIUM("中", 2),
    HIGH("高", 3),
    HIGHEST("最高", 4);

    private final String label;
    private final int order;

    TaskPriority(String label, int order) {
        this.label = label;
        this.order = order;
    }

    public boolean isHigherThan(TaskPriority other) {
        return this.order > other.order;
    }
}

