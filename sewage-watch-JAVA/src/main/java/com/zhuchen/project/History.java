package com.zhuchen.project;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class History {
    private Integer id;
    private Integer taskId;
    private String type;
    private String src;
    private LocalDateTime createdTime;
}
