package com.zhuchen.Service;

import com.zhuchen.project.History;
import com.zhuchen.project.Result;

import java.util.List;

public interface HistoryService {
    public Result findAllHistory();
    public Result deleteHistoryById(int id);
    public Result findHistory(History history);
    public Result updateHistory(History history);
}
