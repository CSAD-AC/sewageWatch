package com.zhuchen.Dao;

import com.zhuchen.project.History;

import java.util.List;

public interface HistoryDao {
    public List<History> findAllHistory();
    public void deleteHistoryById(int id);
    public List<History> findHistory(History history);
    public void updateHistory(History history);
}
