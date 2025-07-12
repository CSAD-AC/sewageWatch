package com.zhuchen.Service.Impl;

import com.zhuchen.Dao.HistoryDao;
import com.zhuchen.Service.HistoryService;
import com.zhuchen.project.History;
import com.zhuchen.project.Result;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;


@Service
public class HistoryServiceImpl implements HistoryService {
    private HistoryDao historyDao;
    @Autowired
    public void setHistoryDao(HistoryDao historyDao) {
        this.historyDao = historyDao;
    }
    @Override
    public Result findAllHistory() {
        return Result.success(historyDao.findAllHistory());
    }

    @Override
    public Result deleteHistoryById(int id) {
        historyDao.deleteHistoryById(id);
        return Result.success("删除成功");
    }

    @Override
    public Result findHistory(History history) {
        return Result.success(historyDao.findHistory(history));
    }

    @Override
    public Result updateHistory(History history) {
        historyDao.updateHistory(history);
        return Result.success("更新成功");
    }
}
