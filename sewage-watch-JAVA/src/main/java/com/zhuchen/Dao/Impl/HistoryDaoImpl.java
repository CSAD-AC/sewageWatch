package com.zhuchen.Dao.Impl;

import com.zhuchen.Dao.HistoryDao;
import com.zhuchen.Mapper.HistoryMapper;
import com.zhuchen.project.History;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
public class HistoryDaoImpl implements HistoryDao {
    HistoryMapper historyMapper;
    @Autowired
    public void setHistoryMapper(HistoryMapper historyMapper) {
        this.historyMapper = historyMapper;
    }
    @Override
    public List<History> findAllHistory() {
        return historyMapper.findAllHistory();
    }

    @Override
    public void deleteHistoryById(int id) {
        historyMapper.deleteHistoryById(id);
    }

    @Override
    public List<History> findHistory(History history) {
        return historyMapper.findHistory(history);
    }

    @Override
    public void updateHistory(History history) {
        historyMapper.updateHistory(history);
    }
}
