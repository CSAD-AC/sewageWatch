package com.zhuchen.Mapper;

import com.zhuchen.project.History;
import org.apache.ibatis.annotations.Mapper;

import java.util.List;

@Mapper
public interface HistoryMapper {
    public List<History> findAllHistory();
    public void deleteHistoryById(int id);
    public List<History> findHistory(History history);
    public void updateHistory(History history);
}
