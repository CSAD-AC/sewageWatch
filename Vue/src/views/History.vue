<template>
  <div class="history-container">
    <h2>历史记录</h2>
    
    <div class="search-bar">
      <el-form :inline="true" :model="searchForm" class="demo-form-inline">
        <el-form-item label="时间范围">
          <el-date-picker
            v-model="searchForm.dateRange"
            type="daterange"
            range-separator="至"
            start-placeholder="开始日期"
            end-placeholder="结束日期"
          />
        </el-form-item>
        <el-form-item label="区域">
          <el-select v-model="searchForm.area" placeholder="选择区域">
            <el-option label="全部区域" value="" />
            <el-option label="区域A" value="A" />
            <el-option label="区域B" value="B" />
            <el-option label="区域C" value="C" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="handleSearch">查询</el-button>
          <el-button @click="resetSearch">重置</el-button>
        </el-form-item>
      </el-form>
    </div>
    
    <el-card class="history-card">
      <el-table :data="historyData" style="width: 100%">
        <el-table-column prop="id" label="记录ID" width="100" />
        <el-table-column prop="date" label="检测日期" width="180" />
        <el-table-column prop="area" label="检测区域" width="120" />
        <el-table-column prop="result" label="检测结果">
          <template #default="scope">
            <el-tag :type="scope.row.result === '已发现污染' ? 'danger' : 'success'">
              {{ scope.row.result }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="level" label="污染等级" width="120" />
        <el-table-column label="操作" width="150">
          <template #default>
            <el-button link type="primary" size="small" @click="viewDetail">查看详情</el-button>
            <el-button link type="primary" size="small" @click="downloadReport">下载报告</el-button>
          </template>
        </el-table-column>
      </el-table>
      
      <div class="pagination">
        <el-pagination
          v-model:current-page="currentPage"
          v-model:page-size="pageSize"
          :page-sizes="[10, 20, 30, 50]"
          layout="total, sizes, prev, pager, next, jumper"
          :total="total"
          @size-change="handleSizeChange"
          @current-change="handleCurrentChange"
        />
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { ElMessage } from 'element-plus'

const searchForm = ref({
  dateRange: '',
  area: ''
})

const historyData = ref([
  {
    id: 'H001',
    date: '2024-01-15 09:30:45',
    area: '区域A',
    result: '已发现污染',
    level: '中度'
  },
  {
    id: 'H002',
    date: '2024-01-14 14:22:10',
    area: '区域B',
    result: '未发现污染',
    level: '-'
  },
  {
    id: 'H003',
    date: '2024-01-13 11:05:33',
    area: '区域C',
    result: '已发现污染',
    level: '轻度'
  }
])

const currentPage = ref(1)
const pageSize = ref(10)
const total = ref(100)

const handleSearch = () => {
  ElMessage({
    message: '搜索功能待实现',
    type: 'info'
  })
}

const resetSearch = () => {
  searchForm.value = {
    dateRange: '',
    area: ''
  }
}

const viewDetail = () => {
  ElMessage({
    message: '查看详情功能待实现',
    type: 'info'
  })
}

const downloadReport = () => {
  ElMessage({
    message: '下载报告功能待实现',
    type: 'info'
  })
}

const handleSizeChange = (val) => {
  pageSize.value = val
  // 这里应该重新加载数据
}

const handleCurrentChange = (val) => {
  currentPage.value = val
  // 这里应该重新加载数据
}
</script>

<style scoped>
.history-container {
  padding: 20px;
}

.search-bar {
  margin-bottom: 20px;
}

.history-card {
  margin-top: 20px;
}

.pagination {
  margin-top: 20px;
  display: flex;
  justify-content: flex-end;
}
</style>