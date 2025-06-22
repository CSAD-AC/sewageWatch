<template>
  <div class="analysis-container">
    <h2>数据分析</h2>
    
    <div class="filter-bar">
      <el-form :inline="true" :model="filterForm">
        <el-form-item label="时间范围">
          <el-select v-model="filterForm.timeRange" placeholder="选择时间范围">
            <el-option label="最近7天" value="7days" />
            <el-option label="最近30天" value="30days" />
            <el-option label="最近3个月" value="3months" />
            <el-option label="最近1年" value="1year" />
          </el-select>
        </el-form-item>
        <el-form-item label="区域">
          <el-select v-model="filterForm.area" placeholder="选择区域">
            <el-option label="全部区域" value="all" />
            <el-option label="区域A" value="A" />
            <el-option label="区域B" value="B" />
            <el-option label="区域C" value="C" />
          </el-select>
        </el-form-item>
        <el-form-item>
          <el-button type="primary" @click="applyFilter">应用</el-button>
          <el-button @click="resetFilter">重置</el-button>
        </el-form-item>
      </el-form>
    </div>
    
    <el-row :gutter="20">
      <el-col :span="24">
        <el-card class="chart-card">
          <template #header>
            <div class="card-header">
              <span>污染检测趋势</span>
              <el-radio-group v-model="trendChartType" size="small">
                <el-radio-button label="line">折线图</el-radio-button>
                <el-radio-button label="bar">柱状图</el-radio-button>
              </el-radio-group>
            </div>
          </template>
          <div class="chart-container" ref="trendChartRef">
            <!-- 这里将来会渲染实际的图表 -->
            <div class="chart-placeholder">
              <el-empty description="暂无图表数据" />
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
    
    <el-row :gutter="20" class="chart-row">
      <el-col :span="12">
        <el-card class="chart-card">
          <template #header>
            <div class="card-header">
              <span>污染类型分布</span>
            </div>
          </template>
          <div class="chart-container" ref="typeChartRef">
            <!-- 这里将来会渲染实际的图表 -->
            <div class="chart-placeholder">
              <el-empty description="暂无图表数据" />
            </div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card class="chart-card">
          <template #header>
            <div class="card-header">
              <span>污染等级统计</span>
            </div>
          </template>
          <div class="chart-container" ref="levelChartRef">
            <!-- 这里将来会渲染实际的图表 -->
            <div class="chart-placeholder">
              <el-empty description="暂无图表数据" />
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
    
    <el-card class="summary-card">
      <template #header>
        <div class="card-header">
          <span>数据摘要</span>
        </div>
      </template>
      <el-descriptions :column="3" border>
        <el-descriptions-item label="检测总次数">125次</el-descriptions-item>
        <el-descriptions-item label="发现污染次数">42次</el-descriptions-item>
        <el-descriptions-item label="污染检出率">33.6%</el-descriptions-item>
        <el-descriptions-item label="重度污染">5次</el-descriptions-item>
        <el-descriptions-item label="中度污染">15次</el-descriptions-item>
        <el-descriptions-item label="轻度污染">22次</el-descriptions-item>
      </el-descriptions>
    </el-card>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { ElMessage } from 'element-plus'

const filterForm = ref({
  timeRange: '30days',
  area: 'all'
})

const trendChartType = ref('line')
const trendChartRef = ref(null)
const typeChartRef = ref(null)
const levelChartRef = ref(null)

const applyFilter = () => {
  ElMessage({
    message: '应用筛选功能待实现',
    type: 'info'
  })
  // 这里应该重新加载图表数据
}

const resetFilter = () => {
  filterForm.value = {
    timeRange: '30days',
    area: 'all'
  }
}

// 在实际应用中，这里应该有初始化图表的代码
// 例如使用ECharts或其他图表库
</script>

<style scoped>
.analysis-container {
  padding: 20px;
}

.filter-bar {
  margin-bottom: 20px;
}

.chart-card {
  margin-bottom: 20px;
}

.chart-row {
  margin-top: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.chart-container {
  height: 300px;
  position: relative;
}

.chart-placeholder {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
}

.summary-card {
  margin-top: 20px;
}
</style>