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
        <el-card class="chart-card" v-loading="loading">
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
            <div v-if="!historyData.length" class="chart-placeholder">
              <el-empty description="暂无图表数据" />
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
    
    <el-row :gutter="20" class="chart-row">
      <el-col :span="12">
        <el-card class="chart-card" v-loading="loading">
          <template #header>
            <div class="card-header">
              <span>污染类型分布</span>
            </div>
          </template>
          <div class="chart-container" ref="typeChartRef">
            <div v-if="!historyData.length" class="chart-placeholder">
              <el-empty description="暂无图表数据" />
            </div>
          </div>
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card class="chart-card" v-loading="loading">
          <template #header>
            <div class="card-header">
              <span>污染等级统计</span>
            </div>
          </template>
          <div class="chart-container" ref="levelChartRef">
            <div v-if="!historyData.length" class="chart-placeholder">
              <el-empty description="暂无图表数据" />
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
    
    <el-card class="summary-card" v-loading="loading">
      <template #header>
        <div class="card-header">
          <span>数据摘要</span>
        </div>
      </template>
      <el-descriptions :column="3" border>
        <el-descriptions-item label="检测总次数">{{ summary.totalDetections }}次</el-descriptions-item>
        <el-descriptions-item label="发现污染次数">{{ summary.pollutionDetections }}次</el-descriptions-item>
        <el-descriptions-item label="污染检出率">{{ summary.detectionRate }}</el-descriptions-item>
        <el-descriptions-item label="重度污染">{{ summary.highPollution }}次</el-descriptions-item>
        <el-descriptions-item label="中度污染">{{ summary.mediumPollution }}次</el-descriptions-item>
        <el-descriptions-item label="轻度污染">{{ summary.lowPollution }}次</el-descriptions-item>
      </el-descriptions>
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted, computed, watch } from 'vue'
import { ElMessage } from 'element-plus'
import * as echarts from 'echarts'
import request from '@/utils/request'

// 过滤表单数据
const filterForm = ref({
  timeRange: '30days',
  area: 'all'
})

// 图表类型和引用
const trendChartType = ref('line')
const trendChartRef = ref(null)
const typeChartRef = ref(null)
const levelChartRef = ref(null)

// 图表实例
let trendChart = null
let typeChart = null
let levelChart = null

// 原始数据
const historyData = ref([])

// 加载状态
const loading = ref(false)

// 数据摘要
const summary = ref({
  totalDetections: 0,
  pollutionDetections: 0,
  detectionRate: '0%',
  highPollution: 0,
  mediumPollution: 0,
  lowPollution: 0
})

// 模拟数据
const mockData = [
  { id: 1, type: 'bottle', createdTime: '2025-07-01T10:30:00', area: 'A' },
  { id: 2, type: 'plastic', createdTime: '2025-07-02T11:20:00', area: 'B' },
  { id: 3, type: 'trash', createdTime: '2025-07-03T09:15:00', area: 'A' },
  { id: 4, type: 'bottle', createdTime: '2025-07-04T14:45:00', area: 'C' },
  { id: 5, type: 'bird', createdTime: '2025-07-05T16:30:00', area: 'B' },
  { id: 6, type: 'plastic', createdTime: '2025-07-06T08:20:00', area: 'A' },
  { id: 7, type: 'trash', createdTime: '2025-07-07T13:10:00', area: 'C' },
  { id: 8, type: 'bottle', createdTime: '2025-07-08T15:25:00', area: 'B' },
  { id: 9, type: 'plastic', createdTime: '2025-07-09T10:40:00', area: 'A' },
  { id: 10, type: 'trash', createdTime: '2025-07-10T12:50:00', area: 'C' }
]

// 获取历史数据
const fetchHistoryData = async () => {
  try {
    loading.value = true
    console.log('开始获取历史数据...')
    
    try {
      const res = await request.get('/history')
      console.log('API响应:', res)
      
      if (res && res.code === 200) {
        if (res.data && Array.isArray(res.data.list)) {
          historyData.value = res.data.list
          console.log('使用res.data.list数据:', historyData.value.length)
        } else if (Array.isArray(res.data)) {
          historyData.value = res.data
          console.log('使用res.data数据:', historyData.value.length)
        } else {
          console.warn('API返回的数据格式不符合预期，使用模拟数据')
          historyData.value = mockData
        }
      } else {
        console.warn('API请求失败或返回错误码，使用模拟数据')
        historyData.value = mockData
      }
    } catch (error) {
      console.error('API请求出错，使用模拟数据:', error)
      historyData.value = mockData
    }
    
    console.log('最终使用的数据:', historyData.value)
    
    // 初始化图表
    setTimeout(() => {
      initCharts()
      // 计算数据摘要
      calculateSummary()
      loading.value = false
    }, 100) // 给DOM一点时间来渲染
    
  } catch (error) {
    historyData.value = mockData
    console.error('处理数据时出错:', error)
    ElMessage.error('处理数据时出错，已使用模拟数据')
    
    // 尝试使用模拟数据初始化图表
    setTimeout(() => {
      initCharts()
      calculateSummary()
      loading.value = false
    }, 100)
  }
}

// 根据时间范围过滤数据
const filteredData = computed(() => {
  if (!historyData.value.length) return []
  
  const now = new Date()
  let startDate = new Date()
  
  // 根据选择的时间范围设置起始日期
  switch (filterForm.value.timeRange) {
    case '7days':
      startDate.setDate(now.getDate() - 7)
      break
    case '30days':
      startDate.setDate(now.getDate() - 30)
      break
    case '3months':
      startDate.setMonth(now.getMonth() - 3)
      break
    case '1year':
      startDate.setFullYear(now.getFullYear() - 1)
      break
    default:
      startDate.setDate(now.getDate() - 30) // 默认30天
  }
  
  // 过滤数据
  return historyData.value.filter(item => {
    const itemDate = new Date(item.createdTime)
    
    // 时间范围过滤
    if (itemDate < startDate) {
      return false
    }
    
    // 区域过滤（如果有区域数据）
    if (filterForm.value.area !== 'all' && item.area && item.area !== filterForm.value.area) {
      return false
    }
    
    return true
  })
})

// 应用过滤器
const applyFilter = () => {
  updateCharts()
  calculateSummary()
}

// 重置过滤器
const resetFilter = () => {
  filterForm.value = {
    timeRange: '30days',
    area: 'all'
  }
  updateCharts()
  calculateSummary()
}

// 初始化图表
const initCharts = () => {
  console.log('开始初始化图表...')
  console.log('DOM引用状态:', {
    trendChartRef: !!trendChartRef.value,
    typeChartRef: !!typeChartRef.value,
    levelChartRef: !!levelChartRef.value
  })
  
  try {
    // 确保DOM元素已经渲染
    if (!trendChartRef.value) {
      console.warn('趋势图表DOM元素未找到')
      return
    }
    
    if (!typeChartRef.value) {
      console.warn('类型分布图表DOM元素未找到')
      return
    }
    
    if (!levelChartRef.value) {
      console.warn('污染等级图表DOM元素未找到')
      return
    }
    
    console.log('所有DOM元素已找到，开始初始化ECharts实例')
    
    try {
      // 销毁已存在的实例
      if (trendChart) {
        trendChart.dispose()
      }
      // 初始化趋势图表
      trendChart = echarts.init(trendChartRef.value)
      console.log('趋势图表初始化成功')
    } catch (error) {
      console.error('初始化趋势图表失败:', error)
    }
    
    try {
      // 销毁已存在的实例
      if (typeChart) {
        typeChart.dispose()
      }
      // 初始化类型分布图表
      typeChart = echarts.init(typeChartRef.value)
      console.log('类型分布图表初始化成功')
    } catch (error) {
      console.error('初始化类型分布图表失败:', error)
    }
    
    try {
      // 销毁已存在的实例
      if (levelChart) {
        levelChart.dispose()
      }
      // 初始化污染等级图表
      levelChart = echarts.init(levelChartRef.value)
      console.log('污染等级图表初始化成功')
    } catch (error) {
      console.error('初始化污染等级图表失败:', error)
    }
    
    // 更新图表数据
    console.log('开始更新图表数据')
    updateCharts()
    
    // 添加窗口大小变化监听器
    window.addEventListener('resize', () => {
      console.log('窗口大小变化，调整图表大小')
      trendChart && trendChart.resize()
      typeChart && typeChart.resize()
      levelChart && levelChart.resize()
    })
    
    console.log('图表初始化完成')
  } catch (error) {
    console.error('图表初始化过程中出错:', error)
  }
}

// 更新图表数据
const updateCharts = () => {
  console.log('开始更新所有图表')
  
  if (!trendChart) {
    console.warn('趋势图表实例不存在，无法更新')
  } else {
    try {
      updateTrendChart()
    } catch (error) {
      console.error('更新趋势图表失败:', error)
    }
  }
  
  if (!typeChart) {
    console.warn('类型分布图表实例不存在，无法更新')
  } else {
    try {
      updateTypeChart()
    } catch (error) {
      console.error('更新类型分布图表失败:', error)
    }
  }
  
  if (!levelChart) {
    console.warn('污染等级图表实例不存在，无法更新')
  } else {
    try {
      updateLevelChart()
    } catch (error) {
      console.error('更新污染等级图表失败:', error)
    }
  }
  
  console.log('所有图表更新完成')
}

// 更新趋势图表
const updateTrendChart = () => {
  console.log('更新趋势图表，数据长度:', filteredData.value.length)
  
  try {
    // 按日期分组数据
    const dateMap = new Map()
    
    filteredData.value.forEach(item => {
      if (!item.createdTime) {
        console.warn('数据项缺少createdTime字段:', item)
        return
      }
      
      try {
        const date = new Date(item.createdTime).toISOString().split('T')[0]
        if (!dateMap.has(date)) {
          dateMap.set(date, 0)
        }
        dateMap.set(date, dateMap.get(date) + 1)
      } catch (error) {
        console.warn('处理日期时出错:', error, item)
      }
    })
    
    // 按日期排序
    const sortedDates = Array.from(dateMap.keys()).sort()
    const counts = sortedDates.map(date => dateMap.get(date))
    
    console.log('趋势图表数据准备完成:', { dates: sortedDates.length, counts })
    
    // 设置图表选项
    const option = {
      title: {
        text: '污染检测趋势',
        left: 'center'
      },
      tooltip: {
        trigger: 'axis'
      },
      xAxis: {
        type: 'category',
        data: sortedDates,
        axisLabel: {
          rotate: 45,
          interval: sortedDates.length > 10 ? Math.floor(sortedDates.length / 10) : 0
        }
      },
      yAxis: {
        type: 'value',
        name: '检测次数'
      },
      series: [
        {
          name: '检测次数',
          type: trendChartType.value,
          data: counts,
          itemStyle: {
            color: '#409EFF'
          }
        }
      ],
      grid: {
        left: '3%',
        right: '4%',
        bottom: '15%',
        containLabel: true
      }
    }
    
    // 应用图表选项
    trendChart.setOption(option)
    console.log('趋势图表更新成功')
  } catch (error) {
    console.error('更新趋势图表时出错:', error)
  }
}

// 更新类型分布图表
const updateTypeChart = () => {
  console.log('更新类型分布图表，数据长度:', filteredData.value.length)
  
  try {
    // 按类型分组数据
    const typeMap = new Map()
    
    filteredData.value.forEach(item => {
      if (!item.type) {
        console.warn('数据项缺少type字段:', item)
        return
      }
      
      if (!typeMap.has(item.type)) {
        typeMap.set(item.type, 0)
      }
      typeMap.set(item.type, typeMap.get(item.type) + 1)
    })
    
    // 转换为饼图数据格式
    const pieData = Array.from(typeMap.entries()).map(([name, value]) => ({ name, value }))
    
    console.log('类型分布图表数据准备完成:', { types: typeMap.size, data: pieData })
    
    // 设置图表选项
    const option = {
      title: {
        text: '污染类型分布',
        left: 'center'
      },
      tooltip: {
        trigger: 'item',
        formatter: '{a} <br/>{b}: {c} ({d}%)'
      },
      legend: {
        orient: 'vertical',
        left: 'left',
        data: Array.from(typeMap.keys())
      },
      series: [
        {
          name: '污染类型',
          type: 'pie',
          radius: '60%',
          center: ['50%', '60%'],
          data: pieData,
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowOffsetX: 0,
              shadowColor: 'rgba(0, 0, 0, 0.5)'
            }
          }
        }
      ]
    }
    
    // 应用图表选项
    typeChart.setOption(option)
    console.log('类型分布图表更新成功')
  } catch (error) {
    console.error('更新类型分布图表时出错:', error)
  }
}

// 更新污染等级图表
const updateLevelChart = () => {
  console.log('更新污染等级图表，数据长度:', filteredData.value.length)
  
  try {
    // 模拟污染等级数据（实际应用中应该从API获取或根据某些规则计算）
    // 这里我们根据污染类型简单分类
    const levelMap = {
      'bottle': '中度污染',
      'plastic': '轻度污染',
      'trash': '重度污染',
      'bird': '轻度污染'
    }
    
    const levelCounts = {
      '轻度污染': 0,
      '中度污染': 0,
      '重度污染': 0
    }
    
    filteredData.value.forEach(item => {
      if (!item.type) {
        console.warn('数据项缺少type字段:', item)
        return
      }
      
      const level = levelMap[item.type] || '轻度污染'
      levelCounts[level]++
    })
    
    console.log('污染等级图表数据准备完成:', levelCounts)
    
    // 设置图表选项
    const option = {
      title: {
        text: '污染等级统计',
        left: 'center'
      },
      tooltip: {
        trigger: 'item'
      },
      xAxis: {
        type: 'category',
        data: Object.keys(levelCounts)
      },
      yAxis: {
        type: 'value',
        name: '数量'
      },
      series: [
        {
          name: '污染等级',
          type: 'bar',
          data: Object.values(levelCounts),
          itemStyle: {
            color: function(params) {
              const colors = {
                '轻度污染': '#91cc75',
                '中度污染': '#fac858',
                '重度污染': '#ee6666'
              }
              return colors[params.name] || '#5470c6'
            }
          }
        }
      ],
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      }
    }
    
    // 应用图表选项
    levelChart.setOption(option)
    console.log('污染等级图表更新成功')
  } catch (error) {
    console.error('更新污染等级图表时出错:', error)
  }
}

// 计算数据摘要
const calculateSummary = () => {
  console.log('开始计算数据摘要')
  
  try {
    // 总检测次数
    summary.value.totalDetections = historyData.value.length
    
    // 污染检测次数（这里假设所有记录都是污染检测）
    summary.value.pollutionDetections = filteredData.value.length
    
    // 污染检出率
    const rate = summary.value.totalDetections > 0 
      ? (summary.value.pollutionDetections / summary.value.totalDetections * 100).toFixed(1) 
      : 0
    summary.value.detectionRate = `${rate}%`
    
    // 按污染等级统计
    const levelMap = {
      'bottle': '中度污染',
      'plastic': '轻度污染',
      'trash': '重度污染',
      'bird': '轻度污染'
    }
    
    try {
      summary.value.highPollution = filteredData.value.filter(item => 
        item.type && levelMap[item.type] === '重度污染'
      ).length
      
      summary.value.mediumPollution = filteredData.value.filter(item => 
        item.type && levelMap[item.type] === '中度污染'
      ).length
      
      summary.value.lowPollution = filteredData.value.filter(item => 
        item.type && levelMap[item.type] === '轻度污染'
      ).length
    } catch (error) {
      console.error('计算污染等级统计时出错:', error)
      summary.value.highPollution = 0
      summary.value.mediumPollution = 0
      summary.value.lowPollution = 0
    }
    
    console.log('数据摘要计算完成:', summary.value)
  } catch (error) {
    console.error('计算数据摘要时出错:', error)
    // 设置默认值
    summary.value = {
      totalDetections: historyData.value.length || 0,
      pollutionDetections: filteredData.value.length || 0,
      detectionRate: '0%',
      highPollution: 0,
      mediumPollution: 0,
      lowPollution: 0
    }
  }
}

// 监听图表类型变化
watch(trendChartType, () => {
  updateTrendChart()
})

// 组件挂载时获取数据并初始化图表
onMounted(() => {
  fetchHistoryData()
})
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