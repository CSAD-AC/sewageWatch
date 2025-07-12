<template>
  <div class="history-container">
    <h2>历史记录</h2>
    
    <!-- 搜索表单 -->
    <div class="search-form">
      <el-form :inline="true" :model="searchForm" class="demo-form-inline">
        <div class="form-row">
          <el-form-item label="时间范围">
            <el-date-picker
              v-model="searchForm.dateRange"
              type="daterange"
              range-separator="至"
              start-placeholder="开始日期"
              end-placeholder="结束日期"
              @change="handleDateChange"
            />
          </el-form-item>
          <el-form-item label="检测类型">
            <el-select v-model="searchForm.type" placeholder="选择类型" clearable>
              <el-option 
                v-for="option in typeOptions" 
                :key="option.value" 
                :label="option.label" 
                :value="option.value" 
              />
            </el-select>
          </el-form-item>
          <el-form-item>
            <el-button type="primary" @click="fetchHistory">查询</el-button>
            <el-button @click="resetSearch">重置</el-button>
          </el-form-item>
        </div>
      </el-form>
    </div>
    
    <el-card class="history-card">
      <el-table 
        :data="historyData" 
        style="width: 100%;" 
        v-loading="loading"
        border
        :header-cell-style="{ background: '#f5f7fa', color: '#606266' }"
        height="calc(100vh - 400px)"
      >
        <el-table-column prop="id" label="ID" width="80" align="center" />
        <el-table-column prop="type" label="检测类型" min-width="120" align="center">
          <template #default="{row}">
            <el-tag :type="getTagType(row.type)">
              {{ row.type }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column label="检测时间" min-width="180" align="center">
          <template #default="{row}">
            {{ formatDate(row.createdTime) }}
          </template>
        </el-table-column>
        <el-table-column label="检测图片" min-width="300" align="center">
          <template #default="{row}">
            <el-image 
              style="width: 220px; height: 180px"
              :src="getImageUrl(row.src)" 
              :preview-src-list="[getImageUrl(row.src)]"
              fit="cover"
              hide-on-click-modal
              preview-teleported
            />
          </template>
        </el-table-column>
        <el-table-column label="操作" min-width="120" align="center" fixed="right">
          <template #default="{row}">
            <el-button link type="primary" size="small" @click="viewDetail(row)">查看</el-button>
          </template>
        </el-table-column>
        
        <!-- 空数据状态 -->
        <template #empty>
          <el-empty description="暂无历史记录" />
        </template>
      </el-table>
      
      <!-- 图片查看对话框 -->
      <el-dialog
        v-model="dialogVisible"
        title="图片详情"
        width="80%"
        center
        destroy-on-close
      >
        <div class="image-detail-container">
          <el-image
            style="width: 100%; max-height: 80vh;"
            :src="currentImage"
            fit="contain"
          />
          <div class="image-info" v-if="currentRecord">
            <p><strong>ID:</strong> {{ currentRecord.id }}</p>
            <p><strong>检测类型:</strong> {{ currentRecord.type }}</p>
            <p><strong>检测时间:</strong> {{ formatDate(currentRecord.createdTime) }}</p>
          </div>
        </div>
      </el-dialog>
      
      <div class="pagination-container">
        <el-pagination
          v-model:current-page="currentPage"
          v-model:page-size="pageSize"
          :page-sizes="[10, 20, 30, 50]"
          :background="background"
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
import { ref, onMounted, computed } from 'vue'
import { ElMessage } from 'element-plus'
import request from '@/utils/request'

// 搜索表单数据
const searchForm = ref({
  dateRange: '',
  type: '',
  startTime: '',
  endTime: ''
})

// 原始历史记录数据
const allHistoryData = ref([])
const loading = ref(false)
const currentPage = ref(1)
const pageSize = ref(10)
const background = ref(true) // 添加背景变量，用于分页组件

// 图片查看对话框相关变量
const dialogVisible = ref(false)
const currentImage = ref('')
const currentRecord = ref(null)

/**
 * 获取所有历史记录数据
 */
const fetchAllHistory = async () => {
  try {
    // 设置加载状态
    loading.value = true
    
    // 发送请求获取所有数据
    const res = await request.get('/history')
    
    // 处理响应数据
    if (res.code === 200) {
      // 处理不同格式的响应数据
      if (res.data && Array.isArray(res.data.list)) {
        // 标准格式：{ list: [...数据列表], total: 总条数 }
        allHistoryData.value = res.data.list
      } else if (Array.isArray(res.data)) {
        // 兼容直接返回数组的情况
        allHistoryData.value = res.data
      } else {
        // 处理意外的数据格式
        allHistoryData.value = []
        console.warn('意外的响应数据格式:', res.data)
      }
    } else {
      // 处理错误响应
      allHistoryData.value = []
      ElMessage.error(res.msg || '获取历史记录失败')
    }
  } catch (error) {
    // 处理网络错误或其他异常
    allHistoryData.value = []
    ElMessage.error('网络错误，请稍后重试')
    console.error('获取历史记录失败:', error)
  } finally {
    // 无论成功失败都关闭加载状态
    loading.value = false
  }
}

// 过滤后的历史记录
const filteredHistory = computed(() => {
  return allHistoryData.value.filter(item => {
    // 检测类型过滤
    if (searchForm.value.type && item.type !== searchForm.value.type) {
      return false
    }
    
    // 日期范围过滤
    if (searchForm.value.startTime && searchForm.value.endTime) {
      const itemDate = new Date(item.createdTime).getTime()
      const startDate = new Date(searchForm.value.startTime).getTime()
      const endDate = new Date(searchForm.value.endTime).setHours(23, 59, 59, 999) // 设置为当天结束时间
      
      if (itemDate < startDate || itemDate > endDate) {
        return false
      }
    }
    
    return true
  })
})

// 分页后的历史记录
const historyData = computed(() => {
  const start = (currentPage.value - 1) * pageSize.value
  const end = start + pageSize.value
  return filteredHistory.value.slice(start, end)
})

// 总记录数
const total = computed(() => {
  return filteredHistory.value.length
})

// 获取所有唯一的检测类型
const typeOptions = computed(() => {
  // 使用Set去重
  const typeSet = new Set(allHistoryData.value.map(item => item.type))
  // 转换为选项数组格式
  const options = Array.from(typeSet).map(type => ({
    label: type,
    value: type
  }))
  // 添加"全部类型"选项
  return [{ label: '全部类型', value: '' }, ...options]
})

// 处理日期范围选择
const handleDateChange = (val) => {
  if (val && val.length === 2) {
    // 转换为YYYY-MM-DD格式，更适合后端API处理
    searchForm.value.startTime = formatDateToString(val[0])
    searchForm.value.endTime = formatDateToString(val[1])
  } else {
    searchForm.value.startTime = ''
    searchForm.value.endTime = ''
  }
}

// 将Date对象转换为YYYY-MM-DD格式的字符串
const formatDateToString = (date) => {
  if (!date) return ''
  
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, '0')
  const day = String(date.getDate()).padStart(2, '0')
  
  return `${year}-${month}-${day}`
}

// 执行查询
const fetchHistory = () => {
  // 查询时重置到第一页
  currentPage.value = 1
  // 由于使用了计算属性，不需要额外的操作，数据会自动更新
}

// 重置搜索条件
const resetSearch = () => {
  // 重置搜索表单
  searchForm.value = {
    dateRange: '',
    type: '',
    startTime: '',
    endTime: ''
  }
  
  // 重置分页到第一页
  currentPage.value = 1
}

// 处理分页大小变化
const handleSizeChange = (val) => {
  pageSize.value = val
  // 重置到第一页
  currentPage.value = 1
}

// 处理页码变化
const handleCurrentChange = (val) => {
  currentPage.value = val
}

/**
 * 获取图片URL
 * 处理不同格式的图片路径，确保正确显示
 * @param {string} src - 图片源路径
 * @returns {string} 格式化后的图片URL
 */
const getImageUrl = (src) => {
  if (!src) return ''; // 处理空路径情况
  
  // 使用正斜杠替代反斜杠，确保URL格式正确
  if (src.startsWith('http') || src.startsWith('https')) {
    // 如果已经是完整URL，直接返回
    return src;
  } else {
    // 如果是相对路径，转换为正确格式
    // 移除路径中可能存在的前导斜杠
    const cleanPath = src.replace(/^[\/\\]+/, '').replace(/\\/g, '/');
    return `../../${cleanPath}`;
  }
}

/**
 * 根据检测类型获取标签类型
 * @param {string} type - 检测类型
 * @returns {string} 标签类型
 */
const getTagType = (type) => {
  // 根据不同的检测类型返回不同的标签类型
  const typeMap = {
    'bottle': 'danger',
    'bird': 'warning',
    'plastic': 'info',
    'trash': 'error'
  }
  
  // 如果有预定义的类型则使用，否则使用默认类型
  return typeMap[type] || 'primary'
}

/**
 * 格式化日期为yyyy-mm-dd
 * @param {string} dateString - 日期字符串
 * @returns {string} 格式化后的日期字符串
 */
const formatDate = (dateString) => {
  if (!dateString) return '';
  
  try {
    const date = new Date(dateString);
    if (isNaN(date.getTime())) return dateString; // 如果日期无效，返回原始字符串
    
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    
    // 可选：添加时间部分
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    
    // 返回格式化的日期，这里只返回日期部分，如果需要时间可以使用下面注释的代码
    return `${year}-${month}-${day}`;
    // 包含时间的格式：return `${year}-${month}-${day} ${hours}:${minutes}`;
  } catch (error) {
    console.error('日期格式化错误:', error);
    return dateString; // 出错时返回原始字符串
  }
};

// 查看详情
const viewDetail = (row) => {
  // 设置当前记录和图片URL
  currentRecord.value = row
  currentImage.value = getImageUrl(row.src)
  // 打开对话框
  dialogVisible.value = true
}

// 初始化加载数据
onMounted(() => {
  fetchAllHistory()
})
</script>

<style scoped>
.history-container {
  padding: 20px;
  height: 100%;
  display: flex;
  flex-direction: column;
  box-sizing: border-box;
  overflow: hidden; /* 防止内容溢出 */
}

.search-form {
  margin-bottom: 20px;
  padding: 15px;
  background-color: #f5f7fa;
  border-radius: 4px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.05);
}

.search-form .el-form {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.form-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  width: 100%;
}

.search-form .el-form-item {
  margin-bottom: 0;
  margin-right: 15px;
}

.search-form .el-form-item:last-child {
  margin-right: 0;
}

.search-form .el-button {
  margin-left: 10px;
}

.history-card {
  margin-top: 20px;
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden; /* 防止内容溢出 */
}

.pagination-container {
  margin-top: 20px;
  padding: 10px 0;
  display: flex;
  justify-content: flex-end;
  background-color: #fff;
  border-radius: 4px;
  box-shadow: 0 -2px 12px 0 rgba(0, 0, 0, 0.05);
  flex-shrink: 0; /* 防止分页条被压缩 */
  position: relative; /* 改为相对定位 */
  z-index: 10; /* 增加z-index确保在其他元素之上 */
}

.image-detail-container {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.image-info {
  margin-top: 20px;
  width: 100%;
  padding: 15px;
  background-color: #f8f9fa;
  border-radius: 4px;
}

.image-info p {
  margin: 8px 0;
  font-size: 14px;
}

/* 自定义对话框样式 */
:deep(.el-dialog__body) {
  padding: 20px;
}

:deep(.el-dialog__header) {
  padding: 15px 20px;
  border-bottom: 1px solid #ebeef5;
}

/* 固定表头样式 */
:deep(.el-table__header-wrapper) {
  position: sticky;
  top: 0;
  z-index: 2;
}

/* 响应式调整 */
@media (max-width: 768px) {
  .search-form .el-form-item {
    width: 100%;
    margin-right: 0;
  }
  
  .pagination-container {
    justify-content: center;
  }
}

/* 优化表格在小屏幕上的显示 */
@media (max-width: 1200px) {
  .el-table {
    width: 100%;
    overflow-x: auto;
  }
}
</style>