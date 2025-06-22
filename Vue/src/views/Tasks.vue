<template>
  <div class="table-container">
    <!-- 搜索表单 -->
    <div class="search-form">
      <el-form :inline="true" :model="searchForm" class="demo-form-inline">
        <div class="form-row">
          <el-form-item label="标题">
            <el-input v-model="searchForm.title" placeholder="请输入标题" clearable />
          </el-form-item>
          <el-form-item label="描述">
            <el-input v-model="searchForm.description" placeholder="请输入描述" clearable />
          </el-form-item>
          <el-form-item label="截止日期">
            <el-date-picker
              v-model="searchForm.dateRange"
              type="daterange"
              range-separator="至"
              start-placeholder="开始日期"
              end-placeholder="结束日期"
              value-format="YYYY-MM-DD"
            />
          </el-form-item>
        </div>
        <div class="form-row">
          <div class="left-items">
            <el-form-item label="状态">
              <el-select v-model="searchForm.status" placeholder="请选择状态" clearable>
                <el-option label="未完成" value="TODO" />
                <el-option label="进行中" value="DOING" />
                <el-option label="已完成" value="DONE" />
              </el-select>
            </el-form-item>
            <el-form-item label="优先级">
              <el-select v-model="searchForm.priority" placeholder="请选择优先级" clearable>
                <el-option label="最低" value="LOWEST" />
                <el-option label="低" value="LOW" />
                <el-option label="中" value="MEDIUM" />
                <el-option label="高" value="HIGH" />
                <el-option label="最高" value="HIGHEST" />
              </el-select>
            </el-form-item>
            <el-form-item>
              <el-button type="primary" @click="handleSearch">搜索</el-button>
              <el-button @click="handleReset">重置</el-button>
            </el-form-item>
          </div>
          <div class="right-items">
            <el-button type="primary" @click="handleAddTask" icon="Plus">新建任务</el-button>
          </div>
        </div>
      </el-form>
    </div>

    <el-table
      :data="paginatedTasks"
      style="width: 100%"
      :row-class-name="tableRowClassName"
      @sort-change="handleSortChange"
      border
      :header-cell-style="{ background: '#f5f7fa', color: '#606266' }"
      max-height="calc(100vh - 300px)"
    >
      <el-table-column
        type="index"
        label="序号"
        width="60"
        align="center"
        fixed
      />
      <el-table-column
        prop="title"
        label="标题"
        min-width="180"
        show-overflow-tooltip
        sortable="custom"
      />
      <el-table-column
        prop="description"
        label="描述"
        min-width="300"
        show-overflow-tooltip
      />
      <el-table-column
        prop="status"
        label="状态"
        width="90"
        align="center"
      >
        <template #default="{ row }">
          {{ Mapping(row.status) }}
        </template>
      </el-table-column>

      <el-table-column
        prop="priority" 
        label="优先级"
        width="110"
        align="center"
        sortable="custom"
      >
        <template #default="{ row }">
          {{ Mapping(row.priority) }}
        </template>
      </el-table-column>

      <el-table-column
        prop="deadline"
        label="截止日期"
        width="140"
        align="center"
        sortable="custom"
        :formatter="dateFormatter"
      >
      </el-table-column>
      <el-table-column
        prop="createdTime"
        label="创建时间"
        width="140"
        align="center"
        sortable="custom"
        :formatter="dateFormatter"
      />

      <el-table-column
        label="操作"
        width="150"
        fixed="right"
        align="center"
      >
        <template #default="scope">
          <el-button
            type="primary"
            link
            @click="handleEdit(scope.row)"
          >
            编辑
          </el-button>
          <el-button
            type="danger"
            link
            @click="handleDelete(scope.row)"
          >
            删除
          </el-button>
        </template>
      </el-table-column>

      <el-empty
        v-if="paginatedTasks.length === 0"
        description="暂无任务"
      />
    </el-table>

    <div class="pagination-container">
      <el-pagination
        v-model:current-page="currentPage"
        v-model:page-size="pageSize"
        :page-sizes="[5, 10, 20, 50]"
        :background="background"
        layout="total, sizes, prev, pager, next, jumper"
        :total="totalItems"
        @size-change="handleSizeChange"
        @current-change="handleCurrentChange"
      />
    </div>
  </div>

  <!-- 任务对话框 -->
  <el-dialog
    v-model="dialogVisible"
    :title="dialogType === 'add' ? '新建任务' : '编辑任务'"
    width="500px"
    :close-on-click-modal="false"
  >
    <el-form
      ref="taskFormRef"
      :model="taskForm"
      :rules="taskRules"
      label-width="80px"
      label-position="right"
    >
      <el-form-item label="标题" prop="title">
        <el-input v-model="taskForm.title" placeholder="请输入任务标题" />
      </el-form-item>
      <el-form-item label="描述" prop="description">
        <el-input
          v-model="taskForm.description"
          type="textarea"
          :rows="3"
          placeholder="请输入任务描述"
        />
      </el-form-item>
      <el-form-item label="状态" prop="status">
        <el-select v-model="taskForm.status" placeholder="请选择任务状态" style="width: 100%">
          <el-option label="未完成" value="TODO" />
          <el-option label="进行中" value="DOING" />
          <el-option label="已完成" value="DONE" />
        </el-select>
      </el-form-item>
      <el-form-item label="优先级" prop="priority">
        <el-select v-model="taskForm.priority" placeholder="请选择优先级" style="width: 100%">
          <el-option label="最低" value="LOWEST" />
          <el-option label="低" value="LOW" />
          <el-option label="中" value="MEDIUM" />
          <el-option label="高" value="HIGH" />
          <el-option label="最高" value="HIGHEST" />
        </el-select>
      </el-form-item>
      <el-form-item label="截止日期" prop="deadline">
        <el-date-picker
          v-model="taskForm.deadline"
          type="date"
          placeholder="选择截止日期"
          style="width: 100%"
          value-format="YYYY-MM-DD"
        />
      </el-form-item>
    </el-form>
          <template #footer>
            <span class="dialog-footer">
              <el-button @click="dialogVisible = false" :disabled="loading">取消</el-button>
              <el-button type="primary" @click="submitTaskForm" :loading="loading">确认</el-button>
            </span>
          </template>
  </el-dialog>
</template>

<script setup>
import { ref, onMounted, computed, reactive } from 'vue'
import service from '../utils/request'
import dayjs from 'dayjs'
import { ElMessage, ElMessageBox } from 'element-plus'

// 日期格式化函数
const dateFormatter = (row, column, cellValue) => {
  return dayjs(cellValue).format('YYYY-MM-DD')
}

const tasks = ref([])

// 搜索表单数据
const searchForm = ref({
  title: '',
  description: '',
  dateRange: [],
  status: '',
  priority: ''
});

// 排序配置
const sortConfig = ref({
  prop: '',
  order: ''
});

// 分页相关数据
const currentPage = ref(1)
const pageSize = ref(10)
const background = ref(true)

async function getTasks() {
  try {
    const result = await service.get('/tasks')
    tasks.value = result.data
    console.log('获取的任务数据:', tasks.value)
  } catch (error) {
    console.error('获取任务失败:', error)
  }
}

onMounted(() => {
  getTasks();
})

// 过滤和排序后的任务列表
const filteredTasks = computed(() => {
  // 先过滤
  let result = tasks.value.filter(task => {
    // 标题过滤
    if (searchForm.value.title && !task.title.toLowerCase().includes(searchForm.value.title.toLowerCase())) {
      return false;
    }
    
    // 描述过滤
    if (searchForm.value.description && !task.description.toLowerCase().includes(searchForm.value.description.toLowerCase())) {
      return false;
    }
    
    // 状态过滤
    if (searchForm.value.status && task.status !== searchForm.value.status) {
      return false;
    }
    
    // 优先级过滤
    if (searchForm.value.priority && task.priority !== searchForm.value.priority) {
      return false;
    }
    
    // 日期范围过滤
    if (searchForm.value.dateRange && searchForm.value.dateRange.length === 2) {
      const startDate = new Date(searchForm.value.dateRange[0]);
      const endDate = new Date(searchForm.value.dateRange[1]);
      const taskDate = new Date(task.deadline);
      
      if (taskDate < startDate || taskDate > endDate) {
        return false;
      }
    }
    
    return true;
  });
  
  // 再排序
  if (sortConfig.value.prop && sortConfig.value.order) {
    const isAsc = sortConfig.value.order === 'ascending';
    const prop = sortConfig.value.prop;
    
    result.sort((a, b) => {
      let valueA, valueB;
      
      // 处理日期类型的排序
      if (prop === 'deadline' || prop === 'createdTime') {
        valueA = new Date(a[prop]).getTime();
        valueB = new Date(b[prop]).getTime();
      } 
      // 处理优先级排序
      else if (prop === 'priority') {
        const priorityOrder = { 'LOWEST': 1, 'LOW': 2, 'MEDIUM': 3, 'HIGH': 4, 'HIGHEST': 5 };
        valueA = priorityOrder[a[prop]];
        valueB = priorityOrder[b[prop]];
      }
      // 处理普通字符串排序
      else {
        valueA = a[prop];
        valueB = b[prop];
      }
      
      if (valueA < valueB) {
        return isAsc ? -1 : 1;
      }
      if (valueA > valueB) {
        return isAsc ? 1 : -1;
      }
      return 0;
    });
  }
  
  return result;
});

// 分页后的任务列表
const paginatedTasks = computed(() => {
  const start = (currentPage.value - 1) * pageSize.value;
  const end = start + pageSize.value;
  return filteredTasks.value.slice(start, end);
});

// 总数据量
const totalItems = computed(() => {
  return filteredTasks.value.length;
});

// 搜索方法
const handleSearch = () => {
  // 搜索逻辑已经在 filteredTasks 计算属性中实现
  console.log('执行搜索，当前搜索条件:', searchForm.value);
  // 重置到第一页
  currentPage.value = 1;
};

// 重置方法
const handleReset = () => {
  searchForm.value = {
    title: '',
    description: '',
    dateRange: [],
    status: '',
    priority: ''
  };
  sortConfig.value = {
    prop: '',
    order: ''
  };
  // 重置到第一页
  currentPage.value = 1;
};

// 排序方法
const handleSortChange = ({ prop, order }) => {
  sortConfig.value = { prop, order };
  // 重置到第一页
  currentPage.value = 1;
};

// 页码变化处理
const handleCurrentChange = (val) => {
  currentPage.value = val;
};

// 每页条数变化处理
const handleSizeChange = (val) => {
  pageSize.value = val;
  // 重置到第一页
  currentPage.value = 1;
};

const Mapping = (field) => {
  return {
    TODO: '未完成',
    DONE: '已完成',
    DOING: '进行中',
    LOWEST: '最低',
    LOW: '低',
    MEDIUM: '中',
    HIGH: '高',
    HIGHEST: '最高',
  }[field] || field
}

const tableRowClassName = ({ row, rowIndex }) => {
  if (row.priority === 'LOWEST') return 'priority-lowest'
  if (row.priority === 'LOW') return 'priority-low'
  if (row.priority === 'MEDIUM') return 'priority-medium'
  if (row.priority === 'HIGH') return 'priority-high'
  if (row.priority === 'HIGHEST') return 'priority-highest'

  return ''
}

// 对话框相关
const dialogVisible = ref(false)
const dialogType = ref('add') // 'add' 或 'edit'
const taskFormRef = ref(null)
const currentTaskId = ref(null)
const loading = ref(false)

const taskForm = reactive({
  title: '',
  description: '',
  status: 'TODO',
  priority: 'MEDIUM',
  deadline: ''
})

const taskRules = {
  title: [
    { required: true, message: '请输入任务标题', trigger: 'blur' },
    { min: 2, max: 50, message: '长度在 2 到 50 个字符', trigger: 'blur' }
  ],
  status: [
    { required: true, message: '请选择任务状态', trigger: 'change' }
  ],
  priority: [
    { required: true, message: '请选择优先级', trigger: 'change' }
  ],
  deadline: [
    { required: true, message: '请选择截止日期', trigger: 'change' }
  ]
}

// 重置表单
const resetTaskForm = () => {
  taskForm.title = ''
  taskForm.description = ''
  taskForm.status = 'TODO'
  taskForm.priority = 'MEDIUM'
  taskForm.deadline = ''
  if (taskFormRef.value) {
    taskFormRef.value.resetFields()
  }
}

// 打开新建任务对话框
const handleAddTask = () => {
  dialogType.value = 'add'
  resetTaskForm()
  dialogVisible.value = true
}

// 打开编辑任务对话框
const handleEdit = (row) => {
  dialogType.value = 'edit'
  currentTaskId.value = row.id
  
  // 填充表单数据
  taskForm.title = row.title
  taskForm.description = row.description || ''
  taskForm.status = row.status
  taskForm.priority = row.priority
  taskForm.deadline = row.deadline
  
  dialogVisible.value = true
}

// 删除任务
const handleDelete = async (row) => {
  try {
    await ElMessageBox.confirm(
      '确定要删除该任务吗？',
      '警告',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )
    
    loading.value = true
    await service.delete(`/tasks/${row.id}`)
    
    ElMessage({
      type: 'success',
      message: '删除成功'
    })
    getTasks()
  } catch (error) {
    if (error !== 'cancel') { // 不是用户取消操作
      console.error('删除任务失败:', error)
      ElMessage({
        type: 'error',
        message: '删除失败'
      })
    }
  } finally {
    loading.value = false
  }
}

// 提交表单
const submitTaskForm = async () => {
  if (!taskFormRef.value) return
  
  try {
    const valid = await taskFormRef.value.validate()
    if (!valid) return
    
    loading.value = true
    const submitData = {
      title: taskForm.title,
      description: taskForm.description,
      status: taskForm.status,
      priority: taskForm.priority,
      deadline: dayjs(taskForm.deadline).format('YYYY-MM-DDTHH:mm:ss')
    }

    if (dialogType.value === 'add') {
      // 新建任务
      await service.post('/tasks', submitData)
      ElMessage({
        type: 'success',
        message: '创建成功'
      })
    } else {
      // 编辑任务
      await service.put(`/tasks/${currentTaskId.value}`, submitData)
      ElMessage({
        type: 'success',
        message: '更新成功'
      })
    }
    
    dialogVisible.value = false
    getTasks()
  } catch (error) {
    console.error('操作失败:', error)
    ElMessage({
      type: 'error',
      message: dialogType.value === 'add' ? '创建失败' : '更新失败'
    })
  } finally {
    loading.value = false
  }
}

</script>

<style>
/* 优先级背景色 */
.el-table .priority-lowest td {
  background-color: var(--el-color-info-light-9) !important;
}
.el-table .priority-low td {
  background-color: var(--el-color-primary-light-9) !important;
}
.el-table .priority-medium td {
  background-color: var(--el-color-success-light-9) !important;
}
.el-table .priority-high td {
  background-color: var(--el-color-warning-light-9) !important;
}
.el-table .priority-highest td {
  background-color: var(--el-color-danger-light-9) !important;
}

/* 确保鼠标悬停时保持背景色 */
.el-table .priority-lowest:hover td {
  background-color: var(--el-color-info-light-8) !important;
}
.el-table .priority-low:hover td {
  background-color: var(--el-color-primary-light-8) !important;
}
.el-table .priority-medium:hover td {
  background-color: var(--el-color-success-light-8) !important;
}
.el-table .priority-high:hover td {
  background-color: var(--el-color-warning-light-8) !important;
}
.el-table .priority-highest:hover td {
  background-color: var(--el-color-danger-light-8) !important;
}

.table-container {
  height: 100%;
  display: flex;
  flex-direction: column;
  padding: 20px;
  box-sizing: border-box;
}

.search-form {
  margin-bottom: 20px;
  padding: 15px;
  background-color: #f5f7fa;
  border-radius: 4px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.05);
}

.search-form {
  margin-bottom: 12px;
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

.left-items {
  display: flex;
  flex-wrap: wrap;
  flex-grow: 1;
  align-items: center;
}

.right-items {
  margin-left: auto;
  display: flex;
  align-items: center;
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

/* 调整表单项的内边距 */
.search-form :deep(.el-form-item__content) {
  line-height: 32px;
}

/* 表格容器样式 */
.el-table {
  flex: 1;
  overflow: auto;
}

/* 固定表头样式 */
:deep(.el-table__header-wrapper) {
  position: sticky;
  top: 0;
  z-index: 2;
}

/* 分页容器样式 */
.pagination-container {
  margin-top: 20px;
  padding: 10px 0;
  display: flex;
  justify-content: flex-end;
  background-color: #fff;
  border-radius: 4px;
  box-shadow: 0 -2px 12px 0 rgba(0, 0, 0, 0.05);
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