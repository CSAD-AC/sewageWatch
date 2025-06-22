<template>
  <div class="task-table-container">
    <!-- 搜索区域 -->
    <div class="search-area">
      <el-row :gutter="20">
        <el-col :span="8">
          <el-input
            v-model="searchKeyword"
            placeholder="搜索标题或描述"
            clearable
            prefix-icon="el-icon-search"
          >
            <template #prefix>
              <el-icon><search /></el-icon>
            </template>
          </el-input>
        </el-col>
        <el-col :span="8">
          <el-select v-model="statusFilter" placeholder="状态筛选" clearable style="width: 100%">
            <el-option
              v-for="item in statusOptions"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </el-col>
        <el-col :span="8">
          <el-select v-model="priorityFilter" placeholder="优先级筛选" clearable style="width: 100%">
            <el-option
              v-for="item in priorityOptions"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
        </el-col>
      </el-row>
    </div>

    <!-- 任务表格 -->
    <el-table
      :data="filteredTasks"
      style="width: 100%"
      border
      stripe
      :row-class-name="tableRowClassName"
    >
      <el-table-column
        label="序号"
        type="index"
        width="70"
        align="center"
      />
      
      <el-table-column
        prop="title"
        label="标题"
        min-width="120"
        show-overflow-tooltip
      />
      
      <el-table-column
        prop="description"
        label="描述"
        min-width="180"
        show-overflow-tooltip
      >
        <template #default="scope">
          {{ scope.row.description || '无' }}
        </template>
      </el-table-column>
      
      <el-table-column
        prop="status"
        label="状态"
        width="100"
        align="center"
      >
        <template #default="scope">
          <el-tag
            :type="getStatusType(scope.row.status)"
            effect="light"
          >
            {{ getStatusText(scope.row.status) }}
          </el-tag>
        </template>

      </el-table-column>
      
      <el-table-column
        prop="priority"
        label="优先级"
        width="100"
        align="center"
      >
        <template #default="scope">
          <el-tag
            :type="getPriorityType(scope.row.priority)"
            effect="light"
          >
            {{ getPriorityText(scope.row.priority) }}
          </el-tag>
        </template>
      </el-table-column>
      
      <el-table-column
        prop="deadline"
        label="截止日期"
        width="160"
        align="center"
      >
        <template #default="scope">
          {{ formatDateTime(scope.row.deadline) }}
        </template>
      </el-table-column>
      
      <el-table-column
        prop="createdTime"
        label="创建时间"
        width="160"
        align="center"
      >
        <template #default="scope">
          {{ formatDateTime(scope.row.createdTime) }}
        </template>
      </el-table-column>
    </el-table>
    
    <el-empty
      v-if="filteredTasks.length === 0"
      description="暂无任务"
    />

    
  </div>
</template>

<script>
import { Search } from '@element-plus/icons-vue'

export default {
  name: 'TaskTable',
  components: {
    Search
  },
  props: {
    tasks: {
      type: Array,
      required: true,
      default: () => []
    }
  },
  data() {
    return {
      searchKeyword: '',
      statusFilter: '',
      priorityFilter: '',
      
      // 状态选项
      statusOptions: [
        { value: 'TODO', label: '待办' },
        { value: 'IN_PROGRESS', label: '进行中' },
        { value: 'DONE', label: '已完成' },
        { value: 'CANCELLED', label: '已取消' }
      ],
      
      // 优先级选项
      priorityOptions: [
        { value: 'LOW', label: '低' },
        { value: 'MEDIUM', label: '中' },
        { value: 'HIGH', label: '高' },
        { value: 'URGENT', label: '紧急' }
      ]
    }
  },
  computed: {
    // 过滤后的任务列表
    filteredTasks() {
      return this.tasks.filter(task => {
        // 关键词过滤（标题和描述）
        const keywordMatch = !this.searchKeyword || 
          (task.title && task.title.toLowerCase().includes(this.searchKeyword.toLowerCase())) || 
          (task.description && task.description.toLowerCase().includes(this.searchKeyword.toLowerCase()));
        
        // 状态过滤
        const statusMatch = !this.statusFilter || task.status === this.statusFilter;
        
        // 优先级过滤
        const priorityMatch = !this.priorityFilter || task.priority === this.priorityFilter;
        
        return keywordMatch && statusMatch && priorityMatch;
      });
    }
  },
  methods: {
    formatDateTime(dateTimeStr) {
      if (!dateTimeStr) return '无';
      
      const date = new Date(dateTimeStr);
      return new Intl.DateTimeFormat('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
      }).format(date);
    },
    
    getStatusText(status) {
      const statusMap = {
        'TODO': '待办',
        'IN_PROGRESS': '进行中',
        'DONE': '已完成',
        'CANCELLED': '已取消'
      };
      return statusMap[status] || status;
    },
    
    getPriorityText(priority) {
      const priorityMap = {
        'LOW': '低',
        'MEDIUM': '中',
        'HIGH': '高',
        'URGENT': '紧急'
      };
      return priorityMap[priority] || priority;
    },
    
    getStatusType(status) {
      const typeMap = {
        'TODO': 'info',
        'IN_PROGRESS': 'warning',
        'DONE': 'success',
        'CANCELLED': 'info'
      };
      return typeMap[status] || '';
    },
    
    getPriorityType(priority) {
      const typeMap = {
        'LOW': 'info',
        'MEDIUM': 'warning',
        'HIGH': 'danger',
        'URGENT': 'danger'
      };
      return typeMap[priority] || '';
    },
    
    tableRowClassName({ row }) {
      if (row.status === 'DONE') {
        return 'task-done-row';
      }
      if (row.priority === 'HIGH' || row.priority === 'URGENT') {
        return 'task-high-priority-row';
      }
      return '';
    }
  }
}
</script>

<style scoped>
.task-table-container {
  width: 100%;
}

.search-area {
  margin-bottom: 20px;
}

/* 自定义表格行样式 */
:deep(.task-done-row) {
  background-color: #f0f9eb;
}

:deep(.task-high-priority-row) {
  background-color: #fdf6ec;
}

/* 覆盖Element Plus的一些默认样式 */
:deep(.el-table th) {
  background-color: #f5f7fa;
  color: #606266;
  font-weight: bold;
}

:deep(.el-tag) {
  margin: 0;
}
</style>