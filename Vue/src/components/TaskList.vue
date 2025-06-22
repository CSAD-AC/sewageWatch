<template>
  <div class="task-list-container">
    <el-card class="task-card">
      <template #header>
        <div class="card-header">
          <h2>任务列表</h2>
          <el-button type="primary" @click="fetchTasks" :loading="loading">
            <el-icon><refresh /></el-icon>
            刷新
          </el-button>
        </div>
      </template>
      
      <!-- 加载状态 -->
      <div v-if="loading" class="loading-container">
        <el-skeleton :rows="5" animated />
      </div>
      
      <!-- 错误信息 -->
      <el-alert
        v-else-if="error"
        :title="error"
        type="error"
        show-icon
        :closable="false"
      />
      
      <!-- 任务表格 -->
      <task-table
        v-else
        :tasks="tasks"
      />
    </el-card>
  </div>
</template>

<script>
import { Refresh } from '@element-plus/icons-vue'
import TaskTable from './TaskTable.vue'
import axios from 'axios'

// 创建axios实例
const apiClient = axios.create({
  baseURL: 'http://localhost:8080',
  timeout: 10000, // 请求超时时间
  headers: {
    'Content-Type': 'application/json'
  }
})

export default {
  name: 'TaskList',
  components: {
    Refresh,
    TaskTable
  },
  data() {
    return {
      tasks: [],
      loading: true,
      error: null,
    }
  },
  mounted() {
    this.fetchTasks();
  },
  methods: {
    async fetchTasks() {
      try {
        this.loading = true;
        this.error = null;
        
        // 使用axios发送请求
        const response = await apiClient.get('/tasks');

        // axios自动解析JSON响应
        const result = response.data;
        
        if (result.code === 200) {
          console.log('获取任务列表成功:', result.data);
          this.tasks = result.data;
        } else {
          throw new Error(result.msg || '获取任务列表失败');
        }
      } catch (error) {
        console.error('获取任务列表出错:', error);
        
        // axios错误处理
        if (error.response) {
          // 服务器返回了错误状态码
          this.error = `服务器错误: ${error.response.status} - ${error.response.data.msg || '未知错误'}`;
        } else if (error.request) {
          // 请求已发送但没有收到响应
          this.error = '无法连接到服务器，请检查网络连接';
        } else {
          // 请求设置时发生错误
          this.error = `请求错误: ${error.message}`;
        }
      } finally {
        this.loading = false;
      }
    }
  }
}
</script>

<style scoped>
.task-list-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
  font-family: Arial, sans-serif;
}

.task-card {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.card-header h2 {
  margin: 0;
  font-size: 1.5rem;
  color: #303133;
}

.loading-container {
  padding: 20px 0;
}
</style>