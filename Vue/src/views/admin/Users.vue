<template>
  <div class="users-container">
    <div class="header">
      <h1>用户管理</h1>
      <div class="search-box">
        <el-input
          v-model="searchText"
          placeholder="搜索用户名或昵称"
          clearable
          @clear="handleSearch"
          style="width: 300px; margin-right: 20px;"
        >
          <template #append>
            <el-button @click="handleSearch">
              <el-icon><Search /></el-icon>
            </el-button>
          </template>
        </el-input>
        <el-button type="primary" @click="handleAdd">新建用户</el-button>
        <el-button type="primary" @click="refreshUsers">刷新</el-button>
      </div>
    </div>
    
    <!-- 错误提示 -->
    <el-alert
      v-if="error"
      :title="error"
      type="error"
      show-icon
      @close="error = ''"
    />
    
    <!-- 用户列表 -->
    <el-table
      :data="paginatedUsers"
      border
      style="width: 100%"
      v-loading="loading"
      :header-cell-style="{ background: '#f5f7fa' }"
      :cell-style="{ padding: '8px 0' }"
    >
      <el-table-column type="index" label="序号" width="70" align="center" :index="startIndex" />
      <el-table-column prop="userName" label="用户名" min-width="150" align="center" />
      <el-table-column prop="name" label="昵称" min-width="120" align="center" />
      <el-table-column prop="role" label="角色" min-width="120" align="center">
        <template #default="{ row }">
          <el-tag
            :type="row.role === 'ADMIN' ? 'danger' : row.role === 'INSPECTOR' ? 'warning' : 'info'"
            style="width: 100%; text-align: center; display: inline-flex; justify-content: center; align-items: center"
          >
            {{ row.role === 'ADMIN' ? '管理员' : row.role === 'INSPECTOR' ? '巡检员' : '普通用户' }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="enabled" label="状态" width="100" align="center">
        <template #default="{ row }">
          <el-tag
            :type="row.enabled ? 'success' : 'info'"
            style="width: 100%; text-align: center"
          >
            {{ row.enabled ? '启用' : '禁用' }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="updateTime" label="更新时间" min-width="180" align="center">
        <template #default="{ row }">
          {{ new Date(row.updateTime).toLocaleString() }}
        </template>
      </el-table-column>
      <el-table-column label="操作" fixed="right" width="200" align="center">
        <template #default="{ row }">
          <el-button
            type="primary"
            size="small"
            @click="handleEdit(row)"
            :disabled="row.role === 'ADMIN'"
          >
            编辑
          </el-button>
          <el-button
            v-if="row.role !== 'ADMIN'"
            :type="row.enabled ? 'danger' : 'success'"
            size="small"
            @click="toggleUserStatus(row)"
          >
            {{ row.enabled ? '禁用' : '启用' }}
          </el-button>
        </template>
      </el-table-column>
    </el-table>

    <!-- 分页 -->
    <div class="pagination">
      <el-pagination
        v-model:current-page="currentPage"
        v-model:page-size="pageSize"
        :page-sizes="[10, 20, 50]"
        :total="total"
        layout="total, sizes, prev, pager, next"
        @size-change="handleSizeChange"
        @current-change="handleCurrentChange"
      />
    </div>

    <!-- 用户表单对话框 -->
    <el-dialog
      v-model="dialogVisible"
      :title="dialogType === 'add' ? '新建用户' : '编辑用户'"
      width="500px"
    >
      <el-form
        ref="formRef"
        :model="form"
        :rules="rules"
        label-width="100px"
        style="max-width: 460px"
      >
        <el-form-item label="用户名" prop="userName">
          <el-input v-model="form.userName" :disabled="dialogType === 'edit'" />
        </el-form-item>
        <el-form-item label="昵称" prop="name">
          <el-input v-model="form.name" />
        </el-form-item>
        <el-form-item label="角色" prop="role">
          <el-select v-model="form.role" style="width: 100%" :disabled="form.role === 'ADMIN'">
            <el-option label="普通用户" value="PUBLIC" />
            <el-option label="巡检员" value="INSPECTOR" />
          </el-select>
        </el-form-item>
        <el-form-item label="状态" prop="enabled">
          <el-switch
            v-model="form.enabled"
            active-text="启用"
            inactive-text="禁用"
          />
        </el-form-item>
        <el-form-item v-if="dialogType === 'add'" label="密码" prop="password">
          <el-input v-model="form.password" type="password" show-password />
        </el-form-item>
      </el-form>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="dialogVisible = false">取消</el-button>
          <el-button type="primary" @click="handleSubmit">确定</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, computed } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Search } from '@element-plus/icons-vue'
import request from '../../utils/request'

// 状态变量
const users = ref([])
const loading = ref(false)
const error = ref('')
const dialogVisible = ref(false)
const dialogType = ref('add')
const formRef = ref(null)
const currentPage = ref(1)
const pageSize = ref(10)
const total = ref(0)
const searchText = ref('')
const allUsers = ref([]) // 存储所有用户数据，用于本地搜索

// 计算分页后的用户数据
const paginatedUsers = computed(() => {
  const startIndex = (currentPage.value - 1) * pageSize.value
  const endIndex = startIndex + pageSize.value
  return users.value.slice(startIndex, endIndex)
})

// 计算序号起始索引
const startIndex = computed(() => {
  return (index) => {
    return (currentPage.value - 1) * pageSize.value + index + 1
  }
})

// 表单数据
const form = reactive({
  userName: '',
  name: '',
  role: 'PUBLIC',
  enabled: true,
  password: ''
})

// 表单验证规则
const rules = {
  userName: [
    { required: true, message: '请输入用户名', trigger: 'blur' },
    { min: 3, max: 20, message: '长度在 3 到 20 个字符', trigger: 'blur' }
  ],
  name: [
    { required: true, message: '请输入昵称', trigger: 'blur' }
  ],
  role: [
    { required: true, message: '请选择角色', trigger: 'change' }
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 6, message: '密码长度不能小于6位', trigger: 'blur' }
  ]
}

// 获取用户列表
const fetchUsers = async () => {
  loading.value = true
  error.value = ''
  
  try {
    const response = await request.get('/admin/users')
    
    if (response?.code === 200) {
      allUsers.value = response.data
      handleSearch() // 应用搜索过滤
    } else {
      throw new Error('获取用户列表失败')
    }
  } catch (err) {
    console.error('获取用户列表错误:', err)
    error.value = '获取用户列表失败，请重试'
  } finally {
    loading.value = false
  }
}

// 处理搜索
const handleSearch = () => {
  if (!searchText.value) {
    users.value = allUsers.value
  } else {
    const searchLower = searchText.value.toLowerCase()
    users.value = allUsers.value.filter(user => 
      user.userName.toLowerCase().includes(searchLower) ||
      user.name.toLowerCase().includes(searchLower)
    )
  }
  total.value = users.value.length
  currentPage.value = 1 // 重置到第一页
}

// 刷新用户列表
const refreshUsers = () => {
  searchText.value = '' // 清空搜索条件
  fetchUsers()
}

// 处理分页大小变化
const handleSizeChange = (val) => {
  pageSize.value = val
  currentPage.value = 1
  // 不需要重新获取数据，只需要更新分页
}

// 处理页码变化
const handleCurrentChange = (val) => {
  currentPage.value = val
  // 不需要重新获取数据，只需要更新分页
}

// 新建用户
const handleAdd = () => {
  dialogType.value = 'add'
  Object.assign(form, {
    userName: '',
    name: '',
    role: 'PUBLIC',
    enabled: true,
    password: ''
  })
  dialogVisible.value = true
}

// 编辑用户
const handleEdit = (user) => {
  dialogType.value = 'edit'
  Object.assign(form, {
    id: user.id,
    userName: user.userName,
    name: user.name,
    role: user.role,
    enabled: user.enabled
  })
  dialogVisible.value = true
}

// 提交表单
const handleSubmit = async () => {
  if (!formRef.value) return
  
  await formRef.value.validate(async (valid) => {
    if (valid) {
      try {
        const method = dialogType.value === 'add' ? 'post' : 'put'
        const url = dialogType.value === 'add' ? '/admin/users' : `/admin/users/${form.id}`
        
        // 创建提交数据的副本，避免修改原始表单
        const submitData = { ...form }
        
        // 如果是编辑模式，删除密码字段，避免将空密码发送到后端
        if (dialogType.value === 'edit') {
          delete submitData.password
        }
        
        const response = await request({
          url,
          method,
          data: submitData
        })
        
        if (response?.code === 200) {
          ElMessage.success(dialogType.value === 'add' ? '创建成功' : '更新成功')
          dialogVisible.value = false
          fetchUsers()
        } else {
          throw new Error(dialogType.value === 'add' ? '创建失败' : '更新失败')
        }
      } catch (err) {
        console.error('提交表单错误:', err)
        ElMessage.error(dialogType.value === 'add' ? '创建失败' : '更新失败')
      }
    }
  })
}

// 切换用户状态
const toggleUserStatus = async (user) => {
  const newStatus = !user.enabled
  const actionText = newStatus ? '启用' : '禁用'
  
  try {
    await ElMessageBox.confirm(
      `确定要${actionText}用户 "${user.name}" 吗？`,
      `${actionText}确认`,
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )

    const response = await request({
      url: `/admin/users/${user.id}`,
      method: 'put',
      data: {
        ...user,
        enabled: newStatus
      }
    })
    
    if (response?.code === 200) {
      ElMessage.success(`${actionText}成功`)
      fetchUsers()
    } else {
      throw new Error(`${actionText}失败`)
    }
  } catch (err) {
    if (err !== 'cancel') {
      console.error(`${actionText}用户错误:`, err)
      ElMessage.error(`${actionText}失败，请重试`)
    }
  }
}

// 初始化
fetchUsers()
</script>

<style scoped>
.users-container {
  padding: 20px;
}

.header {
  display: flex;
  justify-content: flex-start; /* 改为靠左对齐 */
  align-items: center;
  margin-bottom: 20px;
  gap: 20px; /* 添加间距 */
}

.header h1 {
  margin: 0;
}

.search-box {
  display: flex;
  align-items: center;
}

.pagination {
  margin-top: 20px;
  display: flex;
  justify-content: flex-end;
}
</style>