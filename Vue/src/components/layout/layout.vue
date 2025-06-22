<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { 
  Fold, 
  Expand,
  User, 
  VideoCamera, 
  List, 
  Search, 
  Document, 
  DataLine,
  ArrowRight,
  Setting
} from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import request from '../../utils/request'

const isCollapse = ref(false)
const router = useRouter()
const isLoggedIn = ref(false)
const userInfo = ref({
  name: '用户',
  role: '',
  userName: ''
})

// 安全导航方法
const safeNavigate = (path) => {
  try {
    console.log('安全导航到:', path)
    router.push(path)
    return true
  } catch (error) {
    console.error('导航错误:', error)
    ElMessage.error('导航失败，请重试')
    return false
  }
}

const handleSelect = (key) => {
  try {
    console.log('菜单选择:', key)
    
    // 处理管理员子菜单的导航
    if (key.startsWith('admin-')) {
      // 将 'admin-users' 转换为 '/admin/users'
      // 使用正则表达式替换第一个破折号
      const path = key.replace(/^admin-/, 'admin/')
      console.log('导航到管理页面:', '/' + path)
      safeNavigate('/' + path)
    } else {
      console.log('导航到页面:', '/' + key)
      safeNavigate('/' + key)
    }
  } catch (error) {
    console.error('导航错误:', error)
    ElMessage.error('导航失败，请重试')
  }
}

const toggleCollapse = () => {
  isCollapse.value = !isCollapse.value
}

const goToLogin = () => {
  router.push('/login')
}

const logout = async () => {
  try {
    // 清除本地存储
    localStorage.removeItem('token')
    localStorage.removeItem('userInfo')
    
    // 更新状态
    isLoggedIn.value = false
    userInfo.value = { name: '用户', role: '', userName: '' }
    
    ElMessage.success('已退出登录')
    // 跳转到登录页
    router.push('/login')
  } catch (error) {
    console.error('退出登录失败:', error)
    // 错误处理已在request.js中统一处理
  }
}

// 验证登录状态
const checkLoginStatus = async () => {
  console.log('开始验证登录状态')
  try {
    // 首先尝试从localStorage获取并验证用户信息
    const storedUserInfo = localStorage.getItem('userInfo')
    const token = localStorage.getItem('token')
    
    console.log('当前token状态:', token ? '存在' : '不存在')
    
    if (!token) {
      console.log('未找到token，设置为未登录状态')
      isLoggedIn.value = false
      return
    }
    
    // 验证token有效性
    console.log('开始验证token有效性')
    const response = await request({
      url: '/token',
      method: 'post'
    })
    
    if (response && response.code === 200 && response.data) {
      console.log('token验证成功，更新用户信息')

      // 依据token更新登录状态和用户信息
      isLoggedIn.value = true
      userInfo.value = {
        name: response.data.name || '用户',
        role: response.data.role || '',
        userName: response.data.userName || ''
      }
      
      // 保存用户信息到本地存储
      try {
        localStorage.setItem('userInfo', JSON.stringify(userInfo.value))
        console.log('用户信息已保存到localStorage')
      } catch (storageError) {
        console.error('保存用户信息到localStorage失败:', storageError)
      }
    } else {
      console.log('token验证失败，处理登录过期')
      handleLoginExpired()
    }
  } catch (error) {
    console.error('验证登录状态失败:', error)
    handleLoginExpired()
  }
}

// 处理登录过期
const handleLoginExpired = () => {
  isLoggedIn.value = false
  localStorage.removeItem('token')
  localStorage.removeItem('userInfo')
  userInfo.value = { name: '用户', role: '', userName: '' }
  router.push('/login')
}

onMounted(() => {
  // 验证身份权限
  checkLoginStatus()
})
</script>

<template>
  <el-container class="layout-container">
    <!-- 侧边栏 -->
    <el-aside :width="isCollapse ? '64px' : '200px'" class="aside">
      <el-menu
        :default-active="$route.path.substring(1).replace('/', '-')"
        class="el-menu-vertical"
        :collapse="isCollapse"
        @select="handleSelect"
      >
        <!-- 巡检员和管理员可见 -->
        <el-menu-item v-if="['INSPECTOR', 'ADMIN'].includes(userInfo.role)" index="monitor">
          <el-icon><VideoCamera /></el-icon>
          <template #title>视频监控</template>
        </el-menu-item>
        
        <!-- 巡检员和管理员可见 -->
        <el-menu-item v-if="['INSPECTOR', 'ADMIN'].includes(userInfo.role)" index="tasks">
          <el-icon><List /></el-icon>
          <template #title>任务管理</template>
        </el-menu-item>
        
        <!-- 所有认证用户可见 -->
        <el-menu-item index="detection">
          <el-icon><Search /></el-icon>
          <template #title>手动检测</template>
        </el-menu-item>
        
        <!-- 所有认证用户可见 -->
        <el-menu-item index="history">
          <el-icon><Document /></el-icon>
          <template #title>历史记录</template>
        </el-menu-item>
        
        <!-- 所有认证用户可见 -->
        <el-menu-item index="analysis">
          <el-icon><DataLine /></el-icon>
          <template #title>数据分析</template>
        </el-menu-item>
        
        <!-- 管理员菜单 -->
        <el-sub-menu v-if="userInfo.role === 'ADMIN'" index="admin">
          <template #title>
            <el-icon><Setting /></el-icon>
            <span>系统管理</span>
          </template>
          <el-menu-item index="admin-users">用户管理</el-menu-item>
          <el-menu-item index="admin-system">系统配置</el-menu-item>
        </el-sub-menu>
      </el-menu>
      <!-- 将折叠按钮移到底部 -->
      <div class="menu-footer">
        <el-icon class="toggle-button" @click="toggleCollapse">
          <component :is="isCollapse ? 'Expand' : 'Fold'" />
        </el-icon>
      </div>
    </el-aside>

    <!-- 主要内容区域 -->
    <el-container>
      <!-- 顶部导航栏 -->
      <el-header class="header">
        <div class="header-left">
          <h1 class="system-title">污水检测系统</h1>
        </div>
        <div class="header-right">
          <template v-if="isLoggedIn">
            <el-dropdown>
              <span class="user-info">
                <el-icon><User /></el-icon>
                <span>{{ userInfo.name }}</span>
                <el-icon class="arrow-icon"><ArrowRight /></el-icon>
              </span>
              <template #dropdown>
                <el-dropdown-menu>
                  <el-dropdown-item @click="logout">退出登录</el-dropdown-item>
                </el-dropdown-menu>
              </template>
            </el-dropdown>
          </template>
          <template v-else>
            <el-button type="primary" size="small" @click="goToLogin">登录</el-button>
          </template>
        </div>
      </el-header>

      <!-- 内容区域 -->
      <el-main class="main">
        <router-view></router-view>
      </el-main>
    </el-container>
  </el-container>
</template>

<style scoped>
.layout-container {
  height: 100vh;
}

.aside {
  background-color: #304156;
  transition: width 0.3s;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}

.menu-footer {
  height: 60px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-top: 1px solid #1f2d3d;
}

.toggle-button {
  color: #fff;
  font-size: 20px;
  cursor: pointer;
  transition: transform 0.3s;
}

.toggle-button:hover {
  transform: scale(1.1);
}

.el-menu-vertical {
  border-right: none;
  flex-grow: 1;
}

.header {
  background-color: #fff;
  border-bottom: 1px solid #e6e6e6;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
}

.header-left {
  flex: 1;
  display: flex;
  justify-content: center;
}

.system-title {
  margin: 0;
  font-size: 24px;
  color: #303133;
  font-weight: bold;
  font-family: 'Arial Black', 'Arial Bold', Gadget, sans-serif;
  text-align: center;
  letter-spacing: 1px;
  text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
}

.header-right {
  display: flex;
  align-items: center;
}

.header-right .user-info {
  display: flex;
  align-items: center;
  cursor: pointer;
  padding: 0 10px;
  outline: none !important;
  border: none !important;
}

.user-info .el-icon {
  margin-right: 8px;
}

.arrow-icon {
  margin-left: 5px;
  font-size: 12px;
}

.main {
  background-color: #f0f2f5;
  padding: 20px;
}

/* 深度选择器，修改Element Plus的默认样式 */
:deep(.el-menu) {
  background-color: #304156;
}

:deep(.el-menu-item) {
  color: #bfcbd9;
}

:deep(.el-menu-item:hover) {
  background-color: #263445;
}

:deep(.el-menu-item.is-active) {
  color: #409eff;
  background-color: #263445;
}

:deep(.el-menu-item .el-icon) {
  color: inherit;
}
</style>