<template>
  <div class="login-container">
    <div class="login-box">
      <h2>污水检测系统</h2>
      <el-form
        ref="loginFormRef"
        :model="loginForm"
        :rules="loginRules"
        class="login-form"
        @keyup.enter="handleLogin"
      >
        <el-form-item prop="userName">
          <el-input
            v-model="loginForm.userName"
            placeholder="请输入账号"
            prefix-icon="User"
          />
        </el-form-item>
        <el-form-item prop="password">
          <el-input
            v-model="loginForm.password"
            type="password"
            placeholder="请输入密码"
            prefix-icon="Lock"
            show-password
          />
        </el-form-item>
        <el-form-item>
          <el-button
            type="primary"
            :loading="loading"
            class="login-button"
            @click="handleLogin"
          >
            登录
          </el-button>
        </el-form-item>
        <div class="register-link">
          还没有账号？
          <router-link to="/register">立即注册</router-link>
        </div>
      </el-form>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { User, Lock } from '@element-plus/icons-vue'
import request from '../utils/request'

const router = useRouter()
const loginFormRef = ref(null)
const loading = ref(false)

const loginForm = reactive({
  userName: '',
  password: ''
})

const loginRules = {
  userName: [
    { required: true, message: '请输入账号', trigger: 'blur' },
    { min: 3, max: 20, message: '长度在 3 到 20 个字符', trigger: 'blur' }
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 6, max: 20, message: '长度在 6 到 20 个字符', trigger: 'blur' }
  ]
}

// 清空表单
const resetForm = () => {
  if (loginFormRef.value) {
    loginFormRef.value.resetFields()
  }
}

const handleLogin = async () => {
  if (!loginFormRef.value) return

  try {
    await loginFormRef.value.validate()
    loading.value = true

    const res = await request({
      url: '/login',
      method: 'post',
      data: {
        userName: loginForm.userName,
        password: loginForm.password
      }
    })

    // 登录成功
    const { token, userInfo } = res.data
    localStorage.setItem('token', token)
    
    ElMessage.success('登录成功')
    
    router.push('/')
  } catch (error) {
    console.error('登录失败:', error)
    // 错误处理已在request.js中统一处理
    resetForm()
  } finally {
    loading.value = false
  }
}

// 检查是否已登录
onMounted(() => {
  const token = localStorage.getItem('token')
  if (token) {
    // 自动根据角色跳转(路由守卫中实现逻辑)
    router.push('/')
    
  }
})
</script>

<style scoped>
.login-container {
  height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  background: linear-gradient(135deg, #1f4037 0%, #99f2c8 100%);
}

.login-box {
  width: 400px;
  padding: 40px;
  background: rgba(255, 255, 255, 0.9);
  border-radius: 10px;
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
}

.login-box h2 {
  text-align: center;
  color: #1f4037;
  margin-bottom: 30px;
  font-size: 28px;
  font-weight: bold;
  text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
}

.login-form {
  margin-top: 20px;
}

.login-button {
  width: 100%;
  padding: 12px 0;
  font-size: 16px;
}

.register-link {
  text-align: center;
  margin-top: 15px;
  color: #606266;
}

.register-link a {
  color: #409EFF;
  text-decoration: none;
}

.register-link a:hover {
  text-decoration: underline;
}

:deep(.el-input__wrapper) {
  padding: 12px;
}

:deep(.el-input__inner) {
  font-size: 16px;
}

:deep(.el-form-item__error) {
  padding-top: 4px;
}
</style>