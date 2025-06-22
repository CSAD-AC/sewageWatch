<template>
  <div class="register-container">
    <div class="register-box">
      <h2>用户注册</h2>
      <el-form
        ref="registerFormRef"
        :model="registerForm"
        :rules="registerRules"
        class="register-form"
        @keyup.enter="handleRegister"
      >
        <el-form-item prop="userName">
          <el-input
            v-model="registerForm.userName"
            placeholder="请输入账号"
            :prefix-icon="User"
            clearable
          />
        </el-form-item>
        <el-form-item prop="name">
          <el-input
            v-model="registerForm.name"
            placeholder="请输入昵称"
            :prefix-icon="UserFilled"
            clearable
          />
        </el-form-item>
        <el-form-item prop="password">
          <el-input
            v-model="registerForm.password"
            type="password"
            placeholder="请输入密码"
            :prefix-icon="Lock"
            show-password
          />
        </el-form-item>
        <el-form-item prop="confirmPassword">
          <el-input
            v-model="registerForm.confirmPassword"
            type="password"
            placeholder="请确认密码"
            :prefix-icon="Key"
            show-password
          />
        </el-form-item>
        <el-form-item>
          <el-button
            type="primary"
            :loading="loading"
            class="register-button"
            @click="handleRegister"
          >
            注册
          </el-button>
        </el-form-item>
        <div class="login-link">
          已有账号？
          <router-link to="/login">立即登录</router-link>
        </div>
      </el-form>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { User, UserFilled, Lock, Key } from '@element-plus/icons-vue'
import request from '../utils/request'

const router = useRouter()
const registerFormRef = ref(null)
const loading = ref(false)

const registerForm = reactive({
  userName: '',
  name: '',
  password: '',
  confirmPassword: ''
})

// 自定义密码验证规则
const validatePass = (rule, value, callback) => {
  if (value === '') {
    callback(new Error('请再次输入密码'))
  } else if (value !== registerForm.password) {
    callback(new Error('两次输入密码不一致'))
  } else {
    callback()
  }
}

// 表单验证规则
const registerRules = {
  userName: [
    { required: true, message: '请输入账号', trigger: 'blur' },
    { min: 3, max: 20, message: '长度在 3 到 20 个字符', trigger: 'blur' }
  ],
  name: [
    { required: true, message: '请输入昵称', trigger: 'blur' },
    { min: 2, max: 10, message: '长度在 2 到 10 个字符', trigger: 'blur' }
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 6, max: 20, message: '长度在 6 到 20 个字符', trigger: 'blur' }
  ],
  confirmPassword: [
    { required: true, message: '请确认密码', trigger: 'blur' },
    { validator: validatePass, trigger: 'blur' }
  ]
}

// 清空表单
const resetForm = () => {
  if (registerFormRef.value) {
    registerFormRef.value.resetFields()
  }
}

// 处理注册逻辑
const handleRegister = async () => {
  if (!registerFormRef.value) return
  
  try {
    // 表单验证
    await registerFormRef.value.validate()
    loading.value = true

    // 发送注册请求
    const res = await request({
      url: '/register',
      method: 'post',
      data: {
        userName: registerForm.userName,
        name: registerForm.name,
        password: registerForm.password
      }
    })

    // 注册成功
    ElMessage.success({
      message: '注册成功，请登录',
      duration: 800,
      onClose: () => {
        router.push('/login')
      }
    })
  } catch (error) {
    console.error('注册失败:', error)
    // 错误处理已在request.js中统一处理
    resetForm()
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.register-container {
  height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  background: linear-gradient(135deg, #1f4037 0%, #99f2c8 100%);
  background-size: 400% 400%;
  animation: gradientBG 15s ease infinite;
}

@keyframes gradientBG {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

.register-box {
  width: 400px;
  padding: 40px;
  background: rgba(255, 255, 255, 0.95);
  border-radius: 12px;
  box-shadow: 0 12px 30px rgba(0, 0, 0, 0.15);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.3);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.register-box:hover {
  transform: translateY(-5px);
  box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2);
}

.register-box h2 {
  text-align: center;
  color: #1f4037;
  margin-bottom: 30px;
  font-size: 28px;
  font-weight: 600;
  letter-spacing: 1px;
  text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
}

.register-form {
  margin-top: 20px;
}

.register-button {
  width: 100%;
  padding: 12px 0;
  font-size: 16px;
  font-weight: 500;
  letter-spacing: 1px;
  background: linear-gradient(to right, #1f4037, #99f2c8);
  border: none;
  transition: all 0.3s ease;
}

.register-button:hover {
  background: linear-gradient(to right, #2a4d42, #a8f5d0);
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(31, 64, 55, 0.2);
}

.login-link {
  text-align: center;
  margin-top: 20px;
  color: #5a6c5d;
  font-size: 14px;
}

.login-link a {
  color: #2a4d42;
  text-decoration: none;
  font-weight: 500;
  transition: color 0.3s ease;
}

.login-link a:hover {
  color: #1f4037;
  text-decoration: underline;
}

:deep(.el-input__wrapper) {
  padding: 12px 15px;
  border-radius: 8px;
  transition: border-color 0.3s ease;
}

:deep(.el-input__wrapper:hover) {
  box-shadow: 0 0 0 1px #99f2c8 inset;
}

:deep(.el-input__inner) {
  font-size: 15px;
  color: #2a4d42;
}

:deep(.el-form-item__error) {
  padding-top: 5px;
  font-size: 13px;
}
</style>