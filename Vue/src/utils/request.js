import axios from 'axios'
import { ElMessage } from 'element-plus'
import router from '../router'

// 创建axios实例
const service = axios.create({
  baseURL: 'http://localhost:8080',
  timeout: 10000 // 请求超时时间
})

// 请求拦截器
service.interceptors.request.use(
  config => {
    // 从localStorage获取token
    const token = localStorage.getItem('token')
    if (token) {
      // 设置请求头携带token
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  error => {
    console.error('请求错误:', error)
    return Promise.reject(error)
  }
)

// 响应拦截器
service.interceptors.response.use(
  response => {
    const res = response.data
    
    // 如果返回的状态码不是200，说明接口请求失败
    if (res.code !== 200) {
      if (res.code === 401 && router.currentRoute.value.path === '/login') {
        ElMessage.error('账号或密码错误')
      } else {
        ElMessage.error(res.msg || '请求失败')
      }
      
      // 401: 未登录或token过期
      if (res.code === 401) {
        // 清除本地token
        localStorage.removeItem('token')
        localStorage.removeItem('userInfo')
        
        // 跳转到登录页
        router.replace('/login')
      }
      
      return Promise.reject(new Error(res.msg || '请求失败'))
    } else {
      return res
    }
  },
  error => {
    console.error('响应错误:', error)
    
    // 处理401错误
    if (error.response && error.response.status === 401) {
      localStorage.removeItem('token')
      localStorage.removeItem('userInfo')

      if (router.currentRoute.value.path === '/login') {
        ElMessage.error('账号或密码错误')
      } else {
        ElMessage.error('登录已过期，请重新登录')
        router.replace('/login')
      }
    } else {
      // 其他错误
      ElMessage.error(error.response?.data?.msg || '服务器异常，请稍后重试')
    }
    
    return Promise.reject(error)
  }
)

export default service