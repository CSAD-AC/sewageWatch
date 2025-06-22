import { createRouter, createWebHistory } from 'vue-router'
import Layout from '../components/layout/layout.vue'
import { ElMessage } from 'element-plus'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/login',
      name: 'Login',
      component: () => import('../views/Login.vue'),
      meta: {
        requiresAuth: false
      }
    },
    {
      path: '/register',
      name: 'Register',
      component: () => import('../views/Register.vue'),
      meta: {
        requiresAuth: false
      }
    },
    {
      path: '/',
      component: Layout,
      // 移除静态重定向，将在路由守卫中动态处理
      children: [
        {
          path: 'monitor',
          name: 'Monitor',
          component: () => import('../views/Monitor.vue'),
          meta: {
            requiresAuth: true,
            roles: ['INSPECTOR', 'ADMIN']
          }
        },
        {
          path: 'tasks',
          name: 'Tasks',
          component: () => import('../views/Tasks.vue'),
          meta: {
            requiresAuth: true,
            roles: ['INSPECTOR', 'ADMIN']
          }
        },
        {
          path: 'detection',
          name: 'Detection',
          component: () => import('../views/Detection.vue'),
          meta: {
            requiresAuth: true,
            roles: ['PUBLIC', 'INSPECTOR', 'ADMIN']
          }
        },
        {
          path: 'history',
          name: 'History',
          component: () => import('../views/History.vue'),
          meta: {
            requiresAuth: true,
            roles: ['PUBLIC', 'INSPECTOR', 'ADMIN']
          }
        },
        {
          path: 'analysis',
          name: 'Analysis',
          component: () => import('../views/Analysis.vue'),
          meta: {
            requiresAuth: true,
            roles: ['PUBLIC', 'INSPECTOR', 'ADMIN']
          }
        }
      ]
    },
    {
      path: '/admin',
      component: Layout,
      meta: {
        requiresAuth: true,
        roles: ['ADMIN']
      },
      children: [
        {
          path: 'users',
          name: 'UserManagement',
          component: () => import('../views/admin/Users.vue'),
          meta: {
            requiresAuth: true,
            roles: ['ADMIN']
          }
        },
        {
          path: 'system',
          name: 'SystemConfig',
          component: () => import('../views/admin/System.vue'),
          meta: {
            requiresAuth: true,
            roles: ['ADMIN']
          }
        }
      ]
    },
    {
      // 捕获所有未匹配的路由，重定向到首页
      path: '/:pathMatch(.*)*',
      redirect: '/'
    }
  ]
})

// 解析Base64URL编码的字符串
function parseBase64Url(base64Url) {
  // 将Base64URL转换为标准Base64
  const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
  // 解码Base64
  const decoded = atob(base64);
  // 将解码后的字符串转换为JSON对象
  try {
    return JSON.parse(decoded);
  } catch (e) {
    console.error('解析Base64URL失败:', e);
    return {};
  }
}

// 从JWT token中提取payload
function parseJwtToken(token) {
  if (!token) return {};
  
  try {
    // JWT token格式: header.payload.signature
    const parts = token.split('.');
    if (parts.length !== 3) {
      console.error('无效的JWT token格式');
      return {};
    }
    
    // 解析payload部分（第二部分）
    return parseBase64Url(parts[1]);
  } catch (error) {
    console.error('解析JWT token失败:', error);
    return {};
  }
}

// 路由守卫
router.beforeEach((to, from, next) => {
  try {
    console.log('路由守卫触发:', to.path)
    
    // 安全获取token
    const token = localStorage.getItem('token')
    console.log('当前token状态:', token ? '已存在' : '不存在')
    
    // 从token中解析用户信息
    let userRole = null
    
    try {
      if (token) {
        const payload = parseJwtToken(token);
        userRole = payload.role;
        console.log('从token解析的用户信息:', payload);
      }
    } catch (error) {
      console.error('解析token失败:', error);
      // 出错时使用默认值继续
    }
    
    console.log('当前用户角色:', userRole);
    // 未登录，且访问/
    if(!token && to.path === '/') {
      console.log('未登录，跳转到登录页')
      next('/login')
      return
    }

    // 不需要认证的页面
    if (!to.meta.requiresAuth) {
      // 处理根路径的动态重定向
      if (to.path === '/') {
        console.log('访问根路径，根据角色重定向')
        if (userRole === 'PUBLIC') {
          next('/detection')
        } else if (userRole === 'INSPECTOR' || userRole === 'ADMIN') {
          next('/monitor')
        } else {
          // 未知角色默认跳转到detection
          console.warn('未知角色类型:', userRole)
          next('/detection')
        }
        return
      }
      console.log('访问无需认证页面')
      // 如果已登录且试图访问登录/注册页，重定向到首页
      if (token && ['/login', '/register'].includes(to.path)) {
        console.log('已登录用户重定向到首页')
        next('/')
        return
      }
      next()
      return
    }

    // 需要认证但没有token
    if (!token) {
      console.log('需要认证但无token，重定向到登录页')
      ElMessage.warning('请先登录')
      next('/login')
      return
    }

    
    // 检查角色权限
    if (to.meta.roles && Array.isArray(to.meta.roles) && !to.meta.roles.includes(userRole)) {
      console.log('用户角色无权限访问该页面')
      ElMessage.error('您没有权限访问该页面')
      
      // 根据角色重定向到合适的页面
      if (userRole === 'PUBLIC') {
        next('/detection')
      } else if (userRole === 'INSPECTOR' || userRole === 'ADMIN') {
        next('/monitor')
      } else {
        // 未知角色默认跳转到detection
        console.warn('未知角色类型:', userRole)
        next('/detection')
      }
      return
    }

    console.log('权限检查通过，允许访问:', to.path)
    next()


  } catch (error) {
    console.error('路由守卫出错:', error)
    // 发生错误时，重定向到安全页面
    ElMessage.error('系统出错，请重新登录')
    localStorage.removeItem('token')
    localStorage.removeItem('userInfo')
    next('/login')
  }
})

export default router