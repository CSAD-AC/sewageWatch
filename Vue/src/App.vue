<template>
  <div class="app-container">
    <router-view />
  </div>
</template>

<style>
/* 重置默认样式 */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html, body {
  height: 100%;
  width: 100%;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

#app, .app-container {
  height: 100vh;
  width: 100%;
}

/* 滚动条样式优化 */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 4px;
}

::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: #a8a8a8;
}

/* Element Plus 样式优化 */
.el-card {
  --el-card-padding: 16px;
}

.el-table {
  --el-table-header-bg-color: #f5f7fa;
}

h2 {
  margin-bottom: 20px;
  font-weight: 500;
  color: var(--el-text-color-primary);
}
</style>