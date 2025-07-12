<template>
  <div class="system-container">
    <h1>系统配置</h1>
    <el-card class="config-card">
      <template #header>
        <div class="card-header">
          <span>基本设置</span>
        </div>
      </template>
      
      <el-form label-width="180px">
        <el-form-item label="系统名称">
          <el-input v-model="systemName" />
        </el-form-item>
      </el-form>
    </el-card>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const systemName = ref('污水监测系统')
</script>

<style scoped>
.system-container {
  padding: 20px;
}

.config-card {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
</style>