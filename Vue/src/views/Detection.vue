<template>
  <div class="detection-container">
    <h2>手动检测</h2>
    <el-card class="detection-card">
      <div class="upload-area">
        <el-upload
          class="upload-demo"
          drag
          action="#"
          :auto-upload="false"
          :on-change="handleFileChange"
        >
          <el-icon class="el-icon--upload"><upload-filled /></el-icon>
          <div class="el-upload__text">
            拖拽文件到此处或 <em>点击上传</em>
          </div>
          <template #tip>
            <div class="el-upload__tip">
              支持上传无人机拍摄的图片或视频文件
            </div>
          </template>
        </el-upload>
      </div>
      
      <div class="detection-actions">
        <el-button type="primary" :disabled="!hasFile" @click="startDetection">
          开始检测
        </el-button>
        <el-button @click="resetDetection">重置</el-button>
      </div>
      
      <div v-if="isDetecting" class="detection-progress">
        <el-progress :percentage="detectionProgress" />
        <p>正在进行污水检测分析，请稍候...</p>
      </div>
      
      <div v-if="showResult" class="detection-result">
        <h3>检测结果</h3>
        <el-result
          icon="success"
          title="检测完成"
          sub-title="已成功完成污水检测分析"
        >
          <template #extra>
            <el-button type="primary" @click="viewDetailedReport">查看详细报告</el-button>
          </template>
        </el-result>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { UploadFilled } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

const hasFile = ref(false)
const isDetecting = ref(false)
const detectionProgress = ref(0)
const showResult = ref(false)

const handleFileChange = (file) => {
  hasFile.value = !!file
}

const startDetection = () => {
  isDetecting.value = true
  showResult.value = false
  detectionProgress.value = 0
  
  // 模拟检测进度
  const interval = setInterval(() => {
    detectionProgress.value += 10
    
    if (detectionProgress.value >= 100) {
      clearInterval(interval)
      isDetecting.value = false
      showResult.value = true
    }
  }, 500)
}

const resetDetection = () => {
  hasFile.value = false
  isDetecting.value = false
  detectionProgress.value = 0
  showResult.value = false
}

const viewDetailedReport = () => {
  ElMessage({
    message: '报告功能待实现',
    type: 'info'
  })
}
</script>

<style scoped>
.detection-container {
  padding: 20px;
}

.detection-card {
  margin-top: 20px;
}

.upload-area {
  margin: 20px 0;
}

.detection-actions {
  margin: 20px 0;
  display: flex;
  gap: 10px;
}

.detection-progress {
  margin: 30px 0;
}

.detection-result {
  margin-top: 30px;
  text-align: center;
}
</style>