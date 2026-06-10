<template>
  <div class="page-container">
    <div class="page-header">
      <h2 class="page-title">
        <el-icon size="24" :color="iconColor">
          <component :is="iconComponent" />
        </el-icon>
        {{ title }}
      </h2>
    </div>

    <el-card class="content-card" shadow="hover">
      <el-space direction="vertical" style="width: 100%;">
        <h4>功能说明</h4>
        <p>{{ description }}</p>

        <h4>主要特性</h4>
        <ul>
          <li v-for="feature in features" :key="feature">{{ feature }}</li>
        </ul>

        <el-button
          type="primary"
          size="large"
          @click="handleAction"
          :loading="isLoading"
        >
          {{ buttonText }}
        </el-button>
      </el-space>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { ElMessage } from 'element-plus'

// Props定义
const props = defineProps<{
  title: string
  description: string
  features: string[]
  buttonText: string
  iconComponent: string
  iconColor: string
}>()

// 响应式数据
const isLoading = ref(false)

// 方法
const handleAction = async () => {
  isLoading.value = true

  try {
    // 模拟API调用
    await new Promise(resolve => setTimeout(resolve, 2000))
    ElMessage.success(`${props.title}功能启动成功`)
  } catch (error) {
    ElMessage.error('操作失败，请重试')
  } finally {
    isLoading.value = false
  }
}
</script>

<style scoped>
.page-container {
  padding: 0;
}

.page-header {
  margin-bottom: 24px;
}

.page-title {
  margin: 0;
  display: flex;
  align-items: center;
  font-size: 24px;
  font-weight: bold;
}

.content-card {
  padding: 24px;
  transition: all 0.3s ease;
}

.content-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
}

.content-card h4 {
  margin: 16px 0 8px 0;
  color: #333;
  font-weight: 600;
}

.content-card p {
  margin: 8px 0;
  color: #666;
  line-height: 1.6;
}

.content-card ul {
  margin: 8px 0;
  padding-left: 20px;
}

.content-card li {
  margin: 4px 0;
  color: #666;
  line-height: 1.5;
}

.el-button {
  margin-top: 16px;
  transition: all 0.3s ease;
}

.el-button:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(24, 144, 255, 0.3);
}

/* 响应式设计 */
@media (max-width: 768px) {
  .page-title {
    font-size: 20px;
  }

  .content-card {
    padding: 16px;
  }
}
</style>