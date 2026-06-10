<template>
  <div class="data-refresh-manager">
    <div class="refresh-controls">
      <el-button-group size="small">
        <el-button
          :type="autoRefresh ? 'primary' : 'default'"
          @click="toggleAutoRefresh"
          :disabled="loading"
        >
          <el-icon><Refresh /></el-icon>
          {{ autoRefresh ? '自动刷新' : '手动刷新' }}
        </el-button>
        <el-button
          v-if="autoRefresh"
          @click="changeRefreshInterval"
          :disabled="loading"
        >
          {{ refreshInterval / 1000 }}秒
        </el-button>
      </el-button-group>

      <div class="refresh-status">
        <el-tag
          :type="getStatusType(lastRefreshStatus)"
          size="small"
          class="status-tag"
        >
          {{ getStatusText(lastRefreshStatus) }}
        </el-tag>
        <span class="last-refresh-time">{{ lastRefreshTime }}</span>
      </div>
    </div>

    <div v-if="showProgress" class="refresh-progress">
      <el-progress
        :percentage="refreshProgress"
        :show-text="false"
        :stroke-width="4"
        color="#1890ff"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { Refresh } from '@element-plus/icons-vue'

interface Props {
  refreshInterval?: number
  showProgress?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  refreshInterval: 30000, // 30秒
  showProgress: true
})

const emit = defineEmits<{
  refresh: []
  intervalChange: [interval: number]
}>()

// 响应式数据
const autoRefresh = ref(true)
const loading = ref(false)
const refreshProgress = ref(0)
const lastRefreshStatus = ref<'success' | 'error' | 'pending'>('success')
const lastRefreshTime = ref('')
const refreshInterval = ref(props.refreshInterval)

let intervalId: number | null = null
let progressIntervalId: number | null = null

// 计算属性
const getStatusType = (status: string) => {
  switch (status) {
    case 'success':
      return 'success'
    case 'error':
      return 'danger'
    case 'pending':
      return 'warning'
    default:
      return 'info'
  }
}

const getStatusText = (status: string) => {
  switch (status) {
    case 'success':
      return '刷新成功'
    case 'error':
      return '刷新失败'
    case 'pending':
      return '刷新中'
    default:
      return '未知状态'
  }
}

// 方法
const updateTime = () => {
  const now = new Date()
  lastRefreshTime.value = now.toLocaleTimeString('zh-CN', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  })
}

const startProgress = () => {
  refreshProgress.value = 0
  progressIntervalId = setInterval(() => {
    if (refreshProgress.value < 90) {
      refreshProgress.value += 2
    }
  }, refreshInterval.value / 50)
}

const stopProgress = () => {
  if (progressIntervalId) {
    clearInterval(progressIntervalId)
    progressIntervalId = null
  }
  refreshProgress.value = 100
  setTimeout(() => {
    refreshProgress.value = 0
  }, 500)
}

const performRefresh = async () => {
  if (loading.value) return

  loading.value = true
  lastRefreshStatus.value = 'pending'
  startProgress()

  try {
    emit('refresh')
    lastRefreshStatus.value = 'success'
    updateTime()
  } catch (error) {
    lastRefreshStatus.value = 'error'
    console.error('刷新失败:', error)
  } finally {
    loading.value = false
    stopProgress()
  }
}

const toggleAutoRefresh = () => {
  autoRefresh.value = !autoRefresh.value

  if (autoRefresh.value) {
    startAutoRefresh()
  } else {
    stopAutoRefresh()
  }
}

const changeRefreshInterval = () => {
  const intervals = [10000, 30000, 60000, 300000] // 10秒, 30秒, 1分钟, 5分钟
  const currentIndex = intervals.indexOf(refreshInterval.value)
  const nextIndex = (currentIndex + 1) % intervals.length
  refreshInterval.value = intervals[nextIndex]

  emit('intervalChange', refreshInterval.value)

  if (autoRefresh.value) {
    stopAutoRefresh()
    startAutoRefresh()
  }
}

const startAutoRefresh = () => {
  if (intervalId) {
    clearInterval(intervalId)
  }

  intervalId = setInterval(() => {
    performRefresh()
  }, refreshInterval.value)
}

const stopAutoRefresh = () => {
  if (intervalId) {
    clearInterval(intervalId)
    intervalId = null
  }
}

// 生命周期
onMounted(() => {
  updateTime()
  if (autoRefresh.value) {
    startAutoRefresh()
  }
})

onUnmounted(() => {
  stopAutoRefresh()
  stopProgress()
})
</script>

<style scoped>
.data-refresh-manager {
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border-radius: 8px;
  padding: 12px;
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.refresh-controls {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.refresh-status {
  display: flex;
  align-items: center;
  gap: 8px;
}

.status-tag {
  font-size: 10px;
}

.last-refresh-time {
  font-size: 10px;
  color: #666;
  font-family: monospace;
}

.refresh-progress {
  margin-top: 8px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .refresh-controls {
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
  }

  .refresh-status {
    align-self: flex-end;
  }
}
</style>
