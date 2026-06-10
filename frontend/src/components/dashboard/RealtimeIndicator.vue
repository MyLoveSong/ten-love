<template>
  <div class="realtime-indicator" :class="statusClass">
    <div class="indicator-icon">
      <el-icon :size="16" :class="iconClass">
        <component :is="iconComponent" />
      </el-icon>
      <div v-if="isConnected" class="pulse-dot"></div>
    </div>
    <div class="indicator-content">
      <div class="indicator-label">{{ label }}</div>
      <div class="indicator-status">{{ statusText }}</div>
    </div>
    <div class="indicator-time">
      {{ lastUpdateTime }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { Monitor, Warning, CircleCheck } from '@element-plus/icons-vue'

interface Props {
  label?: string
  updateInterval?: number
}

const props = withDefaults(defineProps<Props>(), {
  label: '实时数据',
  updateInterval: 5000
})

const isConnected = ref(true)
const lastUpdateTime = ref('')
const connectionStatus = ref<'connected' | 'disconnected' | 'error'>('connected')

const statusClass = computed(() => {
  return `status-${connectionStatus.value}`
})

const iconClass = computed(() => {
  return `icon-${connectionStatus.value}`
})

const iconComponent = computed(() => {
  switch (connectionStatus.value) {
    case 'connected':
      return CircleCheck
    case 'disconnected':
      return Warning
    case 'error':
      return Warning
    default:
      return Monitor
  }
})

const statusText = computed(() => {
  switch (connectionStatus.value) {
    case 'connected':
      return '已连接'
    case 'disconnected':
      return '连接断开'
    case 'error':
      return '连接错误'
    default:
      return '未知状态'
  }
})

const updateTime = () => {
  const now = new Date()
  lastUpdateTime.value = now.toLocaleTimeString('zh-CN', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  })
}

const simulateConnectionStatus = () => {
  // 模拟连接状态变化
  const random = Math.random()
  if (random > 0.9) {
    connectionStatus.value = 'disconnected'
    isConnected.value = false
  } else if (random > 0.8) {
    connectionStatus.value = 'error'
    isConnected.value = false
  } else {
    connectionStatus.value = 'connected'
    isConnected.value = true
  }
  updateTime()
}

let intervalId: number | null = null

onMounted(() => {
  updateTime()
  simulateConnectionStatus()

  intervalId = setInterval(() => {
    simulateConnectionStatus()
  }, props.updateInterval)
})

onUnmounted(() => {
  if (intervalId) {
    clearInterval(intervalId)
  }
})
</script>

<style scoped>
.realtime-indicator {
  display: flex;
  align-items: center;
  padding: 8px 12px;
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  transition: all 0.3s ease;
  font-size: 12px;
}

.realtime-indicator:hover {
  background: rgba(255, 255, 255, 0.15);
}

.indicator-icon {
  position: relative;
  margin-right: 8px;
}

.pulse-dot {
  position: absolute;
  top: -2px;
  right: -2px;
  width: 6px;
  height: 6px;
  background-color: #52c41a;
  border-radius: 50%;
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
    transform: scale(1);
  }
  50% {
    opacity: 0.5;
    transform: scale(1.2);
  }
}

.indicator-content {
  flex: 1;
}

.indicator-label {
  font-weight: 500;
  color: #333;
  margin-bottom: 2px;
}

.indicator-status {
  font-size: 10px;
  color: #666;
}

.indicator-time {
  font-size: 10px;
  color: #999;
  font-family: monospace;
}

/* 状态样式 */
.status-connected {
  border-left: 3px solid #52c41a;
}

.status-connected .indicator-status {
  color: #52c41a;
}

.status-disconnected {
  border-left: 3px solid #faad14;
}

.status-disconnected .indicator-status {
  color: #faad14;
}

.status-error {
  border-left: 3px solid #ff4d4f;
}

.status-error .indicator-status {
  color: #ff4d4f;
}

/* 图标样式 */
.icon-connected {
  color: #52c41a;
}

.icon-disconnected {
  color: #faad14;
}

.icon-error {
  color: #ff4d4f;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .realtime-indicator {
    padding: 6px 8px;
    font-size: 11px;
  }

  .indicator-time {
    display: none;
  }
}
</style>
