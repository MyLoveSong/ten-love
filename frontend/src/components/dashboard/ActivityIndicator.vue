<template>
  <div class="activity-indicator" :class="statusClass">
    <div class="activity-icon-wrapper" :class="iconClass">
      <el-icon :size="iconSize" :class="iconAnimationClass">
        <component :is="iconComponent" />
      </el-icon>
      <div v-if="showPulse" class="pulse-ring"></div>
    </div>
    <div class="activity-content">
      <div class="activity-title">{{ title }}</div>
      <div class="activity-value">{{ formattedValue }}</div>
      <div class="activity-description">{{ description }}</div>
      <div v-if="trend" class="activity-trend" :class="trendClass">
        <el-icon size="12"><TrendCharts /></el-icon>
        <span>{{ trend }}</span>
      </div>
    </div>
    <div class="activity-status-badge" :class="statusClass">
      {{ statusText }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { TrendCharts, Heart, Lightning, User, Monitor, Warning } from '@element-plus/icons-vue'

interface Props {
  title: string
  value: number | string
  description: string
  status: 'active' | 'inactive' | 'warning' | 'danger'
  icon: string
  trend?: string
  iconSize?: number
  showPulse?: boolean
  formatType?: 'number' | 'percentage' | 'time' | 'text'
}

const props = withDefaults(defineProps<Props>(), {
  iconSize: 24,
  showPulse: true,
  formatType: 'number'
})

const iconComponent = computed(() => {
  const iconMap: Record<string, any> = {
    Heart,
    Lightning,
    User,
    Monitor,
    Warning
  }
  return iconMap[props.icon] || Monitor
})

const statusClass = computed(() => {
  return `status-${props.status}`
})

const iconClass = computed(() => {
  return `icon-${props.status}`
})

const iconAnimationClass = computed(() => {
  if (props.status === 'active' && props.showPulse) {
    return 'pulse-animation'
  }
  return ''
})

const statusText = computed(() => {
  const statusMap = {
    active: '活跃',
    inactive: '离线',
    warning: '警告',
    danger: '危险'
  }
  return statusMap[props.status]
})

const trendClass = computed(() => {
  if (props.trend?.includes('+')) return 'trend-positive'
  if (props.trend?.includes('-')) return 'trend-negative'
  return 'trend-neutral'
})

const formattedValue = computed(() => {
  if (typeof props.value === 'string') return props.value

  switch (props.formatType) {
    case 'percentage':
      return `${props.value}%`
    case 'time':
      return `${props.value}分钟`
    case 'text':
      return props.value.toString()
    case 'number':
    default:
      return formatNumber(props.value)
  }
})

const formatNumber = (num: number): string => {
  if (num >= 1000000) {
    return (num / 1000000).toFixed(1) + 'M'
  } else if (num >= 1000) {
    return (num / 1000).toFixed(1) + 'K'
  }
  return num.toString()
}
</script>

<style scoped>
.activity-indicator {
  display: flex;
  align-items: center;
  padding: 16px;
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border-radius: 12px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.activity-indicator:hover {
  background: rgba(255, 255, 255, 0.15);
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
}

.activity-icon-wrapper {
  position: relative;
  width: 48px;
  height: 48px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 16px;
  transition: all 0.3s ease;
}

.icon-active {
  background: linear-gradient(135deg, #52c41a, #73d13d);
  color: white;
}

.icon-inactive {
  background: linear-gradient(135deg, #d9d9d9, #f0f0f0);
  color: #666;
}

.icon-warning {
  background: linear-gradient(135deg, #faad14, #ffc53d);
  color: white;
}

.icon-danger {
  background: linear-gradient(135deg, #ff4d4f, #ff7875);
  color: white;
}

.pulse-animation {
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% {
    transform: scale(1);
    opacity: 1;
  }
  50% {
    transform: scale(1.1);
    opacity: 0.8;
  }
}

.pulse-ring {
  position: absolute;
  top: 50%;
  left: 50%;
  width: 100%;
  height: 100%;
  border: 2px solid currentColor;
  border-radius: 50%;
  transform: translate(-50%, -50%);
  animation: pulse-ring 2s ease-out infinite;
  opacity: 0;
}

@keyframes pulse-ring {
  0% {
    transform: translate(-50%, -50%) scale(0.8);
    opacity: 1;
  }
  100% {
    transform: translate(-50%, -50%) scale(1.4);
    opacity: 0;
  }
}

.activity-content {
  flex: 1;
}

.activity-title {
  font-size: 14px;
  font-weight: 600;
  color: #333;
  margin-bottom: 4px;
}

.activity-value {
  font-size: 20px;
  font-weight: 700;
  color: #333;
  margin-bottom: 2px;
}

.activity-description {
  font-size: 12px;
  color: #666;
  margin-bottom: 8px;
}

.activity-trend {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  font-weight: 500;
}

.trend-positive {
  color: #52c41a;
}

.trend-negative {
  color: #ff4d4f;
}

.trend-neutral {
  color: #666;
}

.activity-status-badge {
  position: absolute;
  top: 8px;
  right: 8px;
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.status-active {
  background: rgba(82, 196, 26, 0.1);
  color: #52c41a;
  border: 1px solid rgba(82, 196, 26, 0.2);
}

.status-inactive {
  background: rgba(217, 217, 217, 0.1);
  color: #666;
  border: 1px solid rgba(217, 217, 217, 0.2);
}

.status-warning {
  background: rgba(250, 173, 20, 0.1);
  color: #faad14;
  border: 1px solid rgba(250, 173, 20, 0.2);
}

.status-danger {
  background: rgba(255, 77, 79, 0.1);
  color: #ff4d4f;
  border: 1px solid rgba(255, 77, 79, 0.2);
}

/* 响应式设计 */
@media (max-width: 768px) {
  .activity-indicator {
    flex-direction: column;
    text-align: center;
    padding: 12px;
  }

  .activity-icon-wrapper {
    margin-right: 0;
    margin-bottom: 12px;
  }

  .activity-value {
    font-size: 18px;
  }

  .activity-status-badge {
    position: static;
    margin-top: 8px;
    align-self: center;
  }
}
</style>
