<template>
  <div
    class="stat-card-enhanced"
    :class="{ 'clickable': clickable }"
    @click="handleClick"
  >
    <div class="stat-icon-wrapper" :class="iconClass">
      <el-icon :size="iconSize">
        <component :is="icon" />
      </el-icon>
    </div>
    <div class="stat-content">
      <div class="stat-value">{{ formattedValue }}</div>
      <div class="stat-label">{{ label }}</div>
      <div class="stat-trend" v-if="trend" :class="trendClass">
        <el-icon size="14" :color="trendColor">
          <TrendCharts />
        </el-icon>
        <span class="trend-text" :class="trendClass">{{ trend }}</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { TrendCharts } from '@element-plus/icons-vue'

interface Props {
  value: number | string
  label: string
  icon: string
  iconClass: string
  iconSize?: number
  trend?: string
  trendClass?: 'positive' | 'negative' | 'neutral'
  clickable?: boolean
  formatType?: 'number' | 'percentage' | 'currency'
}

const props = withDefaults(defineProps<Props>(), {
  iconSize: 32,
  trendClass: 'positive',
  clickable: false,
  formatType: 'number'
})

const emit = defineEmits<{
  click: []
}>()

const formattedValue = computed(() => {
  if (typeof props.value === 'string') return props.value

  switch (props.formatType) {
    case 'percentage':
      return `${props.value}%`
    case 'currency':
      return `¥${props.value.toLocaleString()}`
    case 'number':
    default:
      return formatNumber(props.value)
  }
})

const trendColor = computed(() => {
  switch (props.trendClass) {
    case 'positive':
      return '#52c41a'
    case 'negative':
      return '#ff4d4f'
    case 'neutral':
    default:
      return '#666'
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

const handleClick = () => {
  if (props.clickable) {
    emit('click')
  }
}
</script>

<style scoped>
.stat-card-enhanced {
  display: flex;
  align-items: center;
  padding: 20px;
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(255, 255, 255, 0.7) 100%);
  border-radius: 12px;
  border: 1px solid rgba(255, 255, 255, 0.3);
  transition: all 0.3s ease;
}

.stat-card-enhanced.clickable {
  cursor: pointer;
}

.stat-card-enhanced:hover {
  transform: translateY(-4px);
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
  background: linear-gradient(135deg, rgba(255, 255, 255, 1) 0%, rgba(255, 255, 255, 0.9) 100%);
}

.stat-icon-wrapper {
  width: 60px;
  height: 60px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  transition: all 0.3s ease;
  margin-right: 16px;
}

.stat-icon-wrapper.users {
  background: linear-gradient(135deg, #1890ff, #40a9ff);
}

.stat-icon-wrapper.predictions {
  background: linear-gradient(135deg, #52c41a, #73d13d);
}

.stat-icon-wrapper.activity {
  background: linear-gradient(135deg, #fa8c16, #ffc53d);
}

.stat-icon-wrapper.glucose {
  background: linear-gradient(135deg, #1890ff, #40a9ff);
}

.stat-icon-wrapper.health {
  background: linear-gradient(135deg, #52c41a, #73d13d);
}

.stat-icon-wrapper.recipe {
  background: linear-gradient(135deg, #fa8c16, #ffc53d);
}

.stat-icon-wrapper.image {
  background: linear-gradient(135deg, #722ed1, #9254de);
}

.stat-icon-wrapper.cultural {
  background: linear-gradient(135deg, #13c2c2, #36cfc9);
}

.stat-icon-wrapper.merl {
  background: linear-gradient(135deg, #eb2f96, #f759ab);
}

.stat-content {
  flex: 1;
}

.stat-value {
  font-size: 28px;
  font-weight: 700;
  color: #333;
  line-height: 1;
  margin-bottom: 4px;
}

.stat-label {
  font-size: 14px;
  color: #666;
  margin-bottom: 8px;
}

.stat-trend {
  display: flex;
  align-items: center;
  gap: 4px;
}

.trend-text {
  font-size: 12px;
  font-weight: 500;
}

.trend-text.positive {
  color: #52c41a;
}

.trend-text.negative {
  color: #ff4d4f;
}

.trend-text.neutral {
  color: #666;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .stat-card-enhanced {
    flex-direction: column;
    text-align: center;
  }

  .stat-icon-wrapper {
    margin-right: 0;
    margin-bottom: 12px;
  }

  .stat-value {
    font-size: 24px;
  }
}
</style>
