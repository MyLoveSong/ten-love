<template>
  <div class="health-gauge-container">
    <div class="gauge-wrapper">
      <el-progress
        type="circle"
        :percentage="healthValue"
        :color="gaugeColors"
        :width="gaugeSize"
        :stroke-width="strokeWidth"
        class="health-gauge"
      />
      <div class="gauge-center">
        <div class="gauge-value">{{ healthValue }}%</div>
        <div class="gauge-label">{{ label }}</div>
      </div>
    </div>
    <div class="gauge-indicators">
      <div
        v-for="indicator in indicators"
        :key="indicator.label"
        class="indicator-item"
        :class="{ 'active': isIndicatorActive(indicator) }"
      >
        <div class="indicator-dot" :class="indicator.type"></div>
        <span>{{ indicator.label }}</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

interface Indicator {
  label: string
  min: number
  max: number
  type: 'healthy' | 'warning' | 'danger'
}

interface Props {
  healthValue: number
  label: string
  gaugeSize?: number
  strokeWidth?: number
  indicators?: Indicator[]
}

const props = withDefaults(defineProps<Props>(), {
  gaugeSize: 120,
  strokeWidth: 12,
  indicators: () => [
    { label: '健康 (90-100%)', min: 90, max: 100, type: 'healthy' },
    { label: '警告 (70-89%)', min: 70, max: 89, type: 'warning' },
    { label: '危险 (<70%)', min: 0, max: 69, type: 'danger' }
  ]
})

const gaugeColors = computed(() => {
  if (props.healthValue >= 90) {
    return [
      { color: '#52c41a', percentage: 20 },
      { color: '#73d13d', percentage: 100 }
    ]
  } else if (props.healthValue >= 70) {
    return [
      { color: '#faad14', percentage: 20 },
      { color: '#ffc53d', percentage: 100 }
    ]
  } else {
    return [
      { color: '#ff4d4f', percentage: 20 },
      { color: '#ff7875', percentage: 100 }
    ]
  }
})

const isIndicatorActive = (indicator: Indicator): boolean => {
  return props.healthValue >= indicator.min && props.healthValue <= indicator.max
}
</script>

<style scoped>
.health-gauge-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px;
}

.gauge-wrapper {
  position: relative;
  margin-bottom: 20px;
}

.gauge-center {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
}

.gauge-value {
  font-size: 24px;
  font-weight: 700;
  color: #333;
  line-height: 1;
}

.gauge-label {
  font-size: 12px;
  color: #666;
  margin-top: 4px;
}

.gauge-indicators {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.indicator-item {
  display: flex;
  align-items: center;
  font-size: 12px;
  color: #666;
  transition: all 0.3s ease;
}

.indicator-item.active {
  color: #333;
  font-weight: 500;
}

.indicator-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  margin-right: 8px;
  transition: all 0.3s ease;
}

.indicator-dot.healthy {
  background-color: #52c41a;
}

.indicator-dot.warning {
  background-color: #faad14;
}

.indicator-dot.danger {
  background-color: #ff4d4f;
}

.indicator-item.active .indicator-dot {
  transform: scale(1.2);
  box-shadow: 0 0 8px rgba(0, 0, 0, 0.2);
}

/* 响应式设计 */
@media (max-width: 768px) {
  .gauge-indicators {
    flex-direction: row;
    flex-wrap: wrap;
    justify-content: center;
  }

  .gauge-value {
    font-size: 20px;
  }
}
</style>
