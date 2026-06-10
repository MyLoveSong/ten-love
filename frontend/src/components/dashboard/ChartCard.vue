<template>
  <el-card class="chart-card enhanced-card" shadow="hover">
    <template #header>
      <div class="card-header">
        <h3 class="card-title">
          <el-icon size="20" color="#1890ff">
            <component :is="headerIcon" />
          </el-icon>
          {{ title }}
        </h3>
        <div class="chart-controls">
          <slot name="controls" />
        </div>
      </div>
    </template>

    <div v-loading="loading" :style="{ height: height + 'px' }">
      <v-chart
        v-if="chartData && chartData.length > 0"
        :option="chartOption"
        :style="{ height: '100%' }"
        @click="handleChartClick"
      />
      <el-empty v-else description="暂无数据" />
    </div>

    <div v-if="showStats" class="chart-stats">
      <slot name="stats" />
    </div>
  </el-card>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart, PieChart as EChartsPieChart, GaugeChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent
} from 'echarts/components'
import VChart from 'vue-echarts'

use([
  CanvasRenderer,
  LineChart,
  EChartsPieChart,
  GaugeChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent
])

interface Props {
  title: string
  headerIcon: string
  chartType: 'line' | 'pie' | 'gauge'
  chartData: any[]
  chartOption: any
  loading?: boolean
  height?: number
  showStats?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  loading: false,
  height: 350,
  showStats: false
})

const emit = defineEmits<{
  chartClick: [params: any]
}>()

const handleChartClick = (params: any) => {
  emit('chartClick', params)
}
</script>

<style scoped>
.chart-card {
  margin-bottom: 24px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0;
}

.card-title {
  margin: 0;
  display: flex;
  align-items: center;
  font-size: 18px;
  font-weight: 600;
  color: #333;
}

.card-title .el-icon {
  margin-right: 8px;
}

.chart-controls {
  display: flex;
  align-items: center;
}

.chart-stats {
  margin-top: 20px;
  padding: 16px;
  background: rgba(255, 255, 255, 0.5);
  border-radius: 8px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .card-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 12px;
  }

  .chart-controls {
    margin-top: 12px;
  }
}
</style>
