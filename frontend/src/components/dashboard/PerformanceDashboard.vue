<template>
  <div class="performance-dashboard">
    <div class="dashboard-header">
      <h3 class="dashboard-title">
        <el-icon size="20" color="#1890ff"><Monitor /></el-icon>
        {{ title }}
      </h3>
      <div class="dashboard-controls">
        <el-button-group size="small">
          <el-button
            :type="timeRange === '1h' ? 'primary' : 'default'"
            @click="timeRange = '1h'"
          >
            1小时
          </el-button>
          <el-button
            :type="timeRange === '24h' ? 'primary' : 'default'"
            @click="timeRange = '24h'"
          >
            24小时
          </el-button>
          <el-button
            :type="timeRange === '7d' ? 'primary' : 'default'"
            @click="timeRange = '7d'"
          >
            7天
          </el-button>
        </el-button-group>
        <el-button size="small" @click="refreshData">
          <el-icon><Refresh /></el-icon>
        </el-button>
      </div>
    </div>

    <div class="metrics-grid">
      <!-- CPU使用率 -->
      <div class="metric-card" :class="getMetricStatusClass(metrics.cpu)">
        <div class="metric-header">
          <div class="metric-icon cpu-icon">
            <el-icon size="20"><Monitor /></el-icon>
          </div>
          <div class="metric-title">CPU使用率</div>
        </div>
        <div class="metric-value">{{ metrics.cpu }}%</div>
        <div class="metric-progress">
          <el-progress
            :percentage="metrics.cpu"
            :color="getProgressColor(metrics.cpu)"
            :stroke-width="8"
            :show-text="false"
          />
        </div>
        <div class="metric-trend" :class="getTrendClass(metrics.cpuTrend)">
          <el-icon size="12"><TrendCharts /></el-icon>
          {{ metrics.cpuTrend }}%
        </div>
      </div>

      <!-- 内存使用率 -->
      <div class="metric-card" :class="getMetricStatusClass(metrics.memory)">
        <div class="metric-header">
          <div class="metric-icon memory-icon">
            <el-icon size="20"><Grid /></el-icon>
          </div>
          <div class="metric-title">内存使用率</div>
        </div>
        <div class="metric-value">{{ metrics.memory }}%</div>
        <div class="metric-progress">
          <el-progress
            :percentage="metrics.memory"
            :color="getProgressColor(metrics.memory)"
            :stroke-width="8"
            :show-text="false"
          />
        </div>
        <div class="metric-trend" :class="getTrendClass(metrics.memoryTrend)">
          <el-icon size="12"><TrendCharts /></el-icon>
          {{ metrics.memoryTrend }}%
        </div>
      </div>

      <!-- 磁盘使用率 -->
      <div class="metric-card" :class="getMetricStatusClass(metrics.disk)">
        <div class="metric-header">
          <div class="metric-icon disk-icon">
            <el-icon size="20"><Grid /></el-icon>
          </div>
          <div class="metric-title">磁盘使用率</div>
        </div>
        <div class="metric-value">{{ metrics.disk }}%</div>
        <div class="metric-progress">
          <el-progress
            :percentage="metrics.disk"
            :color="getProgressColor(metrics.disk)"
            :stroke-width="8"
            :show-text="false"
          />
        </div>
        <div class="metric-trend" :class="getTrendClass(metrics.diskTrend)">
          <el-icon size="12"><TrendCharts /></el-icon>
          {{ metrics.diskTrend }}%
        </div>
      </div>

      <!-- 网络延迟 -->
      <div class="metric-card" :class="getMetricStatusClass(metrics.network)">
        <div class="metric-header">
          <div class="metric-icon network-icon">
            <el-icon size="20"><Lightning /></el-icon>
          </div>
          <div class="metric-title">网络延迟</div>
        </div>
        <div class="metric-value">{{ metrics.network }}ms</div>
        <div class="metric-progress">
          <el-progress
            :percentage="getNetworkPercentage(metrics.network)"
            :color="getProgressColor(getNetworkPercentage(metrics.network))"
            :stroke-width="8"
            :show-text="false"
          />
        </div>
        <div class="metric-trend" :class="getTrendClass(metrics.networkTrend)">
          <el-icon size="12"><TrendCharts /></el-icon>
          {{ metrics.networkTrend }}ms
        </div>
      </div>
    </div>

    <!-- 性能趋势图 -->
    <div class="performance-chart">
      <div class="chart-header">
        <h4>性能趋势</h4>
        <div class="chart-legend">
          <div class="legend-item">
            <div class="legend-color cpu-color"></div>
            <span>CPU</span>
          </div>
          <div class="legend-item">
            <div class="legend-color memory-color"></div>
            <span>内存</span>
          </div>
          <div class="legend-item">
            <div class="legend-color disk-color"></div>
            <span>磁盘</span>
          </div>
          <div class="legend-item">
            <div class="legend-color network-color"></div>
            <span>网络</span>
          </div>
        </div>
      </div>
      <div class="chart-container">
        <v-chart
          :option="chartOption"
          style="height: 200px;"
        />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { Monitor, Grid, Lightning, Refresh, TrendCharts } from '@element-plus/icons-vue'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart } from 'echarts/charts'
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
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent
])

interface Props {
  title?: string
}

const props = withDefaults(defineProps<Props>(), {
  title: '系统性能指标'
})

const emit = defineEmits<{
  refresh: []
}>()

// 响应式数据
const timeRange = ref('24h')
const metrics = ref({
  cpu: 45,
  memory: 62,
  disk: 38,
  network: 12,
  cpuTrend: 2.3,
  memoryTrend: -1.2,
  diskTrend: 0.8,
  networkTrend: -0.5
})

// 计算属性
const chartOption = computed(() => {
  return {
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(255, 255, 255, 0.9)',
      borderColor: '#ccc',
      textStyle: {
        color: '#333'
      }
    },
    legend: {
      data: ['CPU', '内存', '磁盘', '网络'],
      top: 10,
      textStyle: {
        color: '#666'
      }
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: generateTimeLabels(),
      axisLine: {
        lineStyle: {
          color: '#e0e0e0'
        }
      },
      axisLabel: {
        color: '#666'
      }
    },
    yAxis: {
      type: 'value',
      axisLine: {
        lineStyle: {
          color: '#e0e0e0'
        }
      },
      axisLabel: {
        color: '#666',
        formatter: '{value}%'
      },
      splitLine: {
        lineStyle: {
          color: '#f0f0f0'
        }
      }
    },
    series: [
      {
        name: 'CPU',
        type: 'line',
        data: generateRandomData(metrics.value.cpu),
        smooth: true,
        lineStyle: {
          color: '#1890ff',
          width: 2
        },
        itemStyle: {
          color: '#1890ff'
        }
      },
      {
        name: '内存',
        type: 'line',
        data: generateRandomData(metrics.value.memory),
        smooth: true,
        lineStyle: {
          color: '#52c41a',
          width: 2
        },
        itemStyle: {
          color: '#52c41a'
        }
      },
      {
        name: '磁盘',
        type: 'line',
        data: generateRandomData(metrics.value.disk),
        smooth: true,
        lineStyle: {
          color: '#faad14',
          width: 2
        },
        itemStyle: {
          color: '#faad14'
        }
      },
      {
        name: '网络',
        type: 'line',
        data: generateRandomData(getNetworkPercentage(metrics.value.network)),
        smooth: true,
        lineStyle: {
          color: '#722ed1',
          width: 2
        },
        itemStyle: {
          color: '#722ed1'
        }
      }
    ]
  }
})

// 方法
const getMetricStatusClass = (value: number): string => {
  if (value >= 80) return 'status-danger'
  if (value >= 60) return 'status-warning'
  return 'status-normal'
}

const getProgressColor = (value: number): string => {
  if (value >= 80) return '#ff4d4f'
  if (value >= 60) return '#faad14'
  return '#52c41a'
}

const getTrendClass = (trend: number): string => {
  if (trend > 0) return 'trend-positive'
  if (trend < 0) return 'trend-negative'
  return 'trend-neutral'
}

const getNetworkPercentage = (latency: number): number => {
  // 将网络延迟转换为百分比（假设100ms为100%）
  return Math.min((latency / 100) * 100, 100)
}

const generateTimeLabels = (): string[] => {
  const labels = []
  const now = new Date()
  const points = timeRange.value === '1h' ? 12 : timeRange.value === '24h' ? 24 : 7

  for (let i = points - 1; i >= 0; i--) {
    const time = new Date(now.getTime() - i * (timeRange.value === '1h' ? 5 * 60000 : timeRange.value === '24h' ? 60 * 60000 : 24 * 60 * 60000))
    labels.push(time.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' }))
  }

  return labels
}

const generateRandomData = (baseValue: number): number[] => {
  const data = []
  const points = timeRange.value === '1h' ? 12 : timeRange.value === '24h' ? 24 : 7

  for (let i = 0; i < points; i++) {
    const variation = (Math.random() - 0.5) * 20
    data.push(Math.max(0, Math.min(100, baseValue + variation)))
  }

  return data
}

const refreshData = () => {
  // 模拟数据刷新
  metrics.value = {
    cpu: Math.floor(Math.random() * 100),
    memory: Math.floor(Math.random() * 100),
    disk: Math.floor(Math.random() * 100),
    network: Math.floor(Math.random() * 50),
    cpuTrend: (Math.random() - 0.5) * 10,
    memoryTrend: (Math.random() - 0.5) * 10,
    diskTrend: (Math.random() - 0.5) * 10,
    networkTrend: (Math.random() - 0.5) * 5
  }
  emit('refresh')
}

// 生命周期
onMounted(() => {
  // 定期更新数据
  setInterval(() => {
    refreshData()
  }, 30000) // 每30秒更新一次
})
</script>

<style scoped>
.performance-dashboard {
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border-radius: 16px;
  padding: 24px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}

.dashboard-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.dashboard-title {
  margin: 0;
  display: flex;
  align-items: center;
  font-size: 18px;
  font-weight: 600;
  color: #333;
}

.dashboard-title .el-icon {
  margin-right: 8px;
}

.dashboard-controls {
  display: flex;
  align-items: center;
  gap: 12px;
}

.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
  margin-bottom: 24px;
}

.metric-card {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  padding: 16px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  transition: all 0.3s ease;
  position: relative;
}

.metric-card:hover {
  background: rgba(255, 255, 255, 0.15);
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
}

.metric-card.status-danger {
  border-left: 4px solid #ff4d4f;
}

.metric-card.status-warning {
  border-left: 4px solid #faad14;
}

.metric-card.status-normal {
  border-left: 4px solid #52c41a;
}

.metric-header {
  display: flex;
  align-items: center;
  margin-bottom: 12px;
}

.metric-icon {
  width: 32px;
  height: 32px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-right: 8px;
  color: white;
}

.cpu-icon {
  background: linear-gradient(135deg, #1890ff, #40a9ff);
}

.memory-icon {
  background: linear-gradient(135deg, #52c41a, #73d13d);
}

.disk-icon {
  background: linear-gradient(135deg, #faad14, #ffc53d);
}

.network-icon {
  background: linear-gradient(135deg, #722ed1, #9254de);
}

.metric-title {
  font-size: 14px;
  font-weight: 500;
  color: #333;
}

.metric-value {
  font-size: 24px;
  font-weight: 700;
  color: #333;
  margin-bottom: 8px;
}

.metric-progress {
  margin-bottom: 8px;
}

.metric-trend {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  font-weight: 500;
}

.trend-positive {
  color: #ff4d4f;
}

.trend-negative {
  color: #52c41a;
}

.trend-neutral {
  color: #666;
}

.performance-chart {
  background: rgba(255, 255, 255, 0.05);
  border-radius: 12px;
  padding: 16px;
}

.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.chart-header h4 {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
  color: #333;
}

.chart-legend {
  display: flex;
  gap: 16px;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: #666;
}

.legend-color {
  width: 12px;
  height: 12px;
  border-radius: 2px;
}

.cpu-color {
  background-color: #1890ff;
}

.memory-color {
  background-color: #52c41a;
}

.disk-color {
  background-color: #faad14;
}

.network-color {
  background-color: #722ed1;
}

.chart-container {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  padding: 8px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .dashboard-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 16px;
  }

  .metrics-grid {
    grid-template-columns: 1fr;
    gap: 12px;
  }

  .chart-legend {
    flex-wrap: wrap;
    gap: 8px;
  }

  .performance-dashboard {
    padding: 16px;
  }
}
</style>
