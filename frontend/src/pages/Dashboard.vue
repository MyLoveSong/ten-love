<template>
  <div class="dashboard enhanced-dashboard">
    <!-- 页面标题和实时时间 -->
    <div class="page-header glass-card">
      <div class="header-left">
        <h2 class="page-title gradient-text">
          <el-icon size="24" color="#ff4d4f">
            <Monitor />
          </el-icon>
          增强学术级智能健康监测集成系统
        </h2>
      </div>
      <div class="header-right">
        <div class="header-controls">
          <RealtimeIndicator label="数据连接" />
          <DataRefreshManager
            :refresh-interval="30000"
            @refresh="onGlobalRefresh"
            @interval-change="onIntervalChange"
          />
        </div>
        <div class="header-info">
          <el-text type="info">当前时间: {{ currentTime }}</el-text>
          <br />
          <el-text type="info">
            系统状态:
            <el-tag :type="getHealthStatusType(systemStats?.system_health)" size="small" class="status-tag">
              {{ getHealthStatusText(systemStats?.system_health) }}
            </el-tag>
          </el-text>
        </div>
      </div>
    </div>

    <!-- 系统健康状态 - 使用组件化设计 -->
    <el-card class="health-status-card glass-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <h3 class="card-title">
            <el-icon size="20" color="#1890ff"><Monitor /></el-icon>
            系统健康状态
          </h3>
          <el-tag
            :type="getHealthStatusType(systemStats?.system_health)"
            size="small"
            class="status-tag"
          >
            {{ getHealthStatusText(systemStats?.system_health) }}
          </el-tag>
        </div>
      </template>

      <el-row :gutter="20">
        <!-- 系统健康度 - 使用健康仪表盘组件 -->
        <el-col :span="6">
          <HealthGauge
            :health-value="systemStats?.system_health || 0"
            label="系统健康度"
            :gauge-size="120"
            :stroke-width="12"
          />
        </el-col>

        <!-- 其他统计卡片 - 使用可复用组件 -->
        <el-col :span="6">
          <StatCard
            :value="systemStats?.online_users || 0"
            label="在线用户"
            icon="User"
            icon-class="users"
            trend="+5.2%"
            trend-class="positive"
            format-type="number"
          />
        </el-col>

        <el-col :span="6">
          <StatCard
            :value="systemStats?.today_predictions || 0"
            label="今日预测"
            icon="Lightning"
            icon-class="predictions"
            trend="+12.8%"
            trend-class="positive"
            format-type="number"
          />
        </el-col>

        <el-col :span="6">
          <StatCard
            :value="systemStats?.activity_score || 0"
            label="系统活跃度"
            icon="Fire"
            icon-class="activity"
            trend="+3.5%"
            trend-class="positive"
            format-type="percentage"
          />
        </el-col>
      </el-row>
    </el-card>

    <!-- 功能模块统计 - 增强版 -->
    <el-card class="module-stats-card glass-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <h3 class="card-title">
            <el-icon size="20" color="#1890ff"><Grid /></el-icon>
            功能模块统计
          </h3>
          <el-button type="primary" size="small" class="btn-enhanced btn-primary" @click="refreshModuleStats">
            <el-icon><Refresh /></el-icon>
            刷新数据
          </el-button>
        </div>
      </template>

      <div class="responsive-grid">
        <div
          v-for="(stat, index) in enhancedStats"
          :key="index"
          class="module-card data-card micro-interaction"
          @click="navigateToModule(stat.route)"
        >
          <div class="module-header">
            <div class="module-icon" :class="stat.iconClass">
              <el-icon size="24">{{ stat.icon }}</el-icon>
            </div>
            <div class="module-trend" :class="stat.trendClass">
              <el-icon size="12"><TrendCharts /></el-icon>
              {{ stat.trend }}
            </div>
          </div>
          <div class="module-content">
            <div class="module-title">{{ stat.title }}</div>
            <div class="module-value stat-number">{{ formatNumber(stat.value) }}</div>
            <div class="module-description">{{ stat.description }}</div>
          </div>
          <div class="module-footer">
            <div class="module-status" :class="stat.statusClass">
              <div class="status-dot"></div>
              {{ stat.status }}
            </div>
            <el-icon class="arrow-icon"><ArrowRight /></el-icon>
          </div>
        </div>
      </div>
    </el-card>

    <!-- 数据可视化 - 使用图表组件 -->
    <el-row :gutter="20">
      <!-- 血糖趋势图 -->
      <el-col :span="12">
        <ChartCard
          title="血糖预测趋势"
          header-icon="TrendCharts"
          chart-type="line"
          :chart-data="glucoseTrend"
          :chart-option="glucoseTrendOption"
          :loading="trendLoading"
          :height="350"
          :show-stats="true"
          @chart-click="onChartClick"
        >
          <template #controls>
            <el-button-group size="small">
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
              <el-button
                :type="timeRange === '30d' ? 'primary' : 'default'"
                @click="timeRange = '30d'"
              >
                30天
              </el-button>
            </el-button-group>
          </template>

          <template #stats>
            <div class="chart-stats-grid">
              <div class="stat-item">
                <div class="stat-label">平均血糖</div>
                <div class="stat-value stat-number">{{ averageGlucose }} mg/dL</div>
              </div>
              <div class="stat-item">
                <div class="stat-label">最高血糖</div>
                <div class="stat-value stat-number">{{ maxGlucose }} mg/dL</div>
              </div>
              <div class="stat-item">
                <div class="stat-label">最低血糖</div>
                <div class="stat-value stat-number">{{ minGlucose }} mg/dL</div>
              </div>
              <div class="stat-item">
                <div class="stat-label">预测准确率</div>
                <div class="stat-value stat-number">{{ predictionAccuracy }}%</div>
              </div>
          </div>
          </template>
        </ChartCard>
      </el-col>

      <!-- 健康分布饼图 -->
      <el-col :span="12">
        <ChartCard
          title="健康状态分布"
          header-icon="PieChartIcon"
          chart-type="pie"
          :chart-data="healthData"
          :chart-option="healthDistributionOption"
          :loading="distributionLoading"
          :height="350"
          :show-stats="true"
          @chart-click="onPieChartClick"
        >
          <template #controls>
            <el-button size="small" class="btn-enhanced btn-primary" @click="refreshHealthDistribution">
              <el-icon><Refresh /></el-icon>
              刷新
            </el-button>
          </template>

          <template #stats>
            <div class="health-details">
              <div
                v-for="item in healthData"
                :key="item.name"
                class="health-item micro-interaction"
                @click="selectHealthCategory(item.name)"
              >
                <div class="health-color" :style="{ backgroundColor: item.itemStyle.color }"></div>
                <div class="health-info">
                  <div class="health-name">{{ item.name }}</div>
                  <div class="health-count">{{ item.value }} 人</div>
                </div>
                <div class="health-percentage stat-number">{{ ((item.value / totalHealthCount) * 100).toFixed(1) }}%</div>
              </div>
          </div>
          </template>
        </ChartCard>
      </el-col>
    </el-row>

    <!-- 活动状态显示 - 增强版 -->
    <el-card class="activity-status-card glass-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <h3 class="card-title">
            <el-icon size="20" color="#1890ff"><Monitor /></el-icon>
            活动状态监控
          </h3>
          <el-button type="primary" size="small" class="btn-enhanced btn-primary" @click="refreshActivityStatus">
            <el-icon><Refresh /></el-icon>
            刷新状态
          </el-button>
        </div>
      </template>

      <div class="activity-grid">
        <ActivityIndicator
          title="血糖监测"
          :value="activityData.glucoseMonitoring"
          description="实时血糖监测状态"
          status="active"
          icon="Monitor"
          trend="+2.1%"
          format-type="percentage"
        />

        <ActivityIndicator
          title="菜谱推荐"
          :value="activityData.recipeRecommendation"
          description="智能菜谱推荐活跃度"
          status="active"
          icon="Monitor"
          trend="+5.3%"
          format-type="number"
        />

        <ActivityIndicator
          title="图像识别"
          :value="activityData.imageRecognition"
          description="食物图像识别处理"
          status="warning"
          icon="Monitor"
          trend="-1.2%"
          format-type="number"
        />

        <ActivityIndicator
          title="文化适配"
          :value="activityData.culturalAdaptation"
          description="文化适配功能使用"
          status="active"
          icon="Monitor"
          trend="+8.7%"
          format-type="percentage"
        />

        <ActivityIndicator
          title="用户交互"
          :value="activityData.userInteraction"
          description="用户交互活跃度"
          status="active"
          icon="Monitor"
          trend="+3.4%"
          format-type="number"
        />

        <ActivityIndicator
          title="系统维护"
          :value="activityData.systemMaintenance"
          description="系统维护状态"
          status="inactive"
          icon="Monitor"
          trend="0%"
          format-type="text"
        />
      </div>
    </el-card>

    <!-- 性能指标仪表盘 -->
    <el-card class="performance-card glass-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <h3 class="card-title">
            <el-icon size="20" color="#1890ff"><Monitor /></el-icon>
            系统性能监控
          </h3>
          <el-tag type="success" size="small" class="status-tag">
            运行正常
          </el-tag>
        </div>
      </template>

      <PerformanceDashboard
        title="系统性能指标"
        @refresh="onPerformanceRefresh"
      />
    </el-card>

    <!-- 系统通知 - 增强版 -->
    <el-card title="系统通知" class="notification-card glass-card" shadow="hover">
      <el-space direction="vertical" style="width: 100%;">
        <el-alert
          title="系统更新"
          description="血糖预测模型已更新至v3.0.0，新增GluFormer算法支持"
          type="info"
          show-icon
          closable
          class="micro-interaction"
        />
        <el-alert
          title="功能优化"
          description="菜谱推荐系统已集成文化适配功能，提供更个性化的推荐"
          type="success"
          show-icon
          closable
          class="micro-interaction"
        />
        <el-alert
          title="维护通知"
          description="系统将于今晚23:00-01:00进行例行维护，期间服务可能短暂中断"
          type="warning"
          show-icon
          closable
          class="micro-interaction"
        />
      </el-space>
    </el-card>

    <!-- 无障碍访问面板 -->
    <AccessibilityPanel />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { LineChart, PieChart as EChartsPieChart, GaugeChart } from 'echarts/charts'
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent
} from 'echarts/components'
import { Monitor, TrendCharts, Grid, Refresh, ArrowRight } from '@element-plus/icons-vue'

// 导入自定义组件
import StatCard from '@/components/dashboard/StatCard.vue'
import ChartCard from '@/components/dashboard/ChartCard.vue'
import HealthGauge from '@/components/dashboard/HealthGauge.vue'
import ActivityIndicator from '@/components/dashboard/ActivityIndicator.vue'
import PerformanceDashboard from '@/components/dashboard/PerformanceDashboard.vue'
import RealtimeIndicator from '@/components/dashboard/RealtimeIndicator.vue'
import DataRefreshManager from '@/components/dashboard/DataRefreshManager.vue'
import AccessibilityPanel from '@/components/dashboard/AccessibilityPanel.vue'

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

const router = useRouter()

// 响应式数据
const currentTime = ref('')
const systemStats = ref<any>(null)
const glucoseTrend = ref<any[]>([])
const healthData = ref<any[]>([])
const trendLoading = ref(false)
const distributionLoading = ref(false)
const timeRange = ref('24h')

// 活动状态数据
const activityData = ref({
  glucoseMonitoring: 95,
  recipeRecommendation: 1247,
  imageRecognition: 89,
  culturalAdaptation: 78,
  userInteraction: 2156,
  systemMaintenance: '正常'
})

// 增强统计数据
const enhancedStats = ref([
  {
    title: '血糖预测',
    value: 1234,
    icon: 'TrendCharts',
    iconClass: 'glucose',
    trend: '+12%',
    trendClass: 'positive',
    description: 'AI驱动的血糖预测分析',
    status: '运行中',
    statusClass: 'active',
    route: '/glucose'
  },
  {
    title: '健康评估',
    value: 567,
    icon: 'Heart',
    iconClass: 'health',
    trend: '+8%',
    trendClass: 'positive',
    description: '综合健康状态评估',
    status: '运行中',
    statusClass: 'active',
    route: '/health'
  },
  {
    title: '菜谱推荐',
    value: 890,
    icon: 'Food',
    iconClass: 'recipe',
    trend: '+15%',
    trendClass: 'positive',
    description: '个性化营养菜谱推荐',
    status: '运行中',
    statusClass: 'active',
    route: '/recipes'
  },
  {
    title: '图像分析',
    value: 234,
    icon: 'Camera',
    iconClass: 'image',
    trend: '+5%',
    trendClass: 'positive',
    description: '智能图像识别分析',
    status: '运行中',
    statusClass: 'active',
    route: '/image'
  },
  {
    title: '文化适配',
    value: 456,
    icon: 'Globe',
    iconClass: 'cultural',
    trend: '+20%',
    trendClass: 'positive',
    description: '多文化背景适配',
    status: '运行中',
    statusClass: 'active',
    route: '/cultural'
  },
  {
    title: 'MERL分析',
    value: 123,
    icon: 'Experiment',
    iconClass: 'merl',
    trend: '+3%',
    trendClass: 'positive',
    description: '监控评估研究学习',
    status: '运行中',
    statusClass: 'active',
    route: '/merl'
  }
])

// 计算属性
const averageGlucose = computed(() => {
  if (!glucoseTrend.value.length) return 0
  const sum = glucoseTrend.value.reduce((acc, item) => acc + item.glucose, 0)
  return Math.round(sum / glucoseTrend.value.length)
})

const maxGlucose = computed(() => {
  if (!glucoseTrend.value.length) return 0
  return Math.max(...glucoseTrend.value.map(item => item.glucose))
})

const minGlucose = computed(() => {
  if (!glucoseTrend.value.length) return 0
  return Math.min(...glucoseTrend.value.map(item => item.glucose))
})

const predictionAccuracy = computed(() => {
  if (!glucoseTrend.value.length) return 0
  let accurateCount = 0
  glucoseTrend.value.forEach(item => {
    const diff = Math.abs(item.glucose - item.prediction)
    if (diff <= 10) accurateCount++ // 10mg/dL以内认为准确
  })
  return Math.round((accurateCount / glucoseTrend.value.length) * 100)
})

const totalHealthCount = computed(() => {
  return healthData.value.reduce((sum, item) => sum + item.value, 0)
})

// 血糖趋势图表配置
const glucoseTrendOption = computed(() => ({
  tooltip: {
    trigger: 'axis',
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    borderColor: '#1890ff',
    borderWidth: 1,
    textStyle: {
      color: '#fff',
      fontSize: 12
    },
    formatter: function(params: any) {
      let result = `<div style="padding: 8px;">`
      result += `<div style="font-weight: bold; margin-bottom: 8px;">${params[0].axisValue}</div>`
      params.forEach((param: any) => {
        result += `<div style="display: flex; align-items: center; margin-bottom: 4px;">`
        result += `<span style="display: inline-block; width: 10px; height: 10px; background-color: ${param.color}; border-radius: 50%; margin-right: 8px;"></span>`
        result += `<span style="margin-right: 8px;">${param.seriesName}:</span>`
        result += `<span style="font-weight: bold;">${param.value} mg/dL</span>`
        result += `</div>`
      })
      result += `</div>`
      return result
    }
  },
  legend: {
    data: ['血糖值', '预测值'],
    bottom: 10,
    textStyle: {
      color: '#666',
      fontSize: 12
    }
  },
  grid: {
    left: '3%',
    right: '4%',
    bottom: '15%',
    top: '10%',
    containLabel: true
  },
  xAxis: {
    type: 'category',
    data: glucoseTrend.value.map(item => item.time),
    axisLine: {
      lineStyle: {
        color: '#e8e8e8'
      }
    },
    axisLabel: {
      color: '#666',
      fontSize: 11
    }
  },
  yAxis: {
    type: 'value',
    name: '血糖值 (mg/dL)',
    nameTextStyle: {
      color: '#666',
      fontSize: 12
    },
    axisLine: {
      lineStyle: {
        color: '#e8e8e8'
      }
    },
    axisLabel: {
      color: '#666',
      fontSize: 11
    },
    splitLine: {
      lineStyle: {
        color: '#f0f0f0',
        type: 'dashed'
      }
    }
  },
  series: [
    {
      name: '血糖值',
      type: 'line',
      data: glucoseTrend.value.map(item => item.glucose),
      smooth: true,
      lineStyle: {
        color: '#1890ff',
        width: 3
      },
      itemStyle: {
        color: '#1890ff',
        borderWidth: 2,
        borderColor: '#fff'
      },
      areaStyle: {
        color: {
          type: 'linear',
          x: 0,
          y: 0,
          x2: 0,
          y2: 1,
          colorStops: [
            { offset: 0, color: 'rgba(24, 144, 255, 0.3)' },
            { offset: 1, color: 'rgba(24, 144, 255, 0.1)' }
          ]
        }
      },
      emphasis: {
        itemStyle: {
          shadowBlur: 10,
          shadowOffsetX: 0,
          shadowColor: 'rgba(24, 144, 255, 0.5)'
        }
      }
    },
    {
      name: '预测值',
      type: 'line',
      data: glucoseTrend.value.map(item => item.prediction),
      smooth: true,
      lineStyle: {
        color: '#52c41a',
        type: 'dashed',
        width: 3
      },
      itemStyle: {
        color: '#52c41a',
        borderWidth: 2,
        borderColor: '#fff'
      },
      emphasis: {
        itemStyle: {
          shadowBlur: 10,
          shadowOffsetX: 0,
          shadowColor: 'rgba(82, 196, 26, 0.5)'
        }
      }
    }
  ]
}))

// 健康分布饼图配置
const healthDistributionOption = computed(() => ({
  tooltip: {
    trigger: 'item',
    formatter: function(params: any) {
      const percentage = ((params.value / totalHealthCount.value) * 100).toFixed(1)
      return `
        <div style="padding: 8px;">
          <div style="font-weight: bold; margin-bottom: 8px;">${params.name}</div>
          <div style="display: flex; align-items: center; margin-bottom: 4px;">
            <span style="display: inline-block; width: 10px; height: 10px; background-color: ${params.color}; border-radius: 50%; margin-right: 8px;"></span>
            <span>人数: ${params.value}</span>
          </div>
          <div style="font-weight: bold;">占比: ${percentage}%</div>
        </div>
      `
    },
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    borderColor: '#1890ff',
    borderWidth: 1,
    textStyle: {
      color: '#fff',
      fontSize: 12
    }
  },
  legend: {
    orient: 'vertical',
    left: 'left',
    top: 'center',
    textStyle: {
      color: '#666',
      fontSize: 12
    },
    itemGap: 20
  },
  series: [
    {
      name: '健康状态',
      type: 'pie',
      radius: ['45%', '75%'],
      center: ['60%', '50%'],
      data: healthData.value,
      emphasis: {
        itemStyle: {
          shadowBlur: 10,
          shadowOffsetX: 0,
          shadowColor: 'rgba(0, 0, 0, 0.5)'
        },
        scale: true,
        scaleSize: 10
      },
      label: {
        show: true,
        formatter: '{b}\n{d}%',
        fontSize: 11,
        color: '#666'
      },
      labelLine: {
        show: true,
        lineStyle: {
          color: '#e8e8e8'
        }
      }
    }
  ]
}))

// 方法
const navigateToModule = (route: string) => {
  router.push(route)
}

// 刷新模块统计
const refreshModuleStats = async () => {
  await fetchSystemStats()
  ElMessage.success('模块统计数据已刷新')
}


// 格式化数字显示
const formatNumber = (num: number): string => {
  if (num >= 1000000) {
    return (num / 1000000).toFixed(1) + 'M'
  } else if (num >= 1000) {
    return (num / 1000).toFixed(1) + 'K'
  }
  return num.toString()
}

// 获取健康状态类型
const getHealthStatusType = (health: number): string => {
  if (health >= 90) return 'success'
  if (health >= 70) return 'warning'
  return 'danger'
}

// 获取健康状态文本
const getHealthStatusText = (health: number): string => {
  if (health >= 90) return '健康'
  if (health >= 70) return '警告'
  return '危险'
}

// 刷新健康分布数据
const refreshHealthDistribution = async () => {
  await fetchHealthDistribution()
  ElMessage.success('健康分布数据已刷新')
}

// 图表点击事件
const onChartClick = (params: any) => {
  console.log('图表点击:', params)
  ElMessage.info(`点击了 ${params.name}: ${params.value} mg/dL`)
}

// 饼图点击事件
const onPieChartClick = (params: any) => {
  console.log('饼图点击:', params)
  ElMessage.info(`选择了健康状态: ${params.name}`)
}

// 选择健康分类
const selectHealthCategory = (category: string) => {
  ElMessage.info(`已选择: ${category}`)
}

// 刷新活动状态
const refreshActivityStatus = () => {
  ElMessage.success('活动状态已刷新')
  // 模拟数据更新
  activityData.value = {
    glucoseMonitoring: Math.floor(Math.random() * 20) + 80,
    recipeRecommendation: Math.floor(Math.random() * 500) + 1000,
    imageRecognition: Math.floor(Math.random() * 20) + 80,
    culturalAdaptation: Math.floor(Math.random() * 20) + 70,
    userInteraction: Math.floor(Math.random() * 500) + 1800,
    systemMaintenance: '正常'
  }
}

// 性能指标刷新
const onPerformanceRefresh = () => {
  ElMessage.success('性能指标已刷新')
}

// 全局数据刷新
const onGlobalRefresh = async () => {
  try {
    await Promise.all([
      fetchSystemStats(),
      fetchGlucoseTrend(),
      fetchHealthDistribution()
    ])
    ElMessage.success('所有数据已刷新')
  } catch (error) {
    ElMessage.error('数据刷新失败')
    console.error('刷新错误:', error)
  }
}

// 刷新间隔变化
const onIntervalChange = (interval: number) => {
  ElMessage.info(`刷新间隔已更改为 ${interval / 1000} 秒`)
}

// 更新时间
const updateTime = () => {
  currentTime.value = new Date().toLocaleString()
}

// 获取系统统计
const fetchSystemStats = async () => {
  try {
    const response = await fetch('/api/v1/stats/comprehensive')
    const data = await response.json()
    if (data.success) {
      systemStats.value = data.data
    }
  } catch (error) {
    console.error('获取系统统计失败:', error)
  }
}

// 获取血糖趋势
const fetchGlucoseTrend = async () => {
  trendLoading.value = true
  try {
    const response = await fetch('/api/v1/stats/glucose/trend')
    const data = await response.json()
    if (data.success) {
      glucoseTrend.value = data.data
    }
  } catch (error) {
    console.error('获取血糖趋势失败:', error)
  } finally {
    trendLoading.value = false
  }
}

// 获取健康分布
const fetchHealthDistribution = async () => {
  distributionLoading.value = true
  try {
    const response = await fetch('/api/v1/stats/health-distribution')
    const data = await response.json()
    if (data.success) {
      healthData.value = data.data.map((item: any) => ({
        value: item.count,
        name: item.category,
        itemStyle: { color: item.color }
      }))
    }
  } catch (error) {
    console.error('获取健康分布失败:', error)
  } finally {
    distributionLoading.value = false
  }
}

// 定时器
let timeInterval: NodeJS.Timeout

onMounted(() => {
  updateTime()
  timeInterval = setInterval(updateTime, 1000)

  fetchSystemStats()
  fetchGlucoseTrend()
  fetchHealthDistribution()
})

onUnmounted(() => {
  if (timeInterval) {
    clearInterval(timeInterval)
  }
})
</script>

<style scoped>
/* 仪表板增强样式 */
.enhanced-dashboard {
  padding: 0;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  min-height: 100vh;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
  padding: 20px;
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.page-title {
  margin: 0;
  display: flex;
  align-items: center;
  font-size: 24px;
  font-weight: bold;
  background: linear-gradient(135deg, #1890ff, #40a9ff);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.header-right {
  text-align: right;
}

.header-controls {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}

.header-info {
  text-align: right;
}

/* 增强卡片样式 */
.glass-card {
  margin-bottom: 24px;
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
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

.status-tag {
  font-weight: 500;
}

/* 图表统计网格 */
.chart-stats-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
}

.stat-item {
  text-align: center;
  padding: 12px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  transition: all 0.3s ease;
}

.stat-item:hover {
  background: rgba(255, 255, 255, 0.2);
  transform: translateY(-2px);
}

.stat-item .stat-label {
  font-size: 12px;
  color: #666;
  margin-bottom: 4px;
}

.stat-item .stat-value {
  font-size: 16px;
  font-weight: 600;
  color: #333;
}

/* 健康详情样式 */
.health-details {
  padding: 16px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 8px;
}

.health-item {
  display: flex;
  align-items: center;
  padding: 12px 0;
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
  cursor: pointer;
  transition: all 0.3s ease;
}

.health-item:last-child {
  border-bottom: none;
}

.health-item:hover {
  background: rgba(24, 144, 255, 0.05);
  border-radius: 6px;
  padding-left: 8px;
  padding-right: 8px;
}

.health-color {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  margin-right: 12px;
}

.health-info {
  flex: 1;
}

.health-name {
  font-size: 14px;
  font-weight: 500;
  color: #333;
  margin-bottom: 2px;
}

.health-count {
  font-size: 12px;
  color: #666;
}

.health-percentage {
  font-size: 14px;
  font-weight: 600;
  color: #1890ff;
}

/* 模块卡片样式 */
.module-card {
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border-radius: 16px;
  padding: 20px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  transition: all 0.3s ease;
  cursor: pointer;
  position: relative;
  overflow: hidden;
}

.module-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: linear-gradient(90deg, #1890ff, #40a9ff);
  opacity: 0;
  transition: opacity 0.3s ease;
}

.module-card:hover::before {
  opacity: 1;
}

.module-card:hover {
  transform: translateY(-6px);
  box-shadow: 0 16px 48px rgba(0, 0, 0, 0.15);
  background: rgba(255, 255, 255, 0.2);
}

.module-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.module-icon {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  transition: all 0.3s ease;
}

.module-icon.glucose {
  background: linear-gradient(135deg, #1890ff, #40a9ff);
}

.module-icon.health {
  background: linear-gradient(135deg, #52c41a, #73d13d);
}

.module-icon.recipe {
  background: linear-gradient(135deg, #fa8c16, #ffc53d);
}

.module-icon.image {
  background: linear-gradient(135deg, #722ed1, #9254de);
}

.module-icon.cultural {
  background: linear-gradient(135deg, #13c2c2, #36cfc9);
}

.module-icon.merl {
  background: linear-gradient(135deg, #eb2f96, #f759ab);
}

.module-trend {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  font-weight: 500;
  padding: 4px 8px;
  border-radius: 12px;
}

.module-trend.positive {
  background: rgba(82, 196, 26, 0.1);
  color: #52c41a;
}

.module-trend.negative {
  background: rgba(255, 77, 79, 0.1);
  color: #ff4d4f;
}

.module-content {
  margin-bottom: 16px;
}

.module-title {
  font-size: 16px;
  font-weight: 600;
  color: #333;
  margin-bottom: 8px;
}

.module-value {
  font-size: 24px;
  font-weight: 700;
  color: #333;
  margin-bottom: 4px;
}

.module-description {
  font-size: 12px;
  color: #666;
  line-height: 1.4;
}

.module-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.module-status {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  font-weight: 500;
}

.status-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
}

.module-status.active .status-dot {
  background-color: #52c41a;
}

.module-status.inactive .status-dot {
  background-color: #d9d9d9;
}

.module-status.warning .status-dot {
  background-color: #faad14;
}

.arrow-icon {
  color: #1890ff;
  transition: transform 0.3s ease;
}

.module-card:hover .arrow-icon {
  transform: translateX(4px);
}

.notification-card {
  margin-top: 24px;
}

/* 活动状态网格 */
.activity-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 16px;
}

.activity-status-card {
  margin-bottom: 24px;
}

/* 性能指标卡片 */
.performance-card {
  margin-bottom: 24px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .page-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 16px;
  }

  .page-title {
    font-size: 20px;
  }

  .header-right {
    text-align: left;
  }

  .chart-stats-grid {
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
  }

  .card-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 12px;
  }

  .module-card {
    padding: 16px;
  }

  .activity-grid {
    grid-template-columns: 1fr;
    gap: 12px;
  }
}

@media (max-width: 480px) {
  .enhanced-dashboard {
    padding: 8px;
  }

  .page-header {
    padding: 16px;
  }

  .chart-stats-grid {
    grid-template-columns: 1fr;
    gap: 8px;
  }

  .stat-item .stat-value {
    font-size: 14px;
  }

  .health-item {
    padding: 8px 0;
  }
}
</style>