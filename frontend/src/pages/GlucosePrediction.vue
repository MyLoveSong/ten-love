<template>
  <div class="glucose-prediction">
    <div class="page-header">
      <h2 class="page-title">
        <el-icon size="24" color="#1890ff">
          <TrendCharts />
        </el-icon>
        血糖预测系统
      </h2>
    </div>

    <el-row :gutter="24">
      <!-- 预测参数输入 -->
      <el-col :span="8">
        <el-card title="预测参数" class="input-card" shadow="hover">
          <el-form :model="predictionForm" label-width="100px">
            <el-form-item label="当前血糖">
              <el-input-number
                v-model="predictionForm.currentGlucose"
                :min="50"
                :max="500"
                style="width: 100%;"
              />
            </el-form-item>

            <el-form-item label="预测模型">
              <el-select v-model="predictionForm.model" style="width: 100%;">
                <el-option label="基础模型" value="basic" />
                <el-option label="增强模型" value="enhanced" />
                <el-option label="GluFormer" value="gluformer" />
              </el-select>
            </el-form-item>

            <el-form-item label="预测时长">
              <el-select v-model="predictionForm.duration" style="width: 100%;">
                <el-option label="1小时" value="1h" />
                <el-option label="2小时" value="2h" />
                <el-option label="4小时" value="4h" />
                <el-option label="8小时" value="8h" />
              </el-select>
            </el-form-item>

            <el-form-item>
              <el-button
                type="primary"
                size="large"
                :loading="isPredicting"
                @click="handlePredict"
                style="width: 100%;"
              >
                {{ isPredicting ? '预测中...' : '开始预测' }}
              </el-button>
            </el-form-item>
          </el-form>
        </el-card>
      </el-col>

      <!-- 预测结果 -->
      <el-col :span="16">
        <el-card title="预测结果" class="result-card" shadow="hover">
          <div v-if="predictionResult" class="prediction-result">
            <el-row :gutter="16">
              <el-col :span="8">
                <el-statistic title="预测血糖值" :value="predictionResult.predictedGlucose">
                  <template #suffix>mg/dL</template>
                </el-statistic>
              </el-col>
              <el-col :span="8">
                <el-statistic title="置信度" :value="predictionResult.confidence">
                  <template #suffix>%</template>
                </el-statistic>
              </el-col>
              <el-col :span="8">
                <el-statistic title="风险等级" :value="predictionResult.riskLevel">
                  <template #suffix>
                    <el-tag :type="getRiskTagType(predictionResult.riskLevel)">
                      {{ predictionResult.riskLevel }}
                    </el-tag>
                  </template>
                </el-statistic>
              </el-col>
            </el-row>

            <el-divider />

            <div class="chart-container">
              <v-chart :option="predictionChartOption" style="height: 300px;" />
            </div>
          </div>

          <el-empty v-else description="请先进行血糖预测" />
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed } from 'vue'
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
import { ElMessage } from 'element-plus'

use([
  CanvasRenderer,
  LineChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent
])

// 表单数据
const predictionForm = reactive({
  currentGlucose: 120,
  model: 'enhanced',
  duration: '2h'
})

// 响应式数据
const isPredicting = ref(false)
const predictionResult = ref<any>(null)

// 预测图表配置
const predictionChartOption = computed(() => ({
  title: {
    text: '血糖预测趋势',
    left: 'center',
    textStyle: {
      color: '#333',
      fontSize: 16
    }
  },
  tooltip: {
    trigger: 'axis',
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    borderColor: '#1890ff',
    textStyle: {
      color: '#fff'
    }
  },
  legend: {
    data: ['实际血糖', '预测血糖'],
    bottom: 0,
    textStyle: {
      color: '#666'
    }
  },
  xAxis: {
    type: 'category',
    data: predictionResult.value?.timeline || [],
    axisLine: {
      lineStyle: {
        color: '#e8e8e8'
      }
    },
    axisLabel: {
      color: '#666'
    }
  },
  yAxis: {
    type: 'value',
    name: '血糖值 (mg/dL)',
    axisLine: {
      lineStyle: {
        color: '#e8e8e8'
      }
    },
    axisLabel: {
      color: '#666'
    },
    splitLine: {
      lineStyle: {
        color: '#f0f0f0'
      }
    }
  },
  series: [
    {
      name: '实际血糖',
      type: 'line',
      data: predictionResult.value?.actualValues || [],
      smooth: true,
      lineStyle: {
        color: '#1890ff',
        width: 3
      },
      itemStyle: {
        color: '#1890ff'
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
      }
    },
    {
      name: '预测血糖',
      type: 'line',
      data: predictionResult.value?.predictedValues || [],
      smooth: true,
      lineStyle: {
        color: '#52c41a',
        type: 'dashed',
        width: 3
      },
      itemStyle: {
        color: '#52c41a'
      }
    }
  ]
}))

// 获取风险标签类型
const getRiskTagType = (riskLevel: string) => {
  const typeMap: Record<string, string> = {
    '低风险': 'success',
    '中等风险': 'warning',
    '高风险': 'danger'
  }
  return typeMap[riskLevel] || 'info'
}

// 执行预测
const handlePredict = async () => {
  isPredicting.value = true

  try {
    // 模拟API调用
    await new Promise(resolve => setTimeout(resolve, 2000))

    // 模拟预测结果
    predictionResult.value = {
      predictedGlucose: Math.round(predictionForm.currentGlucose + (Math.random() - 0.5) * 20),
      confidence: Math.round(85 + Math.random() * 10),
      riskLevel: predictionForm.currentGlucose > 140 ? '高风险' :
                 predictionForm.currentGlucose > 100 ? '中等风险' : '低风险',
      timeline: ['当前', '30分钟', '1小时', '1.5小时', '2小时'],
      actualValues: [predictionForm.currentGlucose, null, null, null, null],
      predictedValues: [
        predictionForm.currentGlucose,
        predictionForm.currentGlucose + 5,
        predictionForm.currentGlucose + 10,
        predictionForm.currentGlucose + 8,
        predictionForm.currentGlucose + 12
      ]
    }

    ElMessage.success('血糖预测完成')
  } catch (error) {
    ElMessage.error('预测失败，请重试')
    console.error('Prediction error:', error)
  } finally {
    isPredicting.value = false
  }
}
</script>

<style scoped>
.glucose-prediction {
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

.input-card,
.result-card {
  height: 100%;
}

.prediction-result {
  padding: 0;
}

.chart-container {
  margin-top: 24px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .glucose-prediction .el-row {
    flex-direction: column;
  }

  .glucose-prediction .el-col {
    width: 100% !important;
    margin-bottom: 24px;
  }
}
</style>