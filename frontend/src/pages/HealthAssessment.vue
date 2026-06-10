<template>
  <div class="health-assessment">
    <div class="page-header">
      <h2 class="page-title">
        <el-icon size="24" color="#ff4d4f">
          <Heart />
        </el-icon>
        健康评估与膳食推荐
      </h2>
    </div>

    <el-row :gutter="24">
      <!-- 健康信息输入 -->
      <el-col :span="12">
        <el-card title="健康信息输入" class="input-card" shadow="hover">
          <el-form
            ref="formRef"
            :model="formData"
            :rules="rules"
            label-width="120px"
            @submit.prevent="handleSubmit"
          >
            <el-form-item label="用户ID" prop="user_id">
              <el-input v-model="formData.user_id" disabled />
            </el-form-item>

            <el-form-item label="年龄" prop="age">
              <el-input-number
                v-model="formData.age"
                :min="1"
                :max="120"
                style="width: 100%;"
              />
            </el-form-item>

            <el-form-item label="性别" prop="gender">
              <el-select v-model="formData.gender" style="width: 100%;">
                <el-option label="男" value="male" />
                <el-option label="女" value="female" />
                <el-option label="其他" value="other" />
              </el-select>
            </el-form-item>

            <el-row :gutter="16">
              <el-col :span="12">
                <el-form-item label="体重 (kg)" prop="weight_kg">
                  <el-input-number
                    v-model="formData.weight_kg"
                    :min="1"
                    :max="300"
                    style="width: 100%;"
                  />
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item label="身高 (cm)" prop="height_cm">
                  <el-input-number
                    v-model="formData.height_cm"
                    :min="50"
                    :max="250"
                    style="width: 100%;"
                  />
                </el-form-item>
              </el-col>
            </el-row>

            <el-form-item label="活动水平" prop="activity_level">
              <el-select v-model="formData.activity_level" style="width: 100%;">
                <el-option label="久坐不动" value="sedentary" />
                <el-option label="轻度活动" value="light" />
                <el-option label="中度活动" value="moderate" />
                <el-option label="活跃" value="active" />
                <el-option label="非常活跃" value="very_active" />
              </el-select>
            </el-form-item>

            <el-form-item label="健康状况" prop="health_conditions">
              <el-select
                v-model="formData.health_conditions"
                multiple
                placeholder="选择健康状况"
                style="width: 100%;"
              >
                <el-option label="糖尿病" value="diabetes" />
                <el-option label="高血压" value="hypertension" />
                <el-option label="心脏病" value="heart_disease" />
                <el-option label="肥胖" value="obesity" />
                <el-option label="过敏" value="allergies" />
              </el-select>
            </el-form-item>

            <el-form-item label="饮食偏好" prop="dietary_preferences">
              <el-select
                v-model="formData.dietary_preferences"
                multiple
                placeholder="选择饮食偏好"
                style="width: 100%;"
              >
                <el-option label="素食" value="vegetarian" />
                <el-option label="纯素" value="vegan" />
                <el-option label="低碳水" value="low_carb" />
                <el-option label="生酮" value="keto" />
                <el-option label="清真" value="halal" />
              </el-select>
            </el-form-item>

            <el-form-item label="文化背景" prop="cultural_background">
              <el-select
                v-model="formData.cultural_background"
                placeholder="选择文化背景"
                style="width: 100%;"
              >
                <el-option label="中华文化" value="chinese" />
                <el-option label="西方文化" value="western" />
                <el-option label="印度文化" value="indian" />
                <el-option label="地中海文化" value="mediterranean" />
                <el-option label="默认" value="default" />
              </el-select>
            </el-form-item>

            <el-form-item label="最近血糖水平" prop="recent_glucose_levels">
              <el-input
                v-model="formData.recent_glucose_levels"
                placeholder="例如: 120,130,115"
              />
            </el-form-item>

            <el-row :gutter="16">
              <el-col :span="12">
                <el-form-item label="收缩压 (mmHg)" prop="systolic_bp">
                  <el-input-number
                    v-model="formData.systolic_bp"
                    :min="50"
                    :max="250"
                    style="width: 100%;"
                  />
                </el-form-item>
              </el-col>
              <el-col :span="12">
                <el-form-item label="舒张压 (mmHg)" prop="diastolic_bp">
                  <el-input-number
                    v-model="formData.diastolic_bp"
                    :min="30"
                    :max="150"
                    style="width: 100%;"
                  />
                </el-form-item>
              </el-col>
            </el-row>

            <el-form-item>
              <el-button
                type="primary"
                size="large"
                :loading="isLoading"
                @click="handleSubmit"
                style="width: 100%;"
              >
                {{ isLoading ? '评估中...' : '开始健康评估' }}
              </el-button>
            </el-form-item>
          </el-form>
        </el-card>
      </el-col>

      <!-- 评估结果与膳食推荐 -->
      <el-col :span="12">
        <el-card title="评估结果与膳食推荐" class="result-card" shadow="hover">
          <div v-if="isLoading" class="loading-container">
            <el-icon size="48" class="loading-icon">
              <Loading />
            </el-icon>
            <div class="loading-text">AI模型正在进行健康评估...</div>
          </div>

          <div v-else-if="assessmentResult" class="result-content">
            <el-alert
              :title="`健康评分: ${assessmentResult.overall_score.toFixed(1)}/100`"
              :type="assessmentResult.overall_score > 70 ? 'success' : 'warning'"
              show-icon
              class="score-alert"
            />

            <h4>您的健康概况</h4>
            <el-descriptions :column="1" border>
              <el-descriptions-item label="BMI">
                {{ assessmentResult.bmi?.toFixed(2) || 'N/A' }}
                ({{ assessmentResult.bmi_category || 'N/A' }})
              </el-descriptions-item>
              <el-descriptions-item label="风险因素">
                {{ assessmentResult.risk_factors?.join(', ') || '无' }}
              </el-descriptions-item>
              <el-descriptions-item label="健康建议">
                {{ assessmentResult.health_suggestions?.join('; ') || '无' }}
              </el-descriptions-item>
              <el-descriptions-item label="个性化目标">
                {{ assessmentResult.personalized_goals?.join('; ') || '无' }}
              </el-descriptions-item>
            </el-descriptions>

            <el-divider />

            <h4>
              <el-icon><Food /></el-icon>
              个性化膳食推荐
            </h4>

            <el-card v-if="recommendedRecipes.length > 0" class="recipe-recommendations">
              <el-row :gutter="16">
                <el-col :span="12" v-for="recipe in recommendedRecipes" :key="recipe.id">
                  <el-card class="recipe-card" shadow="hover">
                    <template #header>
                      <div class="recipe-header">
                        <span class="recipe-name">{{ recipe.name }}</span>
                        <el-rate v-model="recipe.rating" disabled />
                      </div>
                    </template>
                    <div class="recipe-content">
                      <p class="recipe-description">{{ recipe.description }}</p>
                      <div class="recipe-tags">
                        <el-tag
                          v-for="tag in recipe.tags"
                          :key="tag"
                          size="small"
                          type="info"
                        >
                          {{ tag }}
                        </el-tag>
                      </div>
                    </div>
                  </el-card>
                </el-col>
              </el-row>
            </el-card>

            <el-empty v-else description="暂无推荐菜谱" />
          </div>

          <div v-else class="empty-state">
            <el-icon size="48" color="#d9d9d9">
              <Heart />
            </el-icon>
            <div class="empty-text">请填写左侧信息并开始健康评估</div>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue'
import { ElMessage } from 'element-plus'

// 表单数据
const formData = reactive({
  user_id: 'test_user_123',
  age: 30,
  gender: 'male',
  weight_kg: 70,
  height_cm: 175,
  activity_level: 'moderate',
  health_conditions: [],
  dietary_preferences: [],
  cultural_background: 'chinese',
  recent_glucose_levels: '',
  systolic_bp: 120,
  diastolic_bp: 80
})

// 表单验证规则
const rules = {
  age: [{ required: true, message: '请输入年龄', trigger: 'blur' }],
  gender: [{ required: true, message: '请选择性别', trigger: 'change' }],
  weight_kg: [{ required: true, message: '请输入体重', trigger: 'blur' }],
  height_cm: [{ required: true, message: '请输入身高', trigger: 'blur' }],
  activity_level: [{ required: true, message: '请选择活动水平', trigger: 'change' }]
}

// 响应式数据
const formRef = ref()
const isLoading = ref(false)
const assessmentResult = ref<any>(null)
const recommendedRecipes = ref<any[]>([])

// 提交表单
const handleSubmit = async () => {
  if (!formRef.value) return

  try {
    await formRef.value.validate()
    isLoading.value = true

    // 准备请求数据
    const requestData = {
      ...formData,
      recent_glucose_levels: formData.recent_glucose_levels
        ? formData.recent_glucose_levels.split(',').map(Number)
        : undefined,
      blood_pressure: formData.systolic_bp && formData.diastolic_bp
        ? { systolic: formData.systolic_bp, diastolic: formData.diastolic_bp }
        : undefined
    }

    // 调用健康评估API
    const response = await fetch('/api/v1/health/assess', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(requestData)
    })

    if (response.ok) {
      const result = await response.json()
      assessmentResult.value = result

      // 模拟推荐菜谱
      recommendedRecipes.value = [
        {
          id: 1,
          name: '清蒸鲈鱼',
          description: '低脂高蛋白，适合糖尿病患者',
          rating: 4.5,
          tags: ['低糖', '高蛋白', '中式']
        },
        {
          id: 2,
          name: '蔬菜沙拉',
          description: '富含维生素和膳食纤维',
          rating: 4.2,
          tags: ['素食', '低卡', '健康']
        }
      ]

      ElMessage.success('健康评估完成')
    } else {
      throw new Error('评估失败')
    }
  } catch (error) {
    ElMessage.error('健康评估失败，请重试')
    console.error('Health assessment error:', error)
  } finally {
    isLoading.value = false
  }
}
</script>

<style scoped>
.health-assessment {
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

.loading-container {
  text-align: center;
  padding: 50px;
}

.loading-icon {
  animation: rotate 2s linear infinite;
}

.loading-text {
  margin-top: 16px;
  color: #666;
}

@keyframes rotate {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.result-content {
  padding: 0;
}

.score-alert {
  margin-bottom: 24px;
}

.result-content h4 {
  margin: 16px 0;
  display: flex;
  align-items: center;
}

.empty-state {
  text-align: center;
  padding: 50px;
}

.empty-text {
  margin-top: 16px;
  color: #999;
}

.recipe-recommendations {
  margin-top: 16px;
}

.recipe-card {
  margin-bottom: 16px;
  transition: all 0.3s ease;
}

.recipe-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
}

.recipe-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.recipe-name {
  font-weight: bold;
}

.recipe-content {
  padding: 0;
}

.recipe-description {
  margin: 8px 0;
  color: #666;
}

.recipe-tags {
  margin-top: 8px;
}

.recipe-tags .el-tag {
  margin-right: 8px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .health-assessment .el-row {
    flex-direction: column;
  }

  .health-assessment .el-col {
    width: 100% !important;
    margin-bottom: 24px;
  }
}
</style>