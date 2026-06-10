/**
 * Vue版本API工具模块
 * 提供统一的前端与后端通信接口
 * 结合最佳实践进行优化
 */

import axios from 'axios'
import { ElMessage, ElLoading } from 'element-plus'
import type { AxiosRequestConfig, AxiosResponse } from 'axios'

// 创建axios实例
const api = axios.create({
  baseURL: 'http://localhost:8000/api/v1',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// IndexedDB 轻量缓存（用于文化相容性结果）
const IDB_DB_NAME = 'health-cache'
const IDB_STORE = 'cultural'
const IDB_VERSION = 1
const DEFAULT_TTL_MS = 24 * 60 * 60 * 1000 // 24h

type CachedEntry = { value: any; ts: number; ttl: number }

const openIDB = (): Promise<IDBDatabase> => {
  return new Promise((resolve, reject) => {
    const req = window.indexedDB.open(IDB_DB_NAME, IDB_VERSION)
    req.onupgradeneeded = () => {
      const db = req.result
      if (!db.objectStoreNames.contains(IDB_STORE)) {
        db.createObjectStore(IDB_STORE)
      }
    }
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}

const idbGet = async (key: string): Promise<CachedEntry | undefined> => {
  try {
    const db = await openIDB()
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_STORE, 'readonly')
      const store = tx.objectStore(IDB_STORE)
      const req = store.get(key)
      req.onsuccess = () => resolve(req.result as CachedEntry)
      req.onerror = () => reject(req.error)
    })
  } catch {
    return undefined
  }
}

const idbSet = async (key: string, value: any, ttl = DEFAULT_TTL_MS): Promise<void> => {
  try {
    const db = await openIDB()
    const entry: CachedEntry = { value, ts: Date.now(), ttl }
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(IDB_STORE, 'readwrite')
      const store = tx.objectStore(IDB_STORE)
      const req = store.put(entry, key)
      req.onsuccess = () => resolve()
      req.onerror = () => reject(req.error)
    })
  } catch {
    // 忽略缓存失败
  }
}

const isFresh = (entry?: CachedEntry): boolean => {
  if (!entry) return false
  return Date.now() - entry.ts <= (entry.ttl || DEFAULT_TTL_MS)
}

// 请求拦截器
api.interceptors.request.use(
  (config) => {
    // 添加认证token
    const token = localStorage.getItem('auth_token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }

    // 添加请求ID用于追踪
    config.headers['X-Request-ID'] = Date.now().toString()

    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// 响应拦截器
api.interceptors.response.use(
  (response: AxiosResponse) => {
    // 统一处理成功响应
    return response
  },
  (error) => {
    // 统一错误处理
    if (error.response) {
      const { status, data } = error.response

      switch (status) {
        case 400:
          ElMessage.error(data.detail || '请求参数错误')
          break
        case 401:
          ElMessage.error('未授权，请重新登录')
          // 清除本地存储的token
          localStorage.removeItem('auth_token')
          // 可以在这里处理登录跳转
          break
        case 403:
          ElMessage.error('权限不足')
          break
        case 404:
          ElMessage.error('请求的资源不存在')
          break
        case 422:
          ElMessage.error(data.detail || '数据验证失败')
          break
        case 429:
          ElMessage.error('请求过于频繁，请稍后再试')
          break
        case 500:
          ElMessage.error('服务器内部错误')
          break
        case 502:
          ElMessage.error('网关错误')
          break
        case 503:
          ElMessage.error('服务暂时不可用')
          break
        default:
          ElMessage.error(data.detail || '请求失败')
      }
    } else if (error.request) {
      ElMessage.error('网络连接失败，请检查网络')
    } else {
      ElMessage.error('请求配置错误')
    }

    return Promise.reject(error)
  }
)

// 加载状态管理
let loadingInstance: any = null

const showLoading = (text = '加载中...') => {
  loadingInstance = ElLoading.service({
    lock: true,
    text,
    background: 'rgba(0, 0, 0, 0.7)',
    spinner: 'el-icon-loading',
  })
}

const hideLoading = () => {
  if (loadingInstance) {
    loadingInstance.close()
    loadingInstance = null
  }
}

// API接口定义
export const apiEndpoints = {
  // 健康评估
  health: {
    assess: (data: any) => api.post('/health/assess', data),
    getTrend: () => api.get('/health/trend'),
    getDistribution: () => api.get('/health/distribution'),
    getCulturalOptions: () => api.get('/health/cultural/options'),
    getDietaryPreferences: () => api.get('/health/recipes/dietary-preferences'),
  },

  // 血糖预测
  glucose: {
    predict: (data: any) => api.post('/glucose/predict', data),
    predictEnhanced: (data: any) => api.post('/glucose/predict/enhanced', data),
    predictGluformer: (data: any) => api.post('/glucose/predict/gluformer', data),
    getModelsInfo: () => api.get('/glucose/models/info'),
    getTrend: () => api.get('/stats/glucose/trend'),
  },

  // 菜谱推荐
  recipes: {
    getList: (params?: any) => api.get('/recipes', { params }),
    getDetail: (id: number) => api.get(`/recipes/${id}`),
    create: (data: any) => api.post('/recipes', data),
    update: (id: number, data: any) => api.put(`/recipes/${id}`, data),
    delete: (id: number) => api.delete(`/recipes/${id}`),
    recommend: (data: any) => api.post('/recipes/recommend', data),
    getCategories: () => api.get('/recipes/categories'),
    getHealthTags: () => api.get('/recipes/health-tags'),
  },

  // 文化适配
  cultural: {
    getRecommendations: (data: any) => api.post('/cultural/recommendations', data),
    getRecipes: (params?: any) => api.get('/cultural/recipes', { params }),
    compatibility: async (data: { region: string; item: string }) => {
      const key = `compat:${data.region}:${data.item}`
      const cached = await idbGet(key)
      if (isFresh(cached)) return { data: cached!.value }
      const res = await api.post('/cultural/compatibility', data)
      // 仅缓存服务端返回的 data
      await idbSet(key, res.data)
      return res
    },
    compatibilityBulk: async (data: { region: string; items: string[] }) => {
      const key = `compat-bulk:${data.region}:${data.items.sort().join(',')}`
      const cached = await idbGet(key)
      if (isFresh(cached)) return { data: cached!.value }
      const res = await api.post('/cultural/compatibility/bulk', data)
      await idbSet(key, res.data)
      return res
    },
    upsertTriples: (data: { triples: [string, string, string][] }) => api.post('/cultural/kg/upsert', data),
  },

  // 图像分析
  image: {
    analyze: (file: File) => {
      const formData = new FormData()
      formData.append('file', file)
      return api.post('/image/analyze', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
    },
  },

  // MERL分析
  merl: {
    analyze: (data: any) => api.post('/merl/analyze', data),
    getResults: (params?: any) => api.get('/merl/results', { params }),
  },

  // 可解释性
  explain: {
    explainPrediction: (data: any) => api.post('/explain/prediction', data),
    explainRecommendation: (data: any) => api.post('/explain/recommendation', data),
  },

  // 模型蒸馏
  modeling: {
    distillAlign: (data: { teacher: Record<string, number[][]>; student: Record<string, number[][]>; temperature?: number; domain_weights?: Record<string, number> }) =>
      api.post('/modeling/distill/align', data),
  },

  // 用户管理
  user: {
    getProfile: () => api.get('/user/profile'),
    updateProfile: (data: any) => api.put('/user/profile', data),
    getPreferences: () => api.get('/user/preferences'),
    updatePreferences: (data: any) => api.put('/user/preferences', data),
  },

  // 工作流
  workflow: {
    getDefinitions: () => api.get('/workflow/definitions'),
    executeWorkflow: (data: any) => api.post('/workflow/execute', data),
    getExecutions: (params?: any) => api.get('/workflow/executions', { params }),
  },

  // 数据处理
  data: {
    batchProcess: (data: any) => api.post('/data/batch_process', data),
    getProcessingStatus: (id: string) => api.get(`/data/processing_status/${id}`),
  },

  // 统计信息
  stats: {
    getComprehensive: () => api.get('/stats/comprehensive'),
    getHealthDistribution: () => api.get('/stats/health-distribution'),
    getUserActivity: () => api.get('/stats/user-activity'),
    getPerformance: () => api.get('/stats/performance'),
    getPredictionAccuracy: () => api.get('/stats/predictions/accuracy'),
  },

  // 反馈系统
  feedback: {
    submitFeedback: (data: any) => api.post('/feedback', data),
    getFeedback: (params?: any) => api.get('/feedback', { params }),
  },
}

// 导出默认的api实例
export default api

// 导出常用的API方法
export const {
  health,
  glucose,
  recipes,
  cultural,
  image,
  merl,
  explain,
  user,
  workflow,
  data,
  stats,
  feedback,
} = apiEndpoints

// 导出工具函数
export { showLoading, hideLoading }