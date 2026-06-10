import { createRouter, createWebHistory } from 'vue-router'
import NProgress from 'nprogress'
import 'nprogress/nprogress.css'

// 配置NProgress
NProgress.configure({
  showSpinner: false,
  minimum: 0.1,
  speed: 200,
  trickleSpeed: 200
})

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'Dashboard',
      component: () => import('@pages/Dashboard.vue'),
      meta: {
        title: '仪表板',
        keepAlive: true
      }
    },
    {
      path: '/glucose',
      name: 'GlucosePrediction',
      component: () => import('@pages/GlucosePrediction.vue'),
      meta: {
        title: '血糖预测',
        keepAlive: true
      }
    },
    {
      path: '/health',
      name: 'HealthAssessment',
      component: () => import('@pages/HealthAssessment.vue'),
      meta: {
        title: '健康评估',
        keepAlive: true
      }
    },
    {
      path: '/image',
      name: 'ImageAnalysis',
      component: () => import('@pages/ImageAnalysis.vue'),
      meta: {
        title: '图像分析',
        keepAlive: false
      }
    },
    {
      path: '/cultural',
      name: 'CulturalRecommendations',
      component: () => import('@pages/CulturalRecommendations.vue'),
      meta: {
        title: '文化适配',
        keepAlive: false
      }
    },
    {
      path: '/merl',
      name: 'MERLAnalysis',
      component: () => import('@pages/MERLAnalysis.vue'),
      meta: {
        title: 'MERL分析',
        keepAlive: false
      }
    },
    {
      path: '/explain',
      name: 'Explainability',
      component: () => import('@pages/Explainability.vue'),
      meta: {
        title: '可解释性',
        keepAlive: false
      }
    },
    {
      path: '/users',
      name: 'UserManagement',
      component: () => import('@pages/UserManagement.vue'),
      meta: {
        title: '用户管理',
        keepAlive: false
      }
    },
    {
      path: '/workflow',
      name: 'WorkflowManagement',
      component: () => import('@pages/WorkflowManagement.vue'),
      meta: {
        title: '工作流管理',
        keepAlive: false
      }
    },
    {
      path: '/data',
      name: 'DataProcessing',
      component: () => import('@pages/DataProcessing.vue'),
      meta: {
        title: '数据处理',
        keepAlive: false
      }
    },
    {
      path: '/stats',
      name: 'Statistics',
      component: () => import('@pages/Statistics.vue'),
      meta: {
        title: '统计分析',
        keepAlive: true
      }
    }
  ]
})

// 路由守卫
router.beforeEach((to, from, next) => {
  NProgress.start()

  // 设置页面标题
  if (to.meta.title) {
    document.title = `${to.meta.title} - 学术级智能健康监测系统`
  }

  next()
})

router.afterEach(() => {
  NProgress.done()
})

export default router