import { createApp } from 'vue'
import { createPinia } from 'pinia'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import 'element-plus/theme-chalk/dark/css-vars.css'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'
import zhCn from 'element-plus/dist/locale/zh-cn.mjs'
import VueLazyload from 'vue-lazyload'

import App from '@/App.vue'
import router from '@/router'
import '@/styles/main.css'

const app = createApp(App)

// 注册Element Plus图标
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(key, component)
}

// 配置图片懒加载
app.use(VueLazyload, {
  loading: '/loading.gif',
  error: '/error.png',
  observer: true,
  observerOptions: {
    rootMargin: '0px',
    threshold: 0.1
  }
})

app.use(createPinia())
app.use(router)
app.use(ElementPlus, {
  locale: zhCn,
})

app.mount('#app')