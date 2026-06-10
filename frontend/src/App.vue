<template>
  <el-container class="layout-container">
    <AppSider />
    <el-container>
      <AppHeader />
      <el-main class="main-content">
        <router-view v-slot="{ Component, route }">
          <keep-alive :include="keepAliveComponents">
            <component :is="Component" :key="route.path" />
          </keep-alive>
        </router-view>
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import AppHeader from '@components/layout/AppHeader.vue'
import AppSider from '@components/layout/AppSider.vue'

const route = useRoute()

// 需要缓存的组件
const keepAliveComponents = computed(() => {
  const routes = route.matched.filter(route => route.meta?.keepAlive)
  return routes.map(route => route.name as string)
})
</script>

<style scoped>
.layout-container {
  min-height: 100vh;
}

.main-content {
  margin: 16px;
  background: #f5f5f5;
  border-radius: 8px;
  padding: 24px;
  margin-left: 256px;
  margin-top: 80px;
  min-height: calc(100vh - 112px);
}

/* 响应式设计 */
@media (max-width: 768px) {
  .main-content {
    margin-left: 16px;
    margin-top: 80px;
    padding: 16px;
  }
}

/* 页面切换动画 */
.router-view-enter-active,
.router-view-leave-active {
  transition: all 0.3s ease;
}

.router-view-enter-from {
  opacity: 0;
  transform: translateX(20px);
}

.router-view-leave-to {
  opacity: 0;
  transform: translateX(-20px);
}
</style>