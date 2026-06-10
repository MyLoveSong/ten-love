<template>
  <el-aside width="240px" class="sidebar">
    <div class="logo">
      <el-icon size="32" color="#1890ff">
        <Heart />
      </el-icon>
      <span class="logo-text">智能健康监测</span>
    </div>

    <el-menu
      :default-active="activeMenu"
      class="sidebar-menu"
      background-color="#001529"
      text-color="#fff"
      active-text-color="#1890ff"
      router
      :collapse="isCollapse"
    >
      <el-menu-item index="/">
        <el-icon><Dashboard /></el-icon>
        <span>仪表板</span>
      </el-menu-item>

      <el-menu-item index="/health">
        <el-icon><Heart /></el-icon>
        <span>健康评估</span>
      </el-menu-item>

      <el-menu-item index="/glucose">
        <el-icon><TrendCharts /></el-icon>
        <span>血糖预测</span>
      </el-menu-item>

      <el-menu-item index="/recipes">
        <el-icon><Food /></el-icon>
        <span>菜谱推荐</span>
      </el-menu-item>

      <el-menu-item index="/image">
        <el-icon><Camera /></el-icon>
        <span>图像分析</span>
      </el-menu-item>

      <el-menu-item index="/cultural">
        <el-icon><Globe /></el-icon>
        <span>文化适配</span>
      </el-menu-item>

      <el-menu-item index="/merl">
        <el-icon><Experiment /></el-icon>
        <span>MERL分析</span>
      </el-menu-item>

      <el-menu-item index="/explain">
        <el-icon><QuestionFilled /></el-icon>
        <span>可解释性</span>
      </el-menu-item>

      <el-menu-item index="/users">
        <el-icon><User /></el-icon>
        <span>用户管理</span>
      </el-menu-item>

      <el-menu-item index="/workflow">
        <el-icon><Setting /></el-icon>
        <span>工作流</span>
      </el-menu-item>

      <el-menu-item index="/data">
        <el-icon><Database /></el-icon>
        <span>数据处理</span>
      </el-menu-item>

      <el-menu-item index="/stats">
        <el-icon><DataAnalysis /></el-icon>
        <span>统计分析</span>
      </el-menu-item>
    </el-menu>

    <!-- 折叠按钮 -->
    <div class="collapse-btn" @click="toggleCollapse">
      <el-icon>
        <component :is="isCollapse ? 'Expand' : 'Fold'" />
      </el-icon>
    </div>
  </el-aside>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const isCollapse = ref(false)

const activeMenu = computed(() => route.path)

const toggleCollapse = () => {
  isCollapse.value = !isCollapse.value
}
</script>

<style scoped>
.sidebar {
  background-color: #001529;
  height: 100vh;
  position: fixed;
  left: 0;
  top: 0;
  z-index: 1000;
  transition: width 0.3s ease;
}

.logo {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 16px;
  border-bottom: 1px solid #002140;
  transition: all 0.3s ease;
}

.logo-text {
  margin-left: 8px;
  font-size: 16px;
  font-weight: bold;
  color: #fff;
  transition: opacity 0.3s ease;
}

.sidebar-menu {
  border: none;
  height: calc(100vh - 120px);
  overflow-y: auto;
  transition: all 0.3s ease;
}

.sidebar-menu .el-menu-item {
  height: 50px;
  line-height: 50px;
  transition: all 0.3s ease;
}

.sidebar-menu .el-menu-item:hover {
  background-color: #1890ff !important;
  transform: translateX(2px);
}

.sidebar-menu .el-menu-item.is-active {
  background-color: #1890ff !important;
  position: relative;
}

.sidebar-menu .el-menu-item.is-active::after {
  content: '';
  position: absolute;
  right: 0;
  top: 0;
  bottom: 0;
  width: 3px;
  background-color: #fff;
}

.collapse-btn {
  position: absolute;
  bottom: 20px;
  right: 20px;
  width: 40px;
  height: 40px;
  background-color: #1890ff;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.3s ease;
  color: #fff;
}

.collapse-btn:hover {
  background-color: #40a9ff;
  transform: scale(1.1);
}

/* 滚动条样式 */
.sidebar-menu::-webkit-scrollbar {
  width: 4px;
}

.sidebar-menu::-webkit-scrollbar-track {
  background: #002140;
}

.sidebar-menu::-webkit-scrollbar-thumb {
  background: #1890ff;
  border-radius: 2px;
}

.sidebar-menu::-webkit-scrollbar-thumb:hover {
  background: #40a9ff;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .sidebar {
    width: 60px !important;
  }

  .logo-text {
    display: none;
  }

  .sidebar-menu .el-menu-item span {
    display: none;
  }
}
</style>