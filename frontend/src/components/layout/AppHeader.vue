<template>
  <el-header class="header">
    <div class="header-left">
      <el-breadcrumb separator="/">
        <el-breadcrumb-item :to="{ path: '/' }">首页</el-breadcrumb-item>
        <el-breadcrumb-item>{{ currentPageTitle }}</el-breadcrumb-item>
      </el-breadcrumb>
    </div>

    <div class="header-right">
      <el-space>
        <!-- 搜索框 -->
        <el-input
          v-model="searchKeyword"
          placeholder="搜索功能..."
          class="search-input"
          @keyup.enter="handleSearch"
        >
          <template #prefix>
            <el-icon><Search /></el-icon>
          </template>
        </el-input>

        <!-- 通知 -->
        <el-badge :value="notificationCount" class="notification-badge">
          <el-button circle @click="showNotifications">
            <el-icon><Bell /></el-icon>
          </el-button>
        </el-badge>

        <!-- 全屏切换 -->
        <el-button circle @click="toggleFullscreen">
          <el-icon>
            <component :is="isFullscreen ? 'Aim' : 'FullScreen'" />
          </el-icon>
        </el-button>

        <!-- 用户头像 -->
        <el-dropdown @command="handleUserCommand">
          <el-avatar
            :size="32"
            :src="userAvatar"
            class="user-avatar"
          >
            <el-icon><User /></el-icon>
          </el-avatar>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item command="profile">
                <el-icon><User /></el-icon>
                个人中心
              </el-dropdown-item>
              <el-dropdown-item command="settings">
                <el-icon><Setting /></el-icon>
                设置
              </el-dropdown-item>
              <el-dropdown-item divided command="logout">
                <el-icon><SwitchButton /></el-icon>
                退出登录
              </el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>
      </el-space>
    </div>
  </el-header>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'

const route = useRoute()
const router = useRouter()

// 响应式数据
const searchKeyword = ref('')
const notificationCount = ref(12)
const isFullscreen = ref(false)
const userAvatar = ref('https://cube.elemecdn.com/0/88/03b0d39583f48206768a7534e55bcpng.png')

// 计算属性
const currentPageTitle = computed(() => {
  const titleMap: Record<string, string> = {
    '/': '仪表板',
    '/health': '健康评估',
    '/glucose': '血糖预测',
    '/recipes': '菜谱推荐',
    '/image': '图像分析',
    '/cultural': '文化适配',
    '/merl': 'MERL分析',
    '/explain': '可解释性',
    '/users': '用户管理',
    '/workflow': '工作流管理',
    '/data': '数据处理',
    '/stats': '统计分析'
  }
  return titleMap[route.path] || '未知页面'
})

// 方法
const handleSearch = () => {
  if (searchKeyword.value.trim()) {
    ElMessage.info(`搜索: ${searchKeyword.value}`)
    // 这里可以添加搜索逻辑
  }
}

const showNotifications = () => {
  ElMessage.info('查看通知')
  // 这里可以添加通知面板逻辑
}

const toggleFullscreen = () => {
  if (!document.fullscreenElement) {
    document.documentElement.requestFullscreen()
    isFullscreen.value = true
  } else {
    document.exitFullscreen()
    isFullscreen.value = false
  }
}

const handleUserCommand = (command: string) => {
  switch (command) {
    case 'profile':
      ElMessage.info('打开个人中心')
      break
    case 'settings':
      ElMessage.info('打开设置')
      break
    case 'logout':
      ElMessage.info('退出登录')
      break
  }
}
</script>

<style scoped>
.header {
  background: #fff;
  border-bottom: 1px solid #e8e8e8;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  margin-left: 240px;
  position: fixed;
  top: 0;
  right: 0;
  left: 240px;
  z-index: 999;
  height: 64px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.header-left {
  flex: 1;
}

.header-right {
  display: flex;
  align-items: center;
}

.search-input {
  width: 200px;
}

.notification-badge {
  margin-right: 16px;
}

.user-avatar {
  cursor: pointer;
  transition: all 0.3s ease;
}

.user-avatar:hover {
  transform: scale(1.1);
}

/* 响应式设计 */
@media (max-width: 768px) {
  .header {
    margin-left: 60px;
    left: 60px;
    padding: 0 16px;
  }

  .search-input {
    width: 120px;
  }

  .header-right .el-space {
    gap: 8px !important;
  }
}

/* 动画效果 */
.header {
  transition: all 0.3s ease;
}

.el-button {
  transition: all 0.3s ease;
}

.el-button:hover {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}
</style>