<template>
  <div class="accessibility-panel" :class="{ 'panel-open': isOpen }">
    <button
      class="accessibility-toggle"
      @click="togglePanel"
      :aria-label="isOpen ? '关闭无障碍面板' : '打开无障碍面板'"
      :aria-expanded="isOpen"
    >
      <el-icon size="20">
        <component :is="isOpen ? 'Close' : 'Setting'" />
      </el-icon>
    </button>

    <div v-if="isOpen" class="accessibility-content">
      <h3>无障碍设置</h3>

      <div class="setting-group">
        <label class="setting-label">
          <input
            type="checkbox"
            v-model="settings.highContrast"
            @change="applySettings"
          />
          高对比度模式
        </label>
      </div>

      <div class="setting-group">
        <label class="setting-label">
          <input
            type="checkbox"
            v-model="settings.reduceMotion"
            @change="applySettings"
          />
          减少动画效果
        </label>
      </div>

      <div class="setting-group">
        <label class="setting-label">
          <input
            type="checkbox"
            v-model="settings.largeText"
            @change="applySettings"
          />
          大字体模式
        </label>
      </div>

      <div class="setting-group">
        <label class="setting-label">
          字体大小
          <select v-model="settings.fontSize" @change="applySettings">
            <option value="small">小</option>
            <option value="medium">中</option>
            <option value="large">大</option>
            <option value="extra-large">特大</option>
          </select>
        </label>
      </div>

      <div class="setting-group">
        <label class="setting-label">
          颜色主题
          <select v-model="settings.colorTheme" @change="applySettings">
            <option value="default">默认</option>
            <option value="dark">深色</option>
            <option value="light">浅色</option>
            <option value="high-contrast">高对比度</option>
          </select>
        </label>
      </div>

      <div class="setting-group">
        <button
          class="reset-button"
          @click="resetSettings"
          aria-label="重置所有设置"
        >
          重置设置
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { Setting, Close } from '@element-plus/icons-vue'

interface AccessibilitySettings {
  highContrast: boolean
  reduceMotion: boolean
  largeText: boolean
  fontSize: string
  colorTheme: string
}

const isOpen = ref(false)
const settings = reactive<AccessibilitySettings>({
  highContrast: false,
  reduceMotion: false,
  largeText: false,
  fontSize: 'medium',
  colorTheme: 'default'
})

const togglePanel = () => {
  isOpen.value = !isOpen.value
}

const applySettings = () => {
  const root = document.documentElement

  // 应用高对比度
  if (settings.highContrast) {
    root.classList.add('high-contrast')
  } else {
    root.classList.remove('high-contrast')
  }

  // 应用减少动画
  if (settings.reduceMotion) {
    root.classList.add('reduce-motion')
  } else {
    root.classList.remove('reduce-motion')
  }

  // 应用大字体
  if (settings.largeText) {
    root.classList.add('large-text')
  } else {
    root.classList.remove('large-text')
  }

  // 应用字体大小
  root.classList.remove('font-small', 'font-medium', 'font-large', 'font-extra-large')
  root.classList.add(`font-${settings.fontSize}`)

  // 应用颜色主题
  root.classList.remove('theme-default', 'theme-dark', 'theme-light', 'theme-high-contrast')
  root.classList.add(`theme-${settings.colorTheme}`)

  // 保存设置到本地存储
  localStorage.setItem('accessibility-settings', JSON.stringify(settings))
}

const resetSettings = () => {
  Object.assign(settings, {
    highContrast: false,
    reduceMotion: false,
    largeText: false,
    fontSize: 'medium',
    colorTheme: 'default'
  })

  // 清除所有类
  const root = document.documentElement
  root.classList.remove(
    'high-contrast', 'reduce-motion', 'large-text',
    'font-small', 'font-medium', 'font-large', 'font-extra-large',
    'theme-default', 'theme-dark', 'theme-light', 'theme-high-contrast'
  )

  // 清除本地存储
  localStorage.removeItem('accessibility-settings')
}

const loadSettings = () => {
  const saved = localStorage.getItem('accessibility-settings')
  if (saved) {
    try {
      const parsedSettings = JSON.parse(saved)
      Object.assign(settings, parsedSettings)
      applySettings()
    } catch (error) {
      console.error('Failed to load accessibility settings:', error)
    }
  }
}

onMounted(() => {
  loadSettings()
})
</script>

<style scoped>
.accessibility-panel {
  position: fixed;
  top: 20px;
  right: 20px;
  z-index: 1000;
  transition: all 0.3s ease;
}

.accessibility-toggle {
  width: 48px;
  height: 48px;
  border-radius: 50%;
  background: rgba(24, 144, 255, 0.9);
  backdrop-filter: blur(10px);
  border: none;
  color: white;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  transition: all 0.3s ease;
}

.accessibility-toggle:hover {
  background: rgba(24, 144, 255, 1);
  transform: scale(1.05);
}

.accessibility-toggle:focus {
  outline: 2px solid #fff;
  outline-offset: 2px;
}

.accessibility-content {
  position: absolute;
  top: 60px;
  right: 0;
  width: 300px;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.accessibility-content h3 {
  margin: 0 0 16px 0;
  font-size: 16px;
  font-weight: 600;
  color: #333;
}

.setting-group {
  margin-bottom: 16px;
}

.setting-label {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  color: #333;
  cursor: pointer;
}

.setting-label input[type="checkbox"] {
  width: 16px;
  height: 16px;
  accent-color: #1890ff;
}

.setting-label select {
  margin-left: 8px;
  padding: 4px 8px;
  border: 1px solid #d9d9d9;
  border-radius: 4px;
  background: white;
  font-size: 12px;
}

.reset-button {
  width: 100%;
  padding: 8px 16px;
  background: #ff4d4f;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  transition: all 0.3s ease;
}

.reset-button:hover {
  background: #ff7875;
}

.reset-button:focus {
  outline: 2px solid #ff4d4f;
  outline-offset: 2px;
}

/* 无障碍样式 */
:global(.high-contrast) {
  --primary-color: #000000;
  --text-color: #000000;
  --background-color: #ffffff;
  --border-color: #000000;
}

:global(.reduce-motion) * {
  animation-duration: 0.01ms !important;
  animation-iteration-count: 1 !important;
  transition-duration: 0.01ms !important;
}

:global(.large-text) {
  font-size: 1.2em;
}

:global(.font-small) {
  font-size: 0.875rem;
}

:global(.font-medium) {
  font-size: 1rem;
}

:global(.font-large) {
  font-size: 1.125rem;
}

:global(.font-extra-large) {
  font-size: 1.25rem;
}

:global(.theme-dark) {
  --primary-color: #1890ff;
  --text-color: #e5eaf3;
  --background-color: #141414;
  --border-color: #4c4d4f;
}

:global(.theme-light) {
  --primary-color: #1890ff;
  --text-color: #262626;
  --background-color: #ffffff;
  --border-color: #d9d9d9;
}

:global(.theme-high-contrast) {
  --primary-color: #000000;
  --text-color: #000000;
  --background-color: #ffffff;
  --border-color: #000000;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .accessibility-panel {
    top: 10px;
    right: 10px;
  }

  .accessibility-content {
    width: 280px;
    right: -20px;
  }
}
</style>
