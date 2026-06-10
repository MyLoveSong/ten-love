<template>
  <PageTemplate
    title="文化适配推荐"
    description="文化适配推荐模块根据用户的文化背景、饮食习惯和偏好，提供个性化的菜谱推荐和饮食建议。"
    :features="[
      '中华文化 - 传统中式饮食',
      '西方文化 - 西式料理',
      '印度文化 - 印度风味',
      '地中海文化 - 地中海饮食',
      '日本文化 - 日式料理'
    ]"
    button-text="开始文化适配推荐"
    icon-component="Globe"
    icon-color="#52c41a"
  />
</template>

<script setup lang="ts">
import PageTemplate from '@components/common/PageTemplate.vue'
</script>