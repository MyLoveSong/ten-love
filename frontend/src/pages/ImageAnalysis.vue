<template>
  <PageTemplate
    title="图像分析"
    description="图像分析模块提供智能图像识别功能，支持食物识别、营养成分分析等。用户可以上传食物图片，系统将自动识别食物类型并提供营养信息。"
    :features="[
      '智能食物识别',
      '营养成分分析',
      '卡路里计算',
      '过敏原检测'
    ]"
    button-text="开始图像分析"
    icon-component="Camera"
    icon-color="#1890ff"
  />
</template>

<script setup lang="ts">
import PageTemplate from '@components/common/PageTemplate.vue'
</script>