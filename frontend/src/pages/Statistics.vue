<template>
  <PageTemplate
    title="统计分析"
    description="统计分析模块提供全面的数据统计和分析功能，包括用户行为分析、系统性能监控、健康趋势分析等。"
    :features="[
      '用户行为统计',
      '系统性能分析',
      '健康趋势分析',
      '预测准确率统计'
    ]"
    button-text="查看统计分析"
    icon-component="DataAnalysis"
    icon-color="#2f54eb"
  />
</template>

<script setup lang="ts">
import PageTemplate from '@components/common/PageTemplate.vue'
</script>