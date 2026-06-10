<template>
  <PageTemplate
    title="可解释性分析"
    description="可解释性分析模块提供AI决策过程的透明化展示，帮助用户理解系统推荐和预测的依据。"
    :features="[
      '预测结果解释',
      '推荐理由说明',
      '影响因素分析',
      '决策路径可视化'
    ]"
    button-text="查看可解释性分析"
    icon-component="QuestionFilled"
    icon-color="#fa8c16"
  />
</template>

<script setup lang="ts">
import PageTemplate from '@components/common/PageTemplate.vue'
</script>