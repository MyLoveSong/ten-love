<template>
  <PageTemplate
    title="工作流管理"
    description="工作流管理模块提供业务流程自动化、任务调度、流程监控等功能，支持复杂的健康数据处理和分析流程。"
    :features="[
      '流程定义与配置',
      '任务调度管理',
      '流程执行监控',
      '异常处理机制'
    ]"
    button-text="管理工作流"
    icon-component="Setting"
    icon-color="#eb2f96"
  />
</template>

<script setup lang="ts">
import PageTemplate from '@components/common/PageTemplate.vue'
</script>