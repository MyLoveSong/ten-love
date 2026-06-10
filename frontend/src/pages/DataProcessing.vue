<template>
  <PageTemplate
    title="数据处理"
    description="数据处理模块提供数据清洗、转换、分析等功能，支持大规模健康数据的批量处理和实时分析。"
    :features="[
      '数据清洗与验证',
      '数据格式转换',
      '批量数据处理',
      '实时数据流处理'
    ]"
    button-text="开始数据处理"
    icon-component="Database"
    icon-color="#fa541c"
  />
</template>

<script setup lang="ts">
import PageTemplate from '@components/common/PageTemplate.vue'
</script>