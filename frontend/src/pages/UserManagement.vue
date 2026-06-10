<template>
  <PageTemplate
    title="用户管理"
    description="用户管理模块提供用户信息管理、权限控制、健康档案维护等功能，确保用户数据的安全性和隐私保护。"
    :features="[
      '用户信息维护',
      '健康档案管理',
      '权限控制',
      '数据隐私保护'
    ]"
    button-text="进入用户管理"
    icon-component="User"
    icon-color="#13c2c2"
  />
</template>

<script setup lang="ts">
import PageTemplate from '@components/common/PageTemplate.vue'
</script>