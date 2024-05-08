<template>
  <div>
    <h1>股票資訊</h1>
    <!-- 子路由將在這裡被渲染 -->
    <router-view></router-view>
  </div>
</template>

<script>
export default {
  props: ['stock_code']
}
</script>
