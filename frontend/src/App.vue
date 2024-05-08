<template>
  <div>
    <div class="container">
      <h1>股票分析助手</h1>
      <p>※股票資訊來自奇摩股市</p>
      <!-- @submit.prevent：阻止表單默認跳轉行為 -->
      <form class="search" @submit.prevent="handleSubmit">
        <div class="search-bar">
          <!-- v-model 會根據輸入框的值動態更新 stockCode變數值 -->
          <input type="text" v-model="stockCode" class="search-input" placeholder="輸入股票代號...">
          <button class="search-button" type="submit">搜尋</button>
        </div>
      </form>
    </div>
    <router-view></router-view>
  </div>
</template>
<script>
import { useRouter } from 'vue-router'
import { ref } from 'vue'

export default {
  name: 'App',
  setup() {
    const router = useRouter()
    const stockCode = ref('');
    const handleSubmit = () => {
      router.push({ path: `/stock/${stockCode.value}/current` });
    }
    return { handleSubmit, stockCode }
  },
}
</script>

<style>
html,
body {
  height: 100%;
  /* 確保html和body元素高度為瀏覽器視窗高度 */
  padding: 0px;
  margin: 0px;
  display: flex;
  /* 啟用flexbox */
  justify-content: center;
  /* 水平置中 */
  align-items: center;
  /* 垂直置中 */
}

.container {
  text-align: center;
  /* 文字置中 */
}

body {
  background-color: #ffb310;
}

.search {
  font-family: Arial, sans-serif;
}

.search-bar {
  display: flex;
  border: 2px solid #11862f;
  border-radius: 5px;
  overflow: hidden;
}

.search-input {
  padding: 10px;
  border: none;
  outline: none;
  font-size: 16px;
  width: 300px;
}

.search-button {
  padding: 10px 20px;
  background-color: #11862f;
  color: white;
  border: none;
  cursor: pointer;
  font-size: 16px;
}

.search-button:hover {
  background-color: #01cd74;
}
</style>
