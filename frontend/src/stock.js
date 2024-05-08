import { createApp } from 'vue'
import router from '../router/index';  // 確保這行導入的路由配置是正確的路徑

// 導入主框架組件
import Stock from "./views/Stock.vue"

createApp(Stock).use(router).mount('#main');
