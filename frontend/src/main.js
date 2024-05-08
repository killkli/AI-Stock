import { createApp } from 'vue'
import App from './App.vue'

import router from '../router/index';  // 確保這行導入的路由配置是正確的路徑
createApp(App).use(router).mount('#app')
