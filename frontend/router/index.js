import { createRouter, createWebHistory } from 'vue-router';
import Stock from '../src/views/Stock.vue'
import CurrentStock from '../src/views/stockpages/CurrentStock.vue'

Vue.use(VueRouter);

const routes = [
  {
    path: '/stock/:stock_code',
    component: Stock,
    props: true,
    children: [
      {
        path: 'current',
        component: CurrentStock,
        props: true  // 允許從路由參數傳遞props到組件
      }
    ]
  }
];

const router = createRouter({
  history: createWebHistory(),
  routes
});

export default router;