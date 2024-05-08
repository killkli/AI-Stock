/* eslint-disable */
import { createRouter, createWebHistory } from 'vue-router';
import StockView from '../src/views/StockView.vue'
import CurrentStock from '../src/views/stockpages/CurrentStock.vue'

const routes = [
  {
    path: '/stock/:stock_code',
    component: StockView,
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
