<template>
    <div>
        <h1>股票編號：{{ stockCode }}</h1>
        <p>當前股價: {{ stockInfo.currentPrice }}</p>
        <p>上一交易日的收盤價: {{ stockInfo.PreviousClose }}</p>    
        <p>公司市值: {{ stockInfo.MarketCap }}</p>
        <p>當日成交量: {{ stockInfo.Volume }}</p>
        <p>過去52週的最高股價: {{ stockInfo.WeekHigh }}</p>
        <p>過去52週的最低股價: {{ stockInfo.WeekLow }}</p>
    </div>
</template>

<script>
import axios from 'axios';

export default {
    name: "currentView",
    props: ['stockCode'],  // 從路由接收 stockCode 參數
    data() {
        return {
            stockInfo: {}
        };
    },
    mounted() {
        this.fetchStockData();
    },
    methods: {
        fetchStockData() {
            axios.get(`/stock/${this.stockCode}/getcurrent`)
                .then(response => {
                    this.stockInfo = response.data;
                })
                .catch(error => {
                    console.error('Error fetching stock data:', error);
                });
        }
    }
}
</script>