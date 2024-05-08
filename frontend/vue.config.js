const { defineConfig } = require("@vue/cli-service");
module.exports = defineConfig({
  transpileDependencies: true,

  // 為了讓 Vue.js 應用能夠與 Flask 後端通訊，需要在 Vue.js 的開發伺服器中設定代理。
  devServer: {
    proxy: {
      "/stock": {
        target: "http://localhost:5000",
        changeOrigin: true,
        pathRewrite: { "^/stock": "" },
      },
    },
  },
});
