# pip install yfinance

# 爬蟲
import requests
from bs4 import BeautifulSoup

# selenium
# https://googlechromelabs.github.io/chrome-for-testing/
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# selenium 無痕模式
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

import time
import pandas as pd


def setup_driver():
    """chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")

    """
    service = Service(executable_path="chromedriver.exe")
    return webdriver.Chrome(service=service)


def fetch_stock_data(stock_id):
    url = f"https://tw.stock.yahoo.com/quote/{stock_id}"
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    stock_data = {}
    price_info = soup.find_all("li", class_="price-detail-item")
    for item in price_info:
        label = item.find("span", class_="C(#232a31)").text.strip()
        value = item.find("span", class_="Fw(600)").text.strip()
        stock_data[label] = value
    volume_info = soup.find("li", class_="price-detail-item", text="總量")
    if volume_info:
        volume = volume_info.find("span", class_="Fw(600)").text.strip()
        stock_data["總量"] = volume
    return stock_data


def fetch_stock_history(stock_id, start_date, end_date):
    driver = setup_driver()
    try:
        url = f"https://hk.finance.yahoo.com/quote/{stock_id}/history"
        driver.get(url)
        time.sleep(5)
        
        # 點擊日期範圍選擇器
        date_range_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//div[@data-test='dropdown']/div/span"))
        )
        date_range_button.click()
        time.sleep(2)

        # 設置開始和結束日期
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.NAME, 'startDate'))
        ).send_keys(start_date)
        
        time.sleep(2)
        
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.NAME, 'endDate'))
        ).send_keys(end_date)
        time.sleep(2)

        # 點選「完成」和「套用」按鈕
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button/span[text()='完成']"))
        ).click()
        time.sleep(2)
        
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button/span[text()='套用']"))
        ).click()
        time.sleep(2)

        # 等待資料表格出現
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "table[data-test='historical-prices']"))
        )
        time.sleep(2)

        # 解析頁面並返回 DataFrame
        soup = BeautifulSoup(driver.page_source, "html.parser")
        table = soup.find("table", {"data-test": "historical-prices"})
        headers = [th.get_text(strip=True) for th in table.find("thead").find_all("th")]
        rows = [[td.get_text(strip=True) for td in tr.find_all("td")] for tr in table.find("tbody").find_all("tr")]
        return pd.DataFrame(rows, columns=headers)

    finally:
        driver.quit()


# 主函數區
"""stock_id = input("請輸入股票編號：")
data = fetch_stock_data(stock_id)
print(data)"""
fetch_stock_history("2618.TW", "2024-01-01", "2024-04-01")
