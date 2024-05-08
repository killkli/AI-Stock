@echo off
echo Installing necessary Python packages...
pip install yfinance
pip install TA-Lib
py -3.10 -m pip install TA_Lib-0.4.28-cp310-cp310-win_amd64.whl
pip install tensorflow
pip install flask
pip install numpy
pip install pandas
echo The packages installation is complete.
pause
