
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import httpx
import numpy as np
from scipy.stats import kurtosis, skew

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def compute_realized_volatility(returns):
    return np.sqrt(np.sum(np.square(returns)))

def compute_jb_stat(returns):
    n = len(returns)
    s2 = skew(returns)**2
    k = kurtosis(returns, fisher=False)
    jb = (n / 6) * (s2 + (1/4) * ((k - 3)**2))
    return jb

@app.get("/api/prediction")
async def get_prediction():
    symbol = "BTCUSDT"
    interval = "30m"
    limit = 31
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"

    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        data = response.json()

    closes = [float(candle[4]) for candle in data]
    returns = np.diff(np.log(closes))

    vol_t = compute_realized_volatility(returns[:-1])
    vol_th = compute_realized_volatility(returns[1:])
    jb_t = compute_jb_stat(returns[:-1])
    jb_th = compute_jb_stat(returns[1:])

    prediction = "NEUTRAL"
    if (vol_th - vol_t > 0) and (jb_th - jb_t > 0):
        prediction = "DOWN"
    elif (vol_th - vol_t < 0) and (jb_th - jb_t < 0):
        prediction = "UP"

    return {
        "prediction": prediction,
        "vol_t": vol_t,
        "vol_th": vol_th,
        "jb_t": jb_t,
        "jb_th": jb_th
    }
