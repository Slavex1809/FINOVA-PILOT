import pandas as pd
import numpy as np
import requests
import yfinance as yf
import ccxt
import time
from datetime import datetime
from config import CONFIG

class RealMoexLoader:
    """Загрузчик данных с Московской биржи (MOEX ISS API)"""
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'FINOVA-PILOT/1.0 (Educational Research)'})
        self.base_url = "https://iss.moex.com/iss"
    
    def _fetch_paginated(self, url, params, data_key):
        all_data = []
        start = 0
        while True:
            params['start'] = start
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data_key not in data or not data[data_key]['data']:
                break
            all_data.extend(data[data_key]['data'])
            if len(data[data_key]['data']) < 100:
                break
            start += len(data[data_key]['data'])
            time.sleep(0.1)
        return all_data, data[data_key]['columns'] if all_data else ([], [])
    
    def get_imoex_history(self, start_date, end_date):
        url = f"{self.base_url}/history/engines/stock/markets/index/securities/IMOEX.json"
        params = {'from': start_date, 'till': end_date, 'interval': 24, 'iss.meta': 'off'}
        data, cols = self._fetch_paginated(url, params, 'history')
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=cols)
        if 'TRADEDATE' in df.columns:
            df['date'] = pd.to_datetime(df['TRADEDATE'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
        return df
    
    def get_stock_history(self, ticker, start_date, end_date, board="TQBR"):
        url = f"{self.base_url}/history/engines/stock/markets/shares/boards/{board}/securities/{ticker}.json"
        params = {'from': start_date, 'till': end_date, 'iss.meta': 'off'}
        data, cols = self._fetch_paginated(url, params, 'history')
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=cols)
        if 'TRADEDATE' in df.columns:
            df['date'] = pd.to_datetime(df['TRADEDATE'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
        return df

class RealTimeDataConnector:
    def __init__(self, config):
        self.config = config
        self.moex = RealMoexLoader()
        self.binance = ccxt.binance({'enableRateLimit': True})
        self.session = requests.Session()
    
    def _fetch_crypto(self, symbols=None, start_date=None, end_date=None) -> pd.DataFrame:
        if symbols is None:
            symbols = ['BTC/USDT', 'ETH/USDT']
        data = {}
        for sym in symbols:
            try:
                since_ts = self.binance.parse8601((start_date or '2018-01-01') + 'T00:00:00')
                ohlcv = self.binance.fetch_ohlcv(sym, '1d', since=since_ts, limit=3000)
                if not ohlcv:
                    raise ValueError("Empty response")
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('date', inplace=True)
                df = df[['close']].rename(columns={'close': sym.replace('/', '_')})
                data[sym.replace('/', '_')] = df
                print(f"✅ Crypto: {sym} loaded")
            except Exception as e:
                print(f"⚠️ CCXT {sym} failed: {e}. Trying CoinGecko fallback...")
                try:
                    base = sym.split('/')[0].lower()
                    url = f"https://api.coingecko.com/api/v3/coins/{base}/market_chart?vs_currency=usd&days=2500"
                    resp = self.session.get(url, timeout=15).json()
                    prices = pd.DataFrame(resp['prices'], columns=['date', 'price'])
                    prices['date'] = pd.to_datetime(prices['date'], unit='ms')
                    prices.set_index('date', inplace=True)
                    prices.rename(columns={'price': sym.replace('/', '_')}, inplace=True)
                    data[sym.replace('/', '_')] = prices
                    print(f"✅ Fallback: {sym} via CoinGecko")
                except Exception as e2:
                    print(f"❌ Fallback failed for {sym}: {e2}")
        return pd.concat(data.values(), axis=1) if data else pd.DataFrame()
    
    def _fetch_sp500(self, start_date, end_date) -> pd.DataFrame:
        try:
            yf.pdr_override()
            df = yf.download('^GSPC', start=start_date, end=end_date, progress=False, timeout=30)
            if df.empty or len(df) < 10:
                raise ValueError("Empty/invalid data")
            df = df[['Close']].rename(columns={'Close': 'GSPC'})
            df.index = df.index.tz_localize(None)
            print("✅ S&P 500 via yfinance")
            return df
        except Exception as e:
            print(f"❌ S&P 500 fully failed. Using synthetic fallback.")
            dates = pd.bdate_range(start=start_date, end=end_date)
            np.random.seed(42)
            synthetic = pd.DataFrame({'GSPC': 4000 + np.cumsum(np.random.randn(len(dates)) * 15)}, index=dates)
            return synthetic
    
    def get_historical_prices(self, start_date=None, end_date=None) -> pd.DataFrame:
        if start_date is None:
            start_date = self.config.START_DATE
        if end_date is None:
            end_date = self.config.END_DATE
        print("📡 Загрузка исторических данных...")
        
        # MOEX
        moex_imoex = self.moex.get_imoex_history(start_date, end_date)
        moex_sber = self.moex.get_stock_history('SBER', start_date, end_date)
        moex_gazp = self.moex.get_stock_history('GAZP', start_date, end_date)
        moex_lkoh = self.moex.get_stock_history('LKOH', start_date, end_date)
        moex_df = pd.concat([moex_imoex['CLOSE'], moex_sber['CLOSE'], moex_gazp['CLOSE'], moex_lkoh['CLOSE']], axis=1)
        moex_df.columns = ['IMOEX', 'SBER', 'GAZP', 'LKOH']
        
        # Crypto
        crypto_df = self._fetch_crypto(start_date=start_date, end_date=end_date)
        
        # S&P 500
        sp500_df = self._fetch_sp500(start_date, end_date)
        
        # Объединяем
        df = pd.concat([moex_df, crypto_df, sp500_df], axis=1).dropna(how='all')
        df = df.ffill().bfill()
        print(f"✅ Загружено {len(df)} дней, активы: {list(df.columns)}")
        return df
    
    def get_market_data(self):
        return {
            'imoex_volatility': 0.2,
            'market_regime': 'neutral',
            'timestamp': datetime.now().isoformat()
        }
