import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from FinMind.data import DataLoader

# --- 頁面設定 ---
st.set_page_config(page_title="台股量化分析 App", layout="wide")
st.title("📈 台股量化分析 App (含策略回測)")

# --- 側邊欄：全域設定 ---
st.sidebar.header("1. 查詢設定")
stock_id = st.sidebar.text_input("輸入台股代號", value="2330")
days_to_look = st.sidebar.selectbox("資料期間", [180, 365, 730, 1095], index=2, format_func=lambda x: f"近 {x} 天")

# --- 核心資料函數 ---
@st.cache_data(ttl=3600) # 快取資料避免重複下載
def get_data(symbol, days):
    """下載股價與籌碼並合併"""
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    # 1. 抓股價 (yfinance)
    ticker = f"{symbol}.TW"
    df_price = yf.download(ticker, start=start_date, progress=False)
    
    if df_price.empty: return None

    # 處理 MultiIndex (yfinance 新版可能的格式問題)
    if isinstance(df_price.columns, pd.MultiIndex):
        df_price.columns = df_price.columns.get_level_values(0)
    
    # 2. 抓籌碼 (FinMind)
    api = DataLoader()
    df_chip = api.taiwan_stock_institutional_investors(
        stock_id=symbol,
        start_date=start_date,
        end_date=datetime.now().strftime('%Y-%m-%d')
    )
    
    # 合併資料流程
    if not df_chip.empty:
        # 整理籌碼
        df_chip['net'] = df_chip['buy'] - df_chip['sell']
        df_chip_pivot = df_chip.pivot_table(index='date', columns='name', values='net', aggfunc='sum').fillna(0)
        df_chip_pivot.index = pd.to_datetime(df_chip_pivot.index)
        
        # 確保時區一致 (移除時區資訊以便合併)
        df_price.index = df_price.index.tz_localize(None)
        
        # 合併 (以股價的日期為準)
        df = df_price.join(df_chip_pivot, how='left').fillna(0)
    else:
        df = df_price
        df['Foreign_Investor'] = 0 # 若無籌碼資料補 0
        
    return df

# --- 回測邏輯函數 ---
def run_backtest(df, ma_window=20):
    """執行向量化回測"""
    data = df.copy()
    
    # 1. 計算指標
    data['MA'] = data['Close'].rolling(window=ma_window).mean()
    data['Daily_Return'] = data['Close'].pct_change()
    
    # 2. 產生訊號 (策略：收盤 > MA 且 外資買超 > 0)
    # 使用 shift(1) 是因為今天的訊號只能用來決定「明天」的動作 (避免偷看答案)
    condition_tech = data['Close'] > data['MA']
    condition_chip = data['Foreign_Investor'] > 0
    
    # 持有訊號：當兩者皆成立，設定為持有 (1)，否則空手 (0)
    data['Signal'] = (condition_tech & condition_chip).astype(int)
    
    # 3. 計算策略報酬
    # 今天的部位 * 明天的漲跌 = 策略獲利
    data['Strategy_Return'] = data['Signal'].shift(1) * data['Daily_Return']
    
    # 4. 計算累計報酬 (Equity Curve)
    data['Cum_Market'] = (1 + data['Daily_Return']).cumprod()
    data['Cum_Strategy'] = (1 + data['Strategy_Return']).cumprod()
    
    return data

# --- 主程式 ---
if stock_id:
    with st.spinner('正在下載並分析大數據...'):
        df = get_data(stock_id, days_to_look)

    if df is None or df.empty:
        st.error("找不到資料，請確認代號是否正確。")
    else:
        # 建立分頁 (Tabs)
        tab1, tab2 = st.tabs(["📊 行情分析", "🧪 策略回測"])

        # === 分頁 1: 行情分析 (原本的功能) ===
        with tab1:
            st.subheader(f"{stock_id} 股價與籌碼走勢")
            
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                row_heights=[0.7, 0.3], vertical_spacing=0.03,
                                subplot_titles=("K線與均線", "外資買賣超"))

            # K線
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                                         low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
            # MA20
            ma20 = df['Close'].rolling(window=20).mean()
            fig.add_trace(go.Scatter(x=df.index, y=ma20, mode='lines', line=dict(color='orange'), name='20MA'), row=1, col=1)

            # 外資
            if 'Foreign_Investor' in df.columns:
                fi = df['Foreign_Investor']
                colors = ['red' if v > 0 else 'green' for v in fi]
                fig.add_trace(go.Bar(x=df.index, y=fi, marker_color=colors, name='外資'), row=2, col=1)

            fig.update_layout(height=600, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

        # === 分頁 2: 策略回測 (新功能) ===
        with tab2:
            st.subheader("🧪 外資順勢策略回測")
            st.markdown("""
            **策略邏輯**：
            1. 當 **收盤價 > 20日均線** (多頭趨勢)
            2. 且 **外資今日買超 > 0** (主力進場)
            3. **隔日開盤買進持有**；若條件消失則賣出空手。
            """)
            
            # 執行回測
            res = run_backtest(df)
            
            # --- 計算績效指標 ---
            total_return = (res['Cum_Strategy'].iloc[-1] - 1) * 100
            market_return = (res['Cum_Market'].iloc[-1] - 1) * 100
            
            # 交易天數 (有持有部位的天數)
            trade_days = res['Signal'].sum()
            # 勝率 (持有且當日上漲的天數 / 總持有天數)
            if trade_days > 0:
                win_days = res[(res['Signal'].shift(1) == 1) & (res['Strategy_Return'] > 0)].shape[0]
                win_rate = (win_days / trade_days) * 100
            else:
                win_rate = 0

            # 顯示指標
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("策略總報酬", f"{total_return:.2f}%", delta=f"{total_return - market_return:.2f}% vs 大盤")
            col2.metric("大盤(買進持有)報酬", f"{market_return:.2f}%")
            col3.metric("交易勝率", f"{win_rate:.2f}%")
            col4.metric("持有天數", f"{int(trade_days)} 天")

            # --- 繪製績效曲線 ---
            fig_backtest = go.Figure()
            fig_backtest.add_trace(go.Scatter(x=res.index, y=res['Cum_Strategy'], mode='lines', name='策略績效', line=dict(color='red', width=2)))
            fig_backtest.add_trace(go.Scatter(x=res.index, y=res['Cum_Market'], mode='lines', name='買進持有 (Benchmark)', line=dict(color='gray', dash='dash')))
            
            fig_backtest.update_layout(title="資產累計淨值曲線 (起始值=1)", xaxis_title="日期", yaxis_title="淨值", height=500)
            st.plotly_chart(fig_backtest, use_container_width=True)
            
            with st.expander("查看詳細每日回測數據"):
                st.dataframe(res[['Close', 'Foreign_Investor', 'Signal', 'Strategy_Return', 'Cum_Strategy']].sort_index(ascending=False))