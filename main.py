import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write("서울의 과거 연평균기온을 분석해 선택한 연도의 예상 기온을 확인합니다.")

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    # 날짜와 기온 열 정리
    df["날짜"] = pd.to_datetime(
        df["날짜"].astype(str).str.strip(),
        errors="coerce"
    )
    df["평균기온"] = pd.to_numeric(
        df["평균기온"], errors="coerce"
    )

    # 날짜 또는 평균기온이 없는 행 제거
    df = df.dropna(subset=["날짜", "평균기온"]).copy()

    # 수업 기준인 2025년까지의 자료만 사용
    df = df[df["날짜"].dt.year <= 2025].copy()
    df["연도"] = df["날짜"].dt.year

    # 연도별 평균기온과 실제 관측일 수 계산
    yearly = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count")
        )
        .reset_index()
    )

    # 관측일이 300일 미만인 연도 제외
    yearly = yearly[yearly["관측일수"] >= 300].copy()
    yearly = yearly.sort_values("연도").reset_index(drop=True)

    return yearly


try:
    yearly = load_data()
except Exception as e:
    st.error("기온 데이터를 불러오지 못했습니다. 인터넷 연결을 확인해 주세요.")
    st.stop()

if len(yearly) < 2:
    st.error("회귀 직선을 계산하기 위한 데이터가 부족합니다.")
    st.stop()

# 선형 회귀 계산
x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)

slope, intercept = np.polyfit(x, y, 1)
correlation = np.corrcoef(x, y)[0, 1]

start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
year_count = len(yearly)

# 회귀 직선
line_x = np.linspace(start_year, 2100, 300)
line_y = slope * line_x + intercept

# 데이터 요약
st.subheader("분석 데이터")

col1, col2, col3 = st.columns(3)
col1.metric("사용한 연도 수", f"{year_count}년")
col2.metric("시작 연도", f"{start_year}년")
col3.metric("끝 연도", f"{end_year}년")

st.caption(
    "2025년까지의 자료 중 유효한 평균기온이 기록된 날이 "
    "300일 이상인 연도만 사용했습니다."
)

# 산점도와 회귀 직선
st.subheader("연도별 연평균기온과 회귀 직선")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=yearly["연도"],
    y=yearly["연평균기온"],
    mode="markers",
    name="연평균기온",
    marker=dict(size=7),
    customdata=yearly[["관측일수"]],
    hovertemplate=(
        "연도: %{x:.0f}년<br>"
        "연평균기온: %{y:.2f}℃<br>"
        "관측일수: %{customdata[0]}일"
        "<extra></extra>"
    )
))

fig.add_trace(go.Scatter(
    x=line_x,
    y=line_y,
    mode="lines",
    name="회귀 직선",
    line=dict(width=3),
    hovertemplate=(
        "연도: %{x:.0f}년<br>"
        "회귀 예상 기온: %{y:.2f}℃"
        "<extra></extra>"
    )
))

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="closest",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(fig, use_container_width=True)

st.metric("연도와 연평균기온의 상관계수", f"{correlation:.4f}")

st.write(
    f"회귀식: 연평균기온 = {slope:.5f} × 연도 "
    f"{intercept:+.2f}"
)

if correlation > 0:
    st.write("상관계수는 양수이므로 연도가 커질수록 기온이 높아지는 경향이 있습니다.")
elif correlation < 0:
    st.write("상관계수는 음수이므로 연도가 커질수록 기온이 낮아지는 경향이 있습니다.")
else:
    st.write("상관계수는 0에 가까워 연도와 기온 사이의 선형 관계가 약합니다.")

# 연도 선택 및 예상 기온
st.subheader("예상 기온 확인")

selected_year = st.slider(
    "예측할 연도를 선택하세요",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1
)

predicted_temp = slope * selected_year + intercept

st.markdown(
    f"<div style='text-align:center; padding:20px;'>"
    f"<div style='font-size:22px;'>{selected_year}년 예상 연평균기온</div>"
    f"<div style='font-size:56px; font-weight:bold;'>{predicted_temp:.2f}℃</div>"
    f"</div>",
    unsafe_allow_html=True
)

st.caption(
    "예상 기온은 과거 연평균기온의 선형 추세를 연장해 계산한 값입니다. "
    "실제 기온과 다를 수 있습니다."
)

with st.expander("연도별 분석 데이터 확인"):
    st.dataframe(
        yearly.rename(columns={
            "연도": "연도",
            "연평균기온": "연평균기온(℃)",
            "관측일수": "관측일수(일)"
        }).round(2),
        use_container_width=True,
        hide_index=True
    )
```
