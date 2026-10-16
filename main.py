
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
st.write("서울의 과거 연평균기온을 분석하고 미래 기온을 예측합니다.")

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


# 1. 데이터 불러오기 및 전처리
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    df["날짜"] = pd.to_datetime(
        df["날짜"].astype(str).str.strip(),
        errors="coerce"
    )
    df["평균기온"] = pd.to_numeric(
        df["평균기온"], errors="coerce"
    )

    df = df.dropna(subset=["날짜", "평균기온"]).copy()

    # 2025년까지의 데이터만 사용
    df = df[df["날짜"].dt.year <= 2025].copy()
    df["연도"] = df["날짜"].dt.year

    # 연도별 평균기온과 관측일 수 계산
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
    st.error(f"데이터를 불러오지 못했습니다: {e}")
    st.stop()


# 2. 데이터 확인
if len(yearly) < 2:
    st.error("회귀 분석에 필요한 데이터가 부족합니다.")
    st.stop()

start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
year_count = len(yearly)

st.subheader("분석 데이터")

c1, c2, c3 = st.columns(3)
c1.metric("사용한 연도 수", f"{year_count}년")
c2.metric("시작 연도", f"{start_year}년")
c3.metric("끝 연도", f"{end_year}년")

st.caption(
    "2025년까지의 자료 중 관측일이 300일 이상인 연도만 사용했습니다."
)


# 3. 전체 기간 회귀 분석
x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)

slope, intercept = np.polyfit(x, y, 1)
correlation = np.corrcoef(x, y)[0, 1]

# 100년당 기온 변화량
change_per_100 = slope * 100


# 4. 최근 20년 회귀 분석
# 마지막으로 사용 가능한 연도를 기준으로 20년 범위 설정
recent_start = end_year - 19
recent = yearly[yearly["연도"] >= recent_start].copy()

recent_slope = None
recent_intercept = None
recent_change_per_100 = None

if len(recent) >= 2:
    recent_x = recent["연도"].to_numpy(dtype=float)
    recent_y = recent["연평균기온"].to_numpy(dtype=float)

    recent_slope, recent_intercept = np.polyfit(
        recent_x, recent_y, 1
    )
    recent_change_per_100 = recent_slope * 100


# 5. 기온 변화량을 크게 표시
st.subheader("100년당 기온 변화량")

st.metric(
    "전체 기간의 기온 변화 추세",
    f"{change_per_100:+.2f}℃ / 100년"
)

left, right = st.columns(2)

with left:
    st.markdown("### 전체 기간")
    st.metric(
        "100년당 변화량",
        f"{change_per_100:+.2f}℃"
    )
    st.caption(f"분석 기간: {start_year}~{end_year}년")

with right:
    st.markdown("### 최근 20년")
    if recent_change_per_100 is not None:
        difference = recent_change_per_100 - change_per_100

        st.metric(
            "100년당 변화량",
            f"{recent_change_per_100:+.2f}℃",
            delta=f"{difference:+.2f}℃ (전체 기간 대비)"
        )
        st.caption(
            f"대상 기간: {recent_start}~{end_year}년 "
            f"중 유효한 연도 {len(recent)}개"
        )
    else:
        st.warning("최근 기간의 자료가 부족합니다.")


# 6. 산점도와 전체 기간 회귀 직선
st.subheader("연도별 연평균기온과 회귀 직선")

line_x = np.linspace(start_year, 2100, 300)
line_y = slope * line_x + intercept

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=yearly["연도"],
    y=yearly["연평균기온"],
    mode="markers",
    name="연평균기온",
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
    name="전체 기간 회귀 직선",
    line=dict(width=3),
    hovertemplate=(
        "연도: %{x:.0f}년<br>"
        "예상 기온: %{y:.2f}℃"
        "<extra></extra>"
    )
))

# 최근 20년 회귀 직선도 함께 표시
if recent_slope is not None:
    recent_line_x = np.linspace(
        max(start_year, recent_start), 2100, 200
    )
    recent_line_y = (
        recent_slope * recent_line_x + recent_intercept
    )

    fig.add_trace(go.Scatter(
        x=recent_line_x,
        y=recent_line_y,
        mode="lines",
        name="최근 20년 회귀 직선",
        line=dict(width=3, dash="dash"),
        hovertemplate=(
            "연도: %{x:.0f}년<br>"
            "최근 추세 예상 기온: %{y:.2f}℃"
            "<extra></extra>"
        )
    ))

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    hovermode="closest",
    legend=dict(orientation="h", y=1.10),
    margin=dict(l=20, r=20, t=50, b=20)
)

st.plotly_chart(fig, use_container_width=True)

st.metric(
    "연도와 연평균기온의 상관계수",
    f"{correlation:.4f}"
)

st.write(
    f"전체 기간 회귀식: y = {slope:.5f}x {intercept:+.2f}"
)

if correlation > 0:
    st.write("상관계수는 양수로, 연도와 연평균기온이 함께 증가하는 경향을 보입니다.")
elif correlation < 0:
    st.write("상관계수는 음수로, 연도가 증가할수록 연평균기온이 낮아지는 선형 경향을 보입니다.")
else:
    st.write("상관계수는 0으로, 두 변수 사이의 선형 상관관계가 없습니다.")


# 7. 연도별 예상 기온
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
    f"""
    <div style="text-align:center; padding:20px;">
        <div style="font-size:22px;">
            {selected_year}년 예상 연평균기온
        </div>
        <div style="font-size:56px; font-weight:bold;">
            {predicted_temp:.2f}℃
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(
    "과거 연평균기온의 선형 추세를 연장한 값입니다. "
    "실제 미래 기온과 다를 수 있습니다."
)


# 8. 연도별 자료표
with st.expander("연도별 분석 데이터 확인"):
    display_df = yearly.rename(columns={
        "연평균기온": "연평균기온(℃)",
        "관측일수": "관측일수(일)"
    }).copy()

    display_df["연평균기온(℃)"] = (
        display_df["연평균기온(℃)"].round(2)
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
