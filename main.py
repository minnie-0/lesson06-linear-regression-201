# 전체 기간 선형 회귀 계산
x = yearly["연도"].to_numpy(dtype=float)
y = yearly["연평균기온"].to_numpy(dtype=float)

slope, intercept = np.polyfit(x, y, 1)
correlation = np.corrcoef(x, y)[0, 1]

start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())
year_count = len(yearly)

# 최근 20년 자료 추출
recent_start = end_year - 19
recent = yearly[yearly["연도"] >= recent_start].copy()

if len(recent) >= 2:
    recent_slope, recent_intercept = np.polyfit(
        recent["연도"].to_numpy(dtype=float),
        recent["연평균기온"].to_numpy(dtype=float),
        1
    )
    recent_correlation = np.corrcoef(
        recent["연도"].to_numpy(dtype=float),
        recent["연평균기온"].to_numpy(dtype=float)
    )[0, 1]
else:
    recent_slope = None
    recent_intercept = None
    recent_correlation = None

# 100년당 기온 변화량으로 환산
change_per_100_years = slope * 100

if recent_slope is not None:
    recent_change_per_100_years = recent_slope * 100

# 회귀 직선
line_x = np.linspace(start_year, 2100, 300)
line_y = slope * line_x + intercept
