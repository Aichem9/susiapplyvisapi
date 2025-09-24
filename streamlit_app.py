
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="대학 지원 현황 시각화", layout="wide")

st.title("대학 지원 현황 시각화 (업로드 즉시 분석)")

st.markdown("""
엑셀 파일을 업로드하면 **G열(대학)**의 빈도와 **J열(전형정보)**를 자동 분류해서 시각화합니다.  
구분 규칙:
- `종합`, `면접`, `탐구`, `우수자` 중 하나라도 포함 → **종합전형**
- 위 네 단어가 없고 `논술` 포함 → **논술**
- 위 네 단어가 없고 `학생부` 포함 → **교과전형**
- 그 외 → **기타/미분류**
""")

uploaded = st.file_uploader("엑셀 파일(.xlsx)을 업로드하세요", type=["xlsx"])

def safe_read_excel(file):
    # 여러 헤더/시트 상황 대비: 첫 시트만 사용, dtype=str로 읽어들이고 공백 제거
    try:
        df = pd.read_excel(file, dtype=str)
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        return df
    except Exception as e:
        st.error(f"엑셀을 읽는 중 오류가 발생했습니다: {e}")
        return None

def default_col_by_letter(df, letter):
    """엑셀 글자열 기준으로 1-indexed 열을 반환(G=7, J=10). 범위를 벗어나면 None."""
    letter = letter.upper()
    # 간단히 A=1, B=2 ... Z=26만 지원
    pos = ord(letter) - ord('A') + 1
    if 1 <= pos <= len(df.columns):
        return df.columns[pos-1]
    return None

def classify_track(text):
    if not isinstance(text, str):
        return "기타/미분류"
    t = text.replace(" ", "")
    # 우선순위: 종합 키워드 -> 논술 -> 학생부 -> 기타
    comprehensive_keys = ["종합", "면접", "탐구", "우수자"]
    if any(k in t for k in comprehensive_keys):
        return "종합전형"
    if "논술" in t:
        return "논술"
    if "학생부" in t:
        return "교과전형"
    return "기타/미분류"

if uploaded is not None:
    df = safe_read_excel(uploaded)
    if df is not None and not df.empty:
        # 기본 선택: G열(대학), J열(전형정보)
        default_univ_col = default_col_by_letter(df, "G")
        default_info_col = default_col_by_letter(df, "J")

        c1, c2 = st.columns(2)
        with c1:
            univ_col = st.selectbox(
                "대학(빈도) 컬럼 선택",
                options=list(df.columns),
                index=(list(df.columns).index(default_univ_col) if default_univ_col in df.columns else 0),
                help="보통 G열(7번째 열)이 대학명입니다."
            )
        with c2:
            info_col = st.selectbox(
                "전형정보(분류) 컬럼 선택",
                options=list(df.columns),
                index=(list(df.columns).index(default_info_col) if default_info_col in df.columns else 0),
                help="보통 J열(10번째 열)입니다."
            )

        st.subheader("데이터 미리보기")
        st.dataframe(df[[univ_col, info_col]].head(20))

        # 1) 대학 빈도 시각화
        st.markdown("### 대학별 지원 빈도")
        univ_counts = (
            df[univ_col]
            .fillna("미기재")
            .replace("", "미기재")
            .value_counts()
            .reset_index()
            .rename(columns={"index": "대학", univ_col: "지원수"})
        )

        tab_bar, tab_table = st.tabs(["막대그래프", "표"])
        with tab_bar:
            fig = px.bar(
                univ_counts,
                x="대학",
                y="지원수",
                title="대학별 지원 빈도",
                text="지원수"
            )
            fig.update_layout(xaxis_tickangle=-45, xaxis_title=None, yaxis_title=None, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with tab_table:
            st.dataframe(univ_counts)

        st.divider()

        # 2) 전형 분류 및 시각화
        st.markdown("### 전형 구분(규칙 기반)")
        df["_전형구분"] = df[info_col].apply(classify_track)
        type_counts = df["_전형구분"].value_counts().reindex(["종합전형", "교과전형", "논술", "기타/미분류"]).fillna(0).astype(int).reset_index()
        type_counts.columns = ["전형", "지원수"]

        tab_pie, tab_bar2, tab_table2 = st.tabs(["원형그래프", "막대그래프", "표"])
        with tab_pie:
            fig2 = px.pie(type_counts, names="전형", values="지원수", title="전형 구분 비율")
            st.plotly_chart(fig2, use_container_width=True)
        with tab_bar2:
            fig3 = px.bar(type_counts, x="전형", y="지원수", text="지원수", title="전형 구분 빈도")
            fig3.update_layout(xaxis_title=None, yaxis_title=None, margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig3, use_container_width=True)
        with tab_table2:
            st.dataframe(type_counts)

        # 3) 필터링 테이블(선택)
        st.markdown("### 전형별 상세 보기")
        chosen = st.multiselect("전형 선택", options=["종합전형", "교과전형", "논술", "기타/미분류"], default=["종합전형", "교과전형", "논술"])
        filtered = df[df["_전형구분"].isin(chosen)][[univ_col, info_col, "_전형구분"]].reset_index(drop=True)
        st.dataframe(filtered, use_container_width=True)

        # 4) 분류 결과 저장
        st.download_button(
            "분류 결과 엑셀 다운로드",
            data=filtered.to_excel(index=False, engine="openpyxl"),
            file_name="전형분류_결과.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.info("파일이 비어 있거나 읽을 수 없습니다. 엑셀 구조를 확인해 주세요.")
else:
    st.info("오른쪽 상단의 버튼으로 예시를 업로드하면 바로 결과를 볼 수 있어요.")
