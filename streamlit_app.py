# streamlit_app.py
# 대학 지원 현황 시각화 + 다중 파일 합산 + '재요청' 행 제거 + GPT 보고서 생성(+다운로드)
# by @ssac9 요청사항 반영

import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
import json

st.set_page_config(page_title="대학 지원 현황 - 통합/보고서", layout="wide")
st.title("대학 지원 현황 (다중 파일·막대그래프·컬러풀 + GPT 보고서)")

st.markdown("""
**사용 안내**  
- 같은 양식의 엑셀 파일을 **여러 개 업로드**하면 **모든 파일을 합산**해 대학(G열)별 지원 빈도 막대그래프를 보여줍니다.  
- 업로드된 파일의 **어느 열에든 '재요청'** 이라는 문구가 포함된 **행은 전부 제외**하고 집계합니다.  
- **그래프 제목(단일 파일 업로드 시)**은 **C, D, B열** 데이터를 조합해 자동 생성됩니다. 예) `2025학년도 3학년 6반 수시 지원 대학 시각화`  
- **보고서 자동 작성**: 아래 **OpenAI API 키**를 입력하면 집계 데이터를 바탕으로 **인서울 → 경기권 → 지방대학** 순서의 분석 보고서를 생성합니다.  
- 공백/결측 값은 `"미기재"`로 처리합니다.  
- 각 대학 막대는 **다채로운 색상 팔레트**로 표시됩니다.  

📂 **엑셀 파일 저장 방법**  
👉 **나이스 > 대입전형 > 제공현황 조회 > 엑셀파일로 저장**
""")

# ----------------------------- 입력 UI -----------------------------
uploaded_files = st.file_uploader(
    "엑셀 파일(.xlsx)을 하나 이상 업로드하세요",
    type=["xlsx"],
    accept_multiple_files=True
)

mapping_file = st.file_uploader(
    "선택: 대학-권역 매핑 CSV 업로드 (열 이름: 대학, 권역 / 권역: 인서울·경기권·지방대학)",
    type=["csv"]
)

with st.expander("GPT 보고서 생성(선택 사항)"):
    st.caption("OpenAI API 키를 입력하면 보고서를 자동 생성합니다. (키는 세션 내에서만 사용)")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    model_name = st.text_input("모델 이름", value="gpt-4o-mini", help="예: gpt-4o-mini, gpt-4o, gpt-4.1 등")
    generate_btn = st.button("보고서 생성")

# ----------------------------- 유틸 함수 -----------------------------
def safe_read_excel(file):
    try:
        df = pd.read_excel(file, dtype=str)
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        return df
    except Exception as e:
        st.error(f"[{getattr(file, 'name', '파일')}] 엑셀을 읽는 중 오류: {e}")
        return None

def default_col_by_letter(df, letter):
    pos = ord(letter.upper()) - ord('A') + 1
    if 1 <= pos <= len(df.columns):
        return df.columns[pos-1]
    return None

def remove_rows_with_keyword(df: pd.DataFrame, keyword: str = "재요청"):
    """어느 열에든 keyword가 포함된 행은 삭제."""
    if df is None or df.empty:
        return df, 0
    # 문자열화 후 포함 여부 체크
    sdf = df.astype(str)
    mask_any = sdf.apply(lambda col: col.str.contains(keyword, na=False)).any(axis=1)
    removed = int(mask_any.sum())
    cleaned = df.loc[~mask_any].copy()
    return cleaned, removed

def build_univ_counts_from_series(series: pd.Series) -> pd.DataFrame:
    s = series.astype(str)
    s = s.replace({"": "미기재", "NaN": "미기재", "nan": "미기재", "None": "미기재"}).fillna("미기재")
    s = s.apply(lambda x: x.strip() if isinstance(x, str) else x)
    vc = s.value_counts(dropna=False)
    out = vc.rename_axis("대학").reset_index(name="지원수")
    out = out.sort_values("지원수", ascending=False, kind="mergesort").reset_index(drop=True)
    return out

def make_title_from_df(df):
    try:
        c_val = str(df.iloc[0, 2]) if df.shape[1] > 2 else ""
        d_val = str(df.iloc[0, 3]) if df.shape[1] > 3 else ""
        b_val = str(df.iloc[0, 1]) if df.shape[1] > 1 else ""
        base = " ".join([v for v in [c_val, d_val, b_val] if v])
        if base.strip():
            return f"{base} 수시 지원 대학 시각화"
    except Exception:
        pass
    return "대학별 지원 빈도 시각화"

def build_region_map_from_csv(file) -> dict:
    try:
        df = pd.read_csv(file, dtype=str)
        df = df.rename(columns={c: c.strip() for c in df.columns})
        assert "대학" in df.columns and "권역" in df.columns
        mp = {}
        for _, row in df.iterrows():
            u = str(row["대학"]).strip()
            r = str(row["권역"]).strip()
            if u and r in ["인서울", "경기권", "지방대학"]:
                mp[u] = r
        return mp
    except Exception as e:
        st.warning(f"매핑 CSV를 읽는 중 문제가 발생했습니다: {e}")
        return {}

# 내장(간이) 매핑: 필요시 CSV로 보완 권장
BUILTIN_REGION_MAP = {
    # 인서울 (대표 예시)
    "서울대": "인서울", "서울대학교": "인서울",
    "연세": "인서울", "고려": "인서울", "한양": "인서울", "성균관": "인서울", "서강": "인서울",
    "중앙": "인서울", "경희": "인서울", "한국외국어": "인서울", "외국어": "인서울", "동국": "인서울",
    "건국": "인서울", "홍익": "인서울", "숙명": "인서울", "이화": "인서울",
    # 경기권 (대표 예시·수도권 포함)
    "아주": "경기권", "경기대": "경기권", "단국": "경기권", "용인": "경기권", "죽전": "경기권",
    "가천": "경기권", "한양대(ERICA)": "경기권", "한경": "경기권", "인천": "경기권", "인하": "경기권",
}

def heuristic_region(univ_name: str) -> str:
    n = (univ_name or "").strip()
    if n == "" or n == "미기재":
        return "지방대학"
    # 내장 키워드/대학명
    for key, region in BUILTIN_REGION_MAP.items():
        if key in n:
            return region
    # 키워드 기반
    if "서울" in n:
        return "인서울"
    if any(k in n for k in ["경기", "수원", "용인", "분당", "성남", "안양", "의정부", "인천", "수도권", "일산", "고양"]):
        return "경기권"
    return "지방대학"

def apply_region(df_counts: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    def _map_one(u):
        if u in mapping:
            return mapping[u]
        for k, v in mapping.items():
            if k and k in u:
                return v
        return heuristic_region(u)
    out = df_counts.copy()
    out["권역"] = out["대학"].apply(_map_one)
    return out

def to_bytes_download(data: str, filename: str, mime: str = "text/markdown"):
    bio = BytesIO()
    bio.write(data.encode("utf-8-sig"))
    bio.seek(0)
    st.download_button(
        label=f"{filename} 다운로드",
        data=bio,
        file_name=filename,
        mime=mime
    )

# ----------------------------- 메인 처리 -----------------------------
if uploaded_files:
    # 1) 첫 파일 로드(컬럼 추정/타이틀 생성용)
    first_df = safe_read_excel(uploaded_files[0])
    if first_df is None or first_df.empty:
        st.warning("첫 번째 파일이 비어 있거나 읽을 수 없습니다.")
        st.stop()

    # '재요청' 제거
    first_df_clean, removed_first = remove_rows_with_keyword(first_df, "재요청")

    default_univ_col = default_col_by_letter(first_df_clean, "G") or first_df_clean.columns[0]
    univ_col = st.selectbox(
        "대학(빈도) 컬럼 선택 (모든 파일에 동일하게 적용)",
        options=list(first_df_clean.columns),
        index=(list(first_df_clean.columns).index(default_univ_col) if default_univ_col in first_df_clean.columns else 0),
        help="보통 G열(7번째 열)이 대학명입니다."
    )

    # 단일/다중 파일에 따른 그래프 제목
    graph_title = make_title_from_df(first_df_clean) if len(uploaded_files) == 1 else "전체(다중 파일) 수시 지원 대학 시각화"

    # 2) 사용자 매핑 로드
    user_map = build_region_map_from_csv(mapping_file) if mapping_file is not None else {}

    # 3) 모든 파일 로드 + '재요청' 제거 + 합산용 시리즈 모음
    all_univ_values = []
    per_file_counts = []
    total_removed = removed_first  # 제거 누적

    for f in uploaded_files:
        df = safe_read_excel(f)
        if df is None or df.empty:
            st.warning(f"비어 있거나 읽을 수 없는 파일이 있습니다: {getattr(f, 'name', '파일')}")
            continue

        # '재요청' 행 제거
        df, removed = remove_rows_with_keyword(df, "재요청")
        total_removed += removed

        if univ_col not in df.columns:
            st.warning(f"선택한 컬럼 '{univ_col}'이 없는 파일이 있습니다: {getattr(f, 'name', '파일')}")
            continue

        s = df[univ_col]
        all_univ_values.append(s)
        per_file_counts.append({
            "file": getattr(f, "name", "파일"),
            "removed_rows": removed,
            "counts": build_univ_counts_from_series(s)
        })

    if total_removed > 0:
        st.info(f"⚠️ '재요청' 문구가 포함된 행 {total_removed}건을 제외하고 집계했습니다.")

    if not all_univ_values:
        st.error("유효한 데이터가 없습니다. 컬럼 선택 또는 파일을 확인해 주세요.")
        st.stop()

    # 4) 합산 집계
    merged_series = pd.concat(all_univ_values, ignore_index=True)
    total_counts = build_univ_counts_from_series(merged_series)

    # 5) 권역 부여 및 권역별 합계
    total_with_region = apply_region(total_counts, user_map)
    region_order = ["인서울", "경기권", "지방대학"]
    region_summary = (
        total_with_region.groupby("권역")["지원수"].sum().reindex(region_order).fillna(0).astype(int).reset_index()
    )

    # 6) 시각화 옵션
    c1, c2 = st.columns([1, 3])
    with c1:
        top_n = st.number_input("상위 N개만 표시 (0=전체)", min_value=0, max_value=int(len(total_counts)), value=min(20, int(len(total_counts))))
    with c2:
        sort_desc = st.checkbox("빈도 내림차순 정렬", value=True)

    plot_df = total_counts.copy()
    if sort_desc:
        plot_df = plot_df.sort_values("지원수", ascending=False, kind="mergesort")
    if top_n and top_n > 0:
        plot_df = plot_df.head(int(top_n))

    palette = px.colors.qualitative.Set3 + px.colors.qualitative.Vivid + px.colors.qualitative.Dark24

    fig = px.bar(
        plot_df,
        x="대학",
        y="지원수",
        color="대학",
        text="지원수",
        title=graph_title,
        color_discrete_sequence=palette
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_tickangle=-45,
        xaxis_title=None,
        yaxis_title=None,
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("전체 합산 표 보기 (대학별)"):
        st.dataframe(total_counts, use_container_width=True)
        st.download_button(
            "대학별 합산 CSV 다운로드",
            data=total_counts.to_csv(index=False).encode("utf-8-sig"),
            file_name="대학별_지원빈도_전체합산.csv",
            mime="text/csv"
        )

    with st.expander("권역별 합계 보기"):
        st.dataframe(region_summary, use_container_width=True)
        st.download_button(
            "권역별 합계 CSV 다운로드",
            data=region_summary.to_csv(index=False).encode("utf-8-sig"),
            file_name="권역별_합계.csv",
            mime="text/csv"
        )

    with st.expander("파일별 집계(검증용)"):
        for item in per_file_counts:
            st.markdown(f"**파일:** {item['file']} (제거된 행: {item['removed_rows']}건)")
            st.dataframe(item["counts"], use_container_width=True)
            st.markdown("---")

    # ----------------------------- GPT 보고서 생성 -----------------------------
    st.subheader("GPT 기반 분석 보고서 (인서울 → 경기권 → 지방대학)")

    # 보고서 생성을 위한 데이터 페이로드(간결 JSON)
    # 각 권역의 상위 대학 TOP 10도 추출
    def top_univs_by_region(df_regioned: pd.DataFrame, region: str, k=10):
        sub = df_regioned[df_regioned["권역"] == region][["대학", "지원수"]].sort_values("지원수", ascending=False)
        return sub.head(k).to_dict(orient="records")

    payload = {
        "total_by_region": region_summary.to_dict(orient="records"),
        "top_univs": {
            "인서울": top_univs_by_region(total_with_region, "인서울"),
            "경기권": top_univs_by_region(total_with_region, "경기권"),
            "지방대학": top_univs_by_region(total_with_region, "지방대학"),
        },
        "overall_top": total_counts.head(20).to_dict(orient="records"),  # 전체 TOP20
    }

    # 프롬프트
    system_prompt = (
        "너는 한국 고등학교 진학부 교사에게 보고서를 작성하는 데이터 분석 비서다. "
        "입력 JSON을 바탕으로 '인서울 → 경기권 → 지방대학' 순으로 지원 현황을 간결하고 명확하게 분석해라. "
        "숫자는 표와 불릿을 적절히 섞고, 의미 있는 인사이트(집중도, 분산도, 상위 대학 클러스터, 특징적인 전반 경향)를 포함하라. "
        "마지막에 '지도·행정 참고사항' 섹션으로 실무 팁을 3~5개 제시하라. "
        "출력 형식은 Markdown으로 작성한다."
    )
    user_prompt = (
        "아래는 집계 데이터다. 이를 바탕으로 인서울, 경기권, 지방대학 순서의 지원 현황 보고서를 한국어 Markdown으로 작성해줘. "
        "가능하면 표(권역별 합계, 권역별 상위 대학 TOP)를 포함해줘.\n\n"
        f"데이터(JSON):\n```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```"
    )

    # 보고서 미리보기 영역
    report_md = st.empty()

    if generate_btn:
        if not api_key:
            st.error("OpenAI API Key를 입력해 주세요.")
        else:
            # OpenAI 호출 (최신/구버전 모두 시도)
            content = None
            error_msg = None
            try:
                # New-style SDK (openai>=1.0)
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2
                )
                content = resp.choices[0].message.content
            except Exception as e_new:
                try:
                    # Legacy fallback
                    import openai
                    openai.api_key = api_key
                    resp = openai.ChatCompletion.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.2
                    )
                    content = resp["choices"][0]["message"]["content"]
                except Exception as e_old:
                    error_msg = f"OpenAI 호출 실패: {e_new} / {e_old}"

            if error_msg:
                st.error(error_msg)
            else:
                report_md.markdown(content)
                # 다운로드(.md, .txt)
                to_bytes_download(content, "지원현황_분석보고서.md", mime="text/markdown")
                to_bytes_download(content, "지원현황_분석보고서.txt", mime="text/plain")
else:
    st.info("엑셀 파일을 1개 이상 업로드하면 전체 합산 결과와 보고서를 생성할 수 있습니다.")
