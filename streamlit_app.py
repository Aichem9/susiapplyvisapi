import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
import time

st.set_page_config(page_title="대학 지원 현황 - 다중 파일 합산", layout="wide")
st.title("대입 전형자료 조회 데이터 기반 지원 현황 시각화 (다중 파일·막대그래프·컬러풀)")

st.markdown("""
**사용 안내**  
- 같은 양식의 엑셀 파일을 **여러 개 업로드**하면 **모든 파일을 합산**해 대학(G열)별 지원 빈도 막대그래프를 보여줍니다.  
- 그래프 제목은 **단일 파일 업로드 시** C, D, B열(예: `2025학년도 3학년 6반`)을 조합해 자동 생성됩니다. **여러 파일 업로드 시**엔 `전체(다중 파일)`로 표시합니다.  
- 공백/결측은 `"미기재"`로 처리합니다.  
- **"재요청"이 포함된 행의 데이터는 자동으로 제외**됩니다.
- 각 대학 막대는 **다채로운 색상 팔레트**로 표시됩니다.  
- **GPT API를 통해 지역별 대학 지원 현황 분석 보고서**를 자동 생성합니다.
- 인창고 AIchem 제작 : ssac9@sen.go.kr

📂 **엑셀 파일 저장 방법**  
👉 **나이스 > 대입전형 > 제공현황 조회 > 엑셀파일로 저장**
""")

def validate_api_key(api_key):
    """
    OpenAI API 키의 유효성을 검증하는 함수
    """
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 간단한 API 호출로 키 검증 (모델 목록 조회)
        response = requests.get(
            "https://api.openai.com/v1/models",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            models_data = response.json()
            # GPT 모델들만 필터링
            available_models = []
            for model in models_data.get('data', []):
                model_id = model.get('id', '')
                if 'gpt' in model_id and ('3.5' in model_id or '4' in model_id):
                    available_models.append(model_id)
            
            return {
                "valid": True, 
                "models": available_models[:5],  # 상위 5개만 표시
                "error": None
            }
        elif response.status_code == 401:
            return {
                "valid": False, 
                "models": [],
                "error": "API 키가 유효하지 않습니다. 올바른 키를 입력해주세요."
            }
        elif response.status_code == 429:
            return {
                "valid": False, 
                "models": [],
                "error": "API 사용량 한도가 초과되었습니다. 나중에 다시 시도해주세요."
            }
        else:
            return {
                "valid": False, 
                "models": [],
                "error": f"API 연결 오류 (코드: {response.status_code})"
            }
            
    except requests.exceptions.Timeout:
        return {
            "valid": False, 
            "models": [],
            "error": "API 응답 시간이 초과되었습니다. 인터넷 연결을 확인해주세요."
        }
    except requests.exceptions.RequestException as e:
        return {
            "valid": False, 
            "models": [],
            "error": f"네트워크 오류: {str(e)}"
        }
    except Exception as e:
        return {
            "valid": False, 
            "models": [],
            "error": f"예상치 못한 오류: {str(e)}"
        }

# GPT API 키 입력 및 검증
with st.sidebar:
    st.header("🤖 GPT API 설정")
    api_key = st.text_input(
        "OpenAI API 키를 입력하세요:",
        type="password",
        help="OpenAI API 키가 필요합니다. https://platform.openai.com/api-keys 에서 발급받으세요."
    )
    
    # API 키 검증 버튼
    if api_key:
        if st.button("🔍 API 키 검증", help="입력한 API 키가 유효한지 확인합니다"):
            with st.spinner("API 키를 검증하는 중..."):
                validation_result = validate_api_key(api_key)
                if validation_result["valid"]:
                    st.success(f"✅ API 키가 유효합니다!\n사용 가능한 모델: {', '.join(validation_result['models'])}")
                else:
                    st.error(f"❌ API 키 오류: {validation_result['error']}")
    
    gpt_model = st.selectbox(
        "GPT 모델 선택:",
        ["gpt-4", "gpt-3.5-turbo"],
        index=0,
        help="gpt-4 모델이 더 정확한 분석을 제공합니다."
    )

uploaded_files = st.file_uploader("엑셀 파일(.xlsx)을 하나 이상 업로드하세요", type=["xlsx"], accept_multiple_files=True)

def safe_read_excel(file):
    try:
        df = pd.read_excel(file, dtype=str)
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        return df
    except Exception as e:
        st.error(f"엑셀을 읽는 중 오류: {e}")
        return None

def remove_reapplication_rows(df):
    """
    데이터프레임에서 '재요청'이 포함된 행을 제거하는 함수
    """
    if df is None or df.empty:
        return df
    
    # 모든 셀에서 '재요청' 문자열이 포함된 행을 찾아서 제거
    mask = df.astype(str).apply(lambda x: x.str.contains('재요청', na=False)).any(axis=1)
    removed_count = mask.sum()
    
    if removed_count > 0:
        st.info(f"'재요청'이 포함된 {removed_count}개 행이 제거되었습니다.")
    
    return df[~mask].reset_index(drop=True)

def classify_university_region(university_name):
    """
    대학명을 기반으로 지역을 분류하는 함수
    """
    university_name = str(university_name).strip()
    
    # 인서울 대학들
    seoul_universities = [
        "서울대", "연세대", "고려대", "성균관대", "한양대", "중앙대", "경희대", "한국외국어대", "서강대", "이화여자대",
        "홍익대", "건국대", "동국대", "국민대", "숭실대", "세종대", "광운대", "명지대", "상명대", "서울시립대",
        "덕성여대", "성신여대", "숙명여대", "동덕여대", "서울여대", "한성대", "서경대", "가톨릭대", "총신대",
        "추계예술대", "한국체육대", "서울교육대", "서울과학기술대", "한국예술종합학교", "육군사관학교",
        "서울기독대", "장로회신학대", "감리교신학대", "한일장신대", "협성대", "서울한영대", "서울디지털대"
    ]
    
    # 경기권 대학들  
    gyeonggi_universities = [
        "성균관대", "한양대", "경기대", "아주대", "인하대", "가천대", "단국대", "강남대", "용인대", "수원대",
        "한신대", "평택대", "을지대", "차의과학대", "대진대", "한국산업기술대", "수원과학대", "경인교육대",
        "한경대", "신한대", "서울신학대", "안양대", "루터대", "서정대", "김포대", "여주대"
    ]
    
    # 지역 키워드로 분류
    if any(univ in university_name for univ in seoul_universities) or "서울" in university_name:
        return "인서울"
    elif any(univ in university_name for univ in gyeonggi_universities) or any(region in university_name for region in ["경기", "인천", "수원", "성남", "안양", "부천", "고양", "용인"]):
        return "경기권"
    elif university_name in ["미기재", "", "NaN", "nan", "None"]:
        return "미기재"
    else:
        return "지방대학"

def analyze_data_by_region(total_counts):
    """
    지역별로 데이터를 분석하는 함수
    """
    # 지역 분류 추가
    total_counts_with_region = total_counts.copy()
    total_counts_with_region['지역'] = total_counts_with_region['대학'].apply(classify_university_region)
    
    # 지역별 집계
    region_summary = total_counts_with_region.groupby('지역')['지원수'].agg(['sum', 'count']).reset_index()
    region_summary.columns = ['지역', '총_지원수', '대학_수']
    region_summary['평균_지원수'] = region_summary['총_지원수'] / region_summary['대학_수']
    region_summary = region_summary.sort_values('총_지원수', ascending=False)
    
    return total_counts_with_region, region_summary

def generate_gpt_report(api_key, model, total_counts, region_summary, total_counts_with_region):
    """
    GPT API를 사용해 분석 보고서를 생성하는 함수 (재시도 및 타임아웃 개선)
    """
    max_retries = 3
    timeouts = [60, 90, 120]  # 점진적으로 타임아웃 증가
    
    for attempt in range(max_retries):
        try:
            # 데이터 준비
            total_students = int(total_counts['지원수'].sum())  # int64를 int로 변환
            top_universities = total_counts.head(10)
            
            # 지역별 상세 데이터 (JSON 직렬화 가능하도록 변환)
            region_details = {}
            for region in ['인서울', '경기권', '지방대학']:
                region_data = total_counts_with_region[total_counts_with_region['지역'] == region]
                if not region_data.empty:
                    # pandas 타입을 Python 기본 타입으로 변환
                    region_details[region] = {
                        '총_지원수': int(region_data['지원수'].sum()),
                        '대학_수': int(len(region_data)),
                        '상위_대학': [
                            {
                                '대학': str(row['대학']),
                                '지원수': int(row['지원수']),
                                '지역': str(row['지역'])
                            }
                            for _, row in region_data.head(5).iterrows()
                        ]
                    }
            
            # region_summary도 JSON 직렬화 가능하도록 변환
            region_summary_dict = []
            for _, row in region_summary.iterrows():
                region_summary_dict.append({
                    '지역': str(row['지역']),
                    '총_지원수': int(row['총_지원수']),
                    '대학_수': int(row['대학_수']),
                    '평균_지원수': float(row['평균_지원수'])
                })
            
            # top_universities도 변환
            top_universities_dict = []
            for _, row in top_universities.iterrows():
                top_universities_dict.append({
                    '대학': str(row['대학']),
                    '지원수': int(row['지원수'])
                })
            
            # 더 간단한 프롬프트로 응답 시간 단축
            prompt = f"""
고등학교 대학 지원 현황 데이터를 분석해주세요.

기본 현황: 전체 {total_students}명, {len(total_counts)}개 대학

지역별 현황:
{json.dumps(region_summary_dict, ensure_ascii=False)}

인기 대학 TOP 5:
{json.dumps(top_universities_dict[:5], ensure_ascii=False)}

다음 구조로 간결한 보고서를 작성해주세요:

1. **전체 현황 요약** (3-4줄)
2. **지역별 분석** (각 지역별 2-3줄)
   - 인서울 대학 특징
   - 경기권 대학 특징
   - 지방대학 특징
3. **주요 발견사항** (3-4개 포인트)
4. **진학 지도 제언** (3-4개 실용적 조언)

각 섹션은 간결하게 작성해주세요.
"""

            # GPT API 호출 (개선된 설정)
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "당신은 고등학교 진학 상담 전문가입니다. 간결하고 실용적인 분석 보고서를 작성합니다."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1500,  # 토큰 수 줄임
                "temperature": 0.3
            }
            
            timeout = timeouts[attempt]
            st.info(f"📡 시도 {attempt + 1}/{max_retries}: API 요청 중... (타임아웃: {timeout}초)")
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            elif response.status_code == 429:
                wait_time = 2 ** attempt  # 지수적 백오프
                st.warning(f"⏳ API 사용량 제한. {wait_time}초 후 재시도...")
                time.sleep(wait_time)
                continue
            else:
                if attempt == max_retries - 1:
                    return f"API 오류: {response.status_code} - {response.text}"
                continue
                
        except requests.exceptions.Timeout:
            if attempt == max_retries - 1:
                return f"⏰ API 응답 시간이 초과되었습니다. 네트워크 상태를 확인하거나 잠시 후 다시 시도해주세요."
            st.warning(f"⏰ 타임아웃 발생. 다시 시도 중... ({attempt + 1}/{max_retries})")
            continue
            
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                return f"🌐 네트워크 오류: {str(e)}"
            st.warning(f"🌐 네트워크 오류. 다시 시도 중... ({attempt + 1}/{max_retries})")
            continue
            
        except Exception as e:
            return f"💥 예상치 못한 오류: {str(e)}"
    
    return "❌ 모든 재시도가 실패했습니다. 잠시 후 다시 시도해주세요."

def default_col_by_letter(df, letter):
    pos = ord(letter.upper()) - ord('A') + 1
    if 1 <= pos <= len(df.columns):
        return df.columns[pos-1]
    return None

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

if uploaded_files:
    # 첫 파일로 기본 컬럼 추정
    first_df = safe_read_excel(uploaded_files[0])
    if first_df is None or first_df.empty:
        st.warning("첫 번째 파일이 비어 있거나 읽을 수 없습니다.")
        st.stop()

    # 재요청 행 제거
    first_df = remove_reapplication_rows(first_df)
    
    if first_df.empty:
        st.warning("재요청 행을 제거한 후 데이터가 없습니다.")
        st.stop()

    default_univ_col = default_col_by_letter(first_df, "G") or first_df.columns[0]
    univ_col = st.selectbox(
        "대학(빈도) 컬럼 선택 (모든 파일에 동일하게 적용)",
        options=list(first_df.columns),
        index=(list(first_df.columns).index(default_univ_col) if default_univ_col in first_df.columns else 0),
        help="보통 G열(7번째 열)이 대학명입니다."
    )

    # 단일/다중에 따른 제목
    if len(uploaded_files) == 1:
        graph_title = make_title_from_df(first_df)
    else:
        graph_title = "전체(다중 파일) 수시 지원 대학 시각화"

    # 모든 파일 로드 & 합산
    per_file_counts = []   # 각 파일별 집계 저장 (검증용)
    all_univ_values = []   # 합산용 시리즈 모음
    total_removed_rows = 0  # 전체 제거된 행 수

    for f in uploaded_files:
        df = safe_read_excel(f)
        if df is None or df.empty:
            st.warning(f"비어 있거나 읽을 수 없는 파일이 있습니다: {getattr(f, 'name', '파일')}")
            continue
        
        # 재요청 행 제거
        original_count = len(df)
        df = remove_reapplication_rows(df)
        removed_count = original_count - len(df)
        total_removed_rows += removed_count
        
        if df.empty:
            st.warning(f"재요청 행 제거 후 데이터가 없는 파일: {getattr(f, 'name', '파일')}")
            continue
            
        if univ_col not in df.columns:
            st.warning(f"선택한 컬럼 '{univ_col}'이 없는 파일이 있습니다: {getattr(f, 'name', '파일')}")
            continue

        # 합산을 위해 원시 시리즈만 모으고, 개별 표도 생성
        s = df[univ_col]
        all_univ_values.append(s)
        per_file_counts.append({
            "file": getattr(f, "name", "파일"),
            "counts": build_univ_counts_from_series(s),
            "removed_rows": removed_count
        })

    if not all_univ_values:
        st.error("유효한 데이터가 없습니다. 컬럼 선택 또는 파일을 확인해 주세요.")
        st.stop()

    # 전체 제거된 행 수 표시
    if total_removed_rows > 0:
        st.success(f"총 {total_removed_rows}개의 '재요청' 행이 제거되었습니다.")

    merged_series = pd.concat(all_univ_values, ignore_index=True)
    total_counts = build_univ_counts_from_series(merged_series)

    # 지역별 분석 데이터 생성
    total_counts_with_region, region_summary = analyze_data_by_region(total_counts)

    # 상위 N개 옵션
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

    # 팔레트 (더 컬러풀)
    palette = px.colors.qualitative.Set3 + px.colors.qualitative.Vivid + px.colors.qualitative.Dark24

    # 막대그래프 (전체 합산)
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

    # 지역별 현황 그래프
    st.subheader("📊 지역별 대학 지원 현황")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 지역별 총 지원수 파이차트
        fig_pie = px.pie(
            region_summary,
            values='총_지원수',
            names='지역',
            title="지역별 지원 비율",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # 지역별 대학 수 막대그래프
        fig_bar = px.bar(
            region_summary,
            x='지역',
            y='대학_수',
            title="지역별 대학 수",
            color='지역',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # 지역별 요약 표
    st.subheader("📋 지역별 요약 통계")
    st.dataframe(region_summary, use_container_width=True)

    # GPT 분석 보고서
    st.subheader("🤖 AI 분석 보고서")
    
    if api_key:
        # API 키 상태 표시
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("📊 AI 분석 보고서 생성", type="primary"):
                with st.spinner("GPT가 데이터를 분석하고 보고서를 작성 중입니다..."):
                    # API 키 재검증
                    validation = validate_api_key(api_key)
                    if not validation["valid"]:
                        st.error(f"❌ API 키 오류: {validation['error']}")
                        st.stop()
                    
                    report = generate_gpt_report(api_key, gpt_model, total_counts, region_summary, total_counts_with_region)
                    
                    if report.startswith("API 오류") or report.startswith("보고서 생성 중 오류"):
                        st.error(report)
                    else:
                        st.markdown("### 📄 대학 지원 현황 분석 보고서")
                        st.markdown(report)
                        
                        # 보고서 다운로드 버튼
                        st.download_button(
                            "📝 보고서 텍스트 다운로드",
                            data=report.encode("utf-8"),
                            file_name="대학지원현황_분석보고서.txt",
                            mime="text/plain"
                        )
        with col2:
            st.info("💡 팁: 먼저 API 키를 검증해보세요!")
    else:
        st.warning("⚠️ GPT API 키를 입력하시면 AI 분석 보고서를 생성할 수 있습니다.")
        st.info("🔗 API 키 발급: https://platform.openai.com/api-keys")

    # 전체 합산 표 & 다운로드
    with st.expander("전체 합산 표 보기"):
        st.dataframe(total_counts_with_region, use_container_width=True)

    st.download_button(
        "전체 합산 CSV 다운로드 (지역분류 포함)",
        data=total_counts_with_region.to_csv(index=False).encode("utf-8-sig"),
        file_name="대학별_지원빈도_전체합산_지역분류.csv",
        mime="text/csv"
    )

    # (선택) 파일별 집계도 확인
    with st.expander("파일별 집계 표 보기"):
        for item in per_file_counts:
            st.markdown(f"**파일:** {item['file']} (재요청 제거: {item['removed_rows']}개 행)")
            st.dataframe(item["counts"], use_container_width=True)
            st.markdown("---")
else:
    st.info("엑셀 파일을 1개 이상 업로드하면 전체 합산 결과를 볼 수 있습니다.")
