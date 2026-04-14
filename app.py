import streamlit as st
import pandas as pd
import fitz
import easyocr
import numpy as np
from google import genai
import json
from PIL import Image
import io
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import re
import calendar

# ==========================================
# Page Config & Custom CSS (Premium Aesthetics)
# ==========================================
st.set_page_config(page_title="일용근로자 평균임금 산정 시스템", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, 'Helvetica Neue', 'Segoe UI', 'Apple SD Gothic Neo', 'Noto Sans KR', 'Malgun Gothic', 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', sans-serif;
}

/* Header gradient */
.main-header {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    color: white;
    padding: 2rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    box-shadow: 0 10px 20px rgba(0,0,0,0.1);
}
.main-header h1 {
    margin: 0;
    font-weight: 800;
    font-size: 2.2rem;
}
.main-header p {
    margin: 10px 0 0 0;
    font-size: 1.1rem;
    opacity: 0.9;
}

/* Metric Cards */
.metric-card {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    margin-bottom: 20px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 15px rgba(0,0,0,0.1);
}
.metric-title {
    font-size: 0.9rem;
    color: #666;
    margin-bottom: 8px;
    font-weight: 600;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #1e3c72;
}

/* Result Boxes */
.result-box {
    padding: 20px;
    border-radius: 12px;
    margin: 20px 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}
.result-box-1 {
    border-left: 6px solid #10b981;
    background-color: #ecfdf5;
}
.result-box-2 {
    border-left: 6px solid #3b82f6;
    background-color: #eff6ff;
}
.result-box-3 {
    border-left: 6px solid #f59e0b;
    background-color: #fffbeb;
}
.result-title {
    font-weight: 800;
    font-size: 1.4rem;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.result-desc {
    font-size: 1.05rem;
    color: #374151;
    line-height: 1.6;
}
.highlight-text {
    font-weight: 700;
    font-size: 1.8rem;
    color: #111827;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# Core Logic Functions
# ==========================================
@st.cache_resource
def load_ocr_reader():
    return easyocr.Reader(['ko', 'en'])

def process_file_with_ocr(file_bytes, file_name):
    """
    업로드된 이미지나 PDF를 텍스트로 인식하고 자체 휴리스틱 알고리즘으로 분석합니다.
    (API Key 제약 문제와 환각(Hallucination) 현상을 피해 로컬 모델로 완전 회귀)
    """
    reader = load_ocr_reader()
    extracted_data = []
    
    images = []
    if file_name.strip().lower().endswith('.pdf'):
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page in doc:
                pix = page.get_pixmap(dpi=150)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                if pix.n == 4:
                    img = img[:, :, :3]
                images.append(img)
        except Exception as e:
            st.error(f"PDF 렌더링 에러: {e}")
            return []
    else:
        try:
            image = Image.open(io.BytesIO(file_bytes)).convert('RGB')
            images.append(np.array(image))
        except Exception as e:
            st.error(f"이미지 인식 에러: {e}")
            return []
            
    def group_by_rows(results):
        boxes = []
        for bbox, text, prob in results:
            ymin = min(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1])
            ymax = max(bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1])
            xmin = min(bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0])
            boxes.append({'ymin': ymin, 'ymax': ymax, 'x': xmin, 'text': text})
            
        boxes.sort(key=lambda b: b['ymin'])
        r_list = []
        for b in boxes:
            placed = False
            for r in r_list:
                row_ymin = sum(x['ymin'] for x in r) / len(r)
                row_ymax = sum(x['ymax'] for x in r) / len(r)
                overlap = max(0, min(b['ymax'], row_ymax) - max(b['ymin'], row_ymin))
                # 최소 30% 높이 겹침 확인 (안정적인 가로줄 획득)
                if overlap > (b['ymax'] - b['ymin']) * 0.3:
                    r.append(b)
                    placed = True
                    break
            if not placed:
                r_list.append([b])
        for r in r_list:
            r.sort(key=lambda b: b['x'])
        return r_list

    parsed_rows = []
    for img in images:
        results = reader.readtext(img, detail=1)
        rows = group_by_rows(results)
        
        for row in rows:
            row_str = " ".join([b['text'] for b in row])
            
            # 정규식을 통한 행 분석 (YYYY-MM)
            date_match = re.search(r'(\d{4}\s*[-\.년/]\s*\d{1,2}\s*월?)', row_str)
            if not date_match: continue
            
            date_str = date_match.group(1).replace(' ', '')
            # 상단 팩스 헤더 오인 방지
            if re.match(r'^\d{4}[-\.년/]\d{1,2}[-\.일]$', date_str) or re.search(r'\d{4}-\d{1,2}-\d{1,2}', date_str): 
                continue
                
            post_date = row_str[date_match.end():]
            tokens = post_date.split()
            wage = 0
            days = 0
            comp_idx = 0
            
            # 우측에서 좌측으로 역순 탐색
            for idx, t in reversed(list(enumerate(tokens))):
                clean_t = re.sub(r'[^\d]', '', t)
                if not clean_t: continue
                val = int(clean_t)
                
                if wage == 0 and val >= 1000:
                    wage = val
                elif wage > 0 and days == 0 and val <= 31:
                    days = val
                    comp_idx = idx
                    break
                    
            if wage > 0 and days > 0:
                raw_company = tokens[:comp_idx]
                
                days_arr = []
                clean_company = []
                
                # 좌측 텍스트들에서 숫자만으로 이루어진 토큰은 근로일자 배열임
                for cp in raw_company:
                    if re.match(r'^[\d,\.\s]+$', cp) or re.match(r'^\d+$', cp):
                        nums = re.findall(r'\d+', cp)
                        for n in nums:
                            if 1 <= int(n) <= 31:
                                days_arr.append(int(n))
                        continue
                        
                    if re.search(r'(종사자|\(\d{3}\)|단순|채굴|생산|관련|건설)', cp): continue
                    res_cp = re.sub(r'[{}[\]\/?;:|*~`!^\-_+<>@\#$%&\\\=\'\"]', '', cp)
                    if len(res_cp.strip()) > 0:
                        clean_company.append(res_cp.strip())
                        
                final_company = " ".join(clean_company) if clean_company else "사업장 인식불가"
                norm_date = re.sub(r'[-\.년]+', '/', date_str).replace('월', '')
                if norm_date.endswith('/'): norm_date = norm_date[:-1]
                
                parsed_rows.append({
                    '근로연월': norm_date,
                    '사업장명': final_company,
                    '근로일자': days_arr,
                    '근로일수': days,
                    '임금총액': wage
                })
                
    extracted_data = []
    if parsed_rows:
        extracted_data.append(pd.DataFrame(parsed_rows))
        
    return extracted_data

def extract_last_daily_wage(raw_dfs, d_date):
    """
    추출된 기록 중에서 재해일(d_date) 이전의 가장 최근 기록의 임금총액 / 근로일수를 역산하여 반환합니다.
    (근로일자 배열 데이터를 활용하여 재해일보다 작은 가장 가까운 일자를 정확히 찾아냅니다.)
    """
    if not raw_dfs:
        return 0, None
        
    df = pd.concat(raw_dfs, ignore_index=True)
    best_row = None
    min_diff = None
    
    for idx, row in df.iterrows():
        ym_str = str(row.get('근로연월', ''))
        match = re.search(r'(\d{4})[^\d]+(\d{1,2})', ym_str)
        if not match:
            continue
            
        year, month = int(match.group(1)), int(match.group(2))
        
        days_arr = row.get('근로일자', [])
        # 근로일자 배열이 없거나 빈 리스트라면 해당 월의 말일을 임시로 잡습니다.
        if not isinstance(days_arr, list) or not days_arr:
            try:
                days_arr = [calendar.monthrange(year, month)[1]]
            except Exception:
                continue
                
        # 해당 표에 존재하는 각각의 근로일자를 모두 순회하며 가장 가까운 날짜를 찾습니다
        for day in days_arr:
            try:
                d = int(day)
                worked_date = date(year, month, d)
            except Exception:
                continue
                
            # 재해일 이전(strictly BEFORE)인 날짜만 골라냅니다
            if worked_date < d_date:
                diff = (d_date - worked_date).days
                if min_diff is None or diff < min_diff:
                    min_diff = diff
                    best_row = row
                    
    # 조건을 만족하는 행이 하나라도 있다면 계산 진행
    if best_row is not None:
        try:
            days = int(re.sub(r'[^\d]', '', str(best_row.get('근로일수', '0'))))
            wage = int(re.sub(r'[^\d]', '', str(best_row.get('임금총액', '0'))))
            if days > 0 and wage > 0:
                return wage / days, str(best_row.get('사업장명', '미상'))
        except Exception:
            pass
            
    return 0, None

def get_latest_working_date_from_dfs(raw_dfs):
    """
    여러 데이터프레임 안에서 근로연월(YYYY-MM) 문자열을 파싱하여,
    가장 마지막으로 일한 날짜(해당 월의 말일)를 추정합니다.
    """
    if not raw_dfs:
        return None
    
    df = pd.concat(raw_dfs, ignore_index=True)
    date_col = None
    for col in df.columns:
        if any(x in col for x in ['연월', '년월', '일자', '날짜']):
            date_col = col
            break
            
    if date_col:
        dates = []
        for val in df[date_col].dropna().astype(str):
            # '2023-05', '2023년 05월', '2023.05.' 등 추출
            match = re.search(r'(\d{4})[-\.년\s]*(\d{1,2})', val)
            if match:
                y = int(match.group(1))
                m = int(match.group(2))
                if 1 <= m <= 12:
                    # 해당 월의 마지막 날짜(말일)로 추정
                    last_day = calendar.monthrange(y, m)[1]
                    dates.append(date(y, m, last_day))
        
        if dates:
            return max(dates)
            
    return None

def clean_extracted_dataframe(df_list):
    """
    여러 테이블 포맷을 하나의 표준 포맷으로 정리합니다.
    """
    if not df_list:
        return pd.DataFrame(columns=['근로연월', '사업장명', '근로일수', '임금총액', '일당'])
    
    raw_df = pd.concat(df_list, ignore_index=True)
    return raw_df

def populate_editor_data(raw_dfs, start_date, end_date):
    """
    raw_dfs에서 start_date와 end_date 기간에 걸치는 데이터를 뽑아 에디터에 자동 채웁니다.
    """
    if not raw_dfs:
        return pd.DataFrame([{"사업장명": "", "근로일수": 0, "임금총액": 0}])
        
    df = pd.concat(raw_dfs, ignore_index=True)
    
    date_col = next((c for c in df.columns if any(x in c for x in ['연월', '년월', '일자', '날짜'])), None)
    company_col = next((c for c in df.columns if '사업장' in c), None)
    days_col = next((c for c in df.columns if '일수' in c), None)
    wage_col = next((c for c in df.columns if '임금' in c), None)
    
    result_rows = []
    
    if date_col and company_col and days_col and wage_col:
        for _, row in df.iterrows():
            val = str(row[date_col])
            match = re.search(r'(\d{4})[-\.년\s]*(\d{1,2})', val)
            if match:
                y = int(match.group(1))
                m = int(match.group(2))
                
                if 1 <= m <= 12:
                    row_date_start = date(y, m, 1)
                    row_date_end = date(y, m, calendar.monthrange(y, m)[1])
                    
                    # 기간이 조금이라도 겹치면 해당 월의 데이터로 자동 산입
                    if row_date_start <= end_date and row_date_end >= start_date:
                        days_str = str(row[days_col])
                        wage_str = str(row[wage_col])
                        
                        try:
                            days_match = re.search(r'\d+', days_str)
                            days = int(days_match.group()) if days_match else 0
                        except:
                            days = 0
                            
                        try:
                            wage = int(re.sub(r'[^\d]', '', wage_str))
                        except:
                            wage = 0
                            
                        c_name = str(row[company_col]).strip() if pd.notna(row[company_col]) else "미상"
                        result_rows.append({
                            "사업장명": c_name,
                            "근로일수": days,
                            "임금총액": wage
                        })
                        
    if not result_rows:
        return pd.DataFrame([{"사업장명": "", "근로일수": 0, "임금총액": 0}])
        
    # 동일 사업장명 합산
    df_res = pd.DataFrame(result_rows)
    df_res = df_res.groupby('사업장명', as_index=False).sum()
    return df_res

def calculate_average_wage(d_date, df_3m, df_1m, last_daily_wage):
    """
    산정 및 분기 로직 (핵심)
    """
    # 1. 날짜 계산 (역산)
    prev_1m_start = d_date - relativedelta(months=1)
    prev_1m_end = d_date - relativedelta(days=1)
    prev_3m_start = d_date - relativedelta(months=3)
    prev_3m_end = d_date - relativedelta(days=1)
    
    days_1m = (prev_1m_end - prev_1m_start).days + 1
    days_3m = (prev_3m_end - prev_3m_start).days + 1
    
    # 2. 데이터 집계
    wage_3m_total = pd.to_numeric(df_3m['임금총액'], errors='coerce').sum()
    days_3m_work = pd.to_numeric(df_3m['근로일수'], errors='coerce').sum()
    companies_3m = df_3m['사업장명'].replace('', pd.NA).dropna().nunique()
    
    wage_1m_total = pd.to_numeric(df_1m['임금총액'], errors='coerce').sum()
    days_1m_work = pd.to_numeric(df_1m['근로일수'], errors='coerce').sum()
    
    # 3. 조건 판별
    # [조건 1] 3개월 이상 동일 사업장 계속 근무
    cond1_met = False
    if companies_3m == 1 and days_3m_work > 0:
        cond1_met = True
        
    # [조건 2] 직전 1개월간 타 사업장 합산 23일 이상 근무
    cond2_met = False
    if days_1m_work >= 23:
        cond2_met = True
        
    # 4. 결과 도출
    result_type = 0
    final_wage = 0.0
    calc_method = ""
    
    if cond1_met:
        result_type = 1
        final_wage = wage_3m_total / days_3m if days_3m > 0 else 0
        calc_method = f"(직전 3개월 총임금 {wage_3m_total:,.0f}원) ÷ (총 역일수 {days_3m}일)"
    elif cond2_met:
        result_type = 2
        avg_wage = wage_1m_total / days_1m if days_1m > 0 else 0
        # 통상적으로 일당과 평균임금 중 높은 금액 적용
        if last_daily_wage > avg_wage:
            final_wage = last_daily_wage
            calc_method = f"직전 사업장 1일 일당 ({last_daily_wage:,.0f}원) 적용 (1개월 평균임금 {avg_wage:,.0f}원보다 유리하여 산정)"
        else:
            final_wage = avg_wage
            calc_method = f"(직전 1개월 총임금 {wage_1m_total:,.0f}원) ÷ (총 역일수 {days_1m}일)"
    else:
        result_type = 3
        final_wage = last_daily_wage * 0.73
        calc_method = f"직전 사업장 1일 일당 ({last_daily_wage:,.0f}원) × 통상근로계수(0.73)"
        
    return {
        "result_type": result_type,
        "final_wage": final_wage,
        "calc_method": calc_method,
        "days_1m": days_1m,
        "days_3m": days_3m,
        "wage_3m_total": wage_3m_total,
        "wage_1m_total": wage_1m_total,
        "prev_1m_start": prev_1m_start,
        "prev_1m_end": prev_1m_end,
        "prev_3m_start": prev_3m_start,
        "prev_3m_end": prev_3m_end,
        "days_1m_work": days_1m_work,
        "companies_3m": companies_3m
    }


# ==========================================
# Main UI Layout
# ==========================================
def main():

    st.markdown("""
        <div class="main-header">
            <h1>재해 일용근로자 평균임금 산정기</h1>
            <p>고용보험 일용근로내역서를 바탕으로 통상근로계수 적용 여부를 판단하고 정확한 1일 평균임금을 산정합니다.</p>
        </div>
    """, unsafe_allow_html=True)

    # 1. 기초 정보 입력 영역
    st.subheader("1. 기본 정보 입력")
    # 완전 자동화 대신 사용자 컨펌용으로 기본값(default_ldw)만 넣어주고 표출은 살려둡니다.
    d_date = st.date_input("재해일 (Disaster Date)", value=date.today())

    # 2. 이미지 및 PDF 업로드 (OCR) 영역
    st.subheader("2. 일용근로내역서 업로드 (사진 또는 스캔본 PDF)")
    uploaded_file = st.file_uploader("이미지(JPG, PNG) 또는 PDF 파일을 업로드해주세요.", type=["pdf", "png", "jpg", "jpeg"])
    
    raw_dfs = []
    max_pdf_date = None
    default_ldw = 0
    last_company = ""
    
    if uploaded_file is not None:
        with st.spinner("로컬 인공지능 문자인식(OCR) 중입니다. 데이터 양에 따라 시간이 소요될 수 있습니다."):
            raw_dfs = process_file_with_ocr(uploaded_file.read(), uploaded_file.name)
            if raw_dfs:
                max_pdf_date = get_latest_working_date_from_dfs(raw_dfs)
                ldw, cname = extract_last_daily_wage(raw_dfs, d_date)
                if ldw > 0:
                    default_ldw = int(ldw)
                    last_company = cname
                    
                st.success(f"데이터 추출 완료! (추론된 직전 일당 ➡️ {last_company} 기준 {default_ldw:,.0f}원)")
                with st.expander("OCR 인공지능이 추출한 텍스트 데이터 (참고용)"):
                    for i, df in enumerate(raw_dfs):
                        st.dataframe(df, use_container_width=True)
            else:
                st.warning("표 데이터를 추출하지 못했습니다. 아래 양식에 직접 수기로 입력해주세요.")

    # 사용자가 최종 확인/수정 할 수 있는 직전 일당 칸 복구
    last_daily_wage = st.number_input(
        "직전 사업장 1일 일당 확정 (원)", 
        min_value=0, 
        value=default_ldw if default_ldw > 0 else 150000, 
        step=10000, 
        help="재해일 기준 가장 가까운 사업장의 임금을 총 역일수로 나눈 1일 일당입니다. OCR 산출이 틀렸다면 수정해주세요."
    )

    # 3. 산정 기준일(Base Date) 결정 (Fallback 로직)
    base_date = d_date
    fallback_applied = False
    
    initial_3m_start = d_date - relativedelta(months=3)
    
    if max_pdf_date and max_pdf_date < initial_3m_start:
        base_date = max_pdf_date
        fallback_applied = True
        st.markdown(f"""
        <div style="background-color: #fef2f2; border-left: 6px solid #ef4444; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05);">
            <div style="color: #ef4444; font-weight: 800; font-size: 1.1rem; margin-bottom: 8px;">
                🚨 공백기 발생에 따른 산정 기준일 자동 변경 안내
            </div>
            <div style="color: #b91c1c; font-size: 1rem; line-height: 1.5;">
                재해일 직전 3개월 기간에 PDF상 근로 내역이 확인되지 않아, 마지막 근로일(<b>{base_date.strftime('%Y-%m-%d')}</b>) 기준으로 산정되었습니다.<br>
                최종 보상 시에는 공단에서 고용노동부 통계에 따른 <b>'평균임금 증감률'</b>이 추가 적용될 수 있습니다.
            </div>
        </div>
        """, unsafe_allow_html=True)

    prev_1m_start = base_date - relativedelta(months=1)
    prev_1m_end = base_date - relativedelta(days=1)
    prev_3m_start = base_date - relativedelta(months=3)
    prev_3m_end = base_date - relativedelta(days=1)

    # 안내 메시지
    st.info(f"**적용되는 역산 산정 기간** (기준일: {base_date.strftime('%Y-%m-%d')})\n\n"
            f"- **직전 3개월:** {prev_3m_start.strftime('%Y-%m-%d')} ~ {prev_3m_end.strftime('%Y-%m-%d')}\n"
            f"- **직전 1개월:** {prev_1m_start.strftime('%Y-%m-%d')} ~ {prev_1m_end.strftime('%Y-%m-%d')}")

    # 4. 근로 내역 자동 맵핑 및 수정 (Fallback)
    st.subheader("3. 산정 기간별 근로내역 최종 확인 및 수정")
    st.write("PDF에서 추출한 내용 중 **해당 기간에 속하는 데이터가 아래 표에 자동으로 채워졌습니다.** 근로일수와 임금이 완벽히 일치하는지 마지막으로 확인 후 필요시 숫자를 올바르게 수정해주세요.")
    
    tab1, tab2 = st.tabs(["직전 3개월 근로내역 (조건1 판별용)", "직전 1개월 근로내역 (조건2 판별용)"])
    
    # 기본 빈 데이터프레임 대신 추출된 데이터로 기간 분리 맵핑
    df_3m_auto = populate_editor_data(raw_dfs, prev_3m_start, prev_3m_end)
    df_1m_auto = populate_editor_data(raw_dfs, prev_1m_start, prev_1m_end)
    
    with tab1:
        st.caption(f"산정기간: {prev_3m_start.strftime('%Y-%m-%d')} ~ {prev_3m_end.strftime('%Y-%m-%d')}")
        df_3m_input = st.data_editor(
            df_3m_auto,
            num_rows="dynamic",
            column_config={
                "사업장명": st.column_config.TextColumn("사업장명", required=True),
                "근로일수": st.column_config.NumberColumn("근로일수 (일)", min_value=0, required=True),
                "임금총액": st.column_config.NumberColumn("임금총액 (원)", min_value=0, required=True),
            },
            key="editor_3m",
            use_container_width=True
        )

    with tab2:
        st.caption(f"산정기간: {prev_1m_start.strftime('%Y-%m-%d')} ~ {prev_1m_end.strftime('%Y-%m-%d')}")
        df_1m_input = st.data_editor(
            df_1m_auto,
            num_rows="dynamic",
            column_config={
                "사업장명": st.column_config.TextColumn("사업장명", required=True),
                "근로일수": st.column_config.NumberColumn("근로일수 (일)", min_value=0, required=True),
                "임금총액": st.column_config.NumberColumn("임금총액 (원)", min_value=0, required=True),
            },
            key="editor_1m",
            use_container_width=True
        )

    # 5. 결과 도출 영역
    if st.button("평균임금 산정하기", type="primary", use_container_width=True):
        res = calculate_average_wage(base_date, df_3m_input, df_1m_input, last_daily_wage)
        
        st.divider()
        st.subheader("📊 결과 대시보드")
        
        # 적용 조건에 따른 박스 표시
        if res["result_type"] == 1:
            st.markdown(f"""
            <div class="result-box result-box-1">
                <div class="result-title">🟢 [조건 1] 3개월 이상 동일 사업장 계속 근무 (통상근로계수 제외)</div>
                <div class="result-desc">
                    직전 3개월 동안 1개의 동일 사업장에서 계속 근무함이 확인되어, <b>통상근로계수를 적용하지 않고 100% 반영</b>하여 평균임금을 산정합니다.<br>
                    (근무 사업장 수: {res["companies_3m"]}개)
                </div>
            </div>
            """, unsafe_allow_html=True)
        elif res["result_type"] == 2:
            st.markdown(f"""
            <div class="result-box result-box-2">
                <div class="result-title">🔵 [조건 2] 직전 1개월간 23일 이상 타 사업장 합산 근무 (통상근로계수 제외)</div>
                <div class="result-desc">
                    직전 1개월 역일 동안 총 <b>{res["days_1m_work"]}일</b>(23일 이상) 근로하였으므로, <b>통상근로계수를 적용하지 않고 100% 반영</b>하여 평균임금을 산정합니다.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box result-box-3">
                <div class="result-title">🟠 [조건 3] 일반 일용근로자 (통상근로계수 적용)</div>
                <div class="result-desc">
                    동일 사업장 3개월 이상 계속 근무 또는 1개월 23일 이상 근로 요건에 해당하지 않습니다.<br>
                    따라서 <b>직전 사업장 1일 일당의 73% (통상근로계수)</b>를 적용하여 평균임금을 산정합니다.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 주요 지표 (Metrics)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">최종 1일 평균임금</div>
                <div class="metric-value">{res['final_wage']:,.0f} 원</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">조건 판별 요약</div>
                <div class="metric-value" style="color:#666; font-size:1.2rem; padding-top:10px;">
                    {'조건 1 (100% 적용)' if res["result_type"] == 1 else '조건 2 (100% 적용)' if res["result_type"] == 2 else '일반 (73% 적용)'}
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">산출 근거</div>
                <div style="font-size: 0.9rem; margin-top:10px; color:#555;">{res["calc_method"]}</div>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
