"""
Streamlit Frontend for Arabic/Darija Fake News Detection
Provides RTL Arabic interface for text analysis
"""

import streamlit as st
import requests
import json
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import arabic_reshaper
import bidi.algorithm

# Configure page
st.set_page_config(
    page_title="🔍 Arabic/Darija Fake News Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for RTL support
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap');

/* RTL and Arabic font support */
.rtl-text {
    direction: rtl;
    text-align: right;
    font-family: 'Tajawal', 'Arial', sans-serif;
    line-height: 1.8;
}

.arabic-font {
    font-family: 'Tajawal', 'Arial', sans-serif;
}

/* Main container */
.main .block-container {
    direction: rtl;
    text-align: right;
    font-family: 'Tajawal', 'Arial', sans-serif;
}

/* Sidebar */
.sidebar .sidebar-content {
    direction: rtl;
    text-align: right;
    font-family: 'Tajawal', 'Arial', sans-serif;
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Tajawal', 'Arial', sans-serif;
    direction: rtl;
    text-align: right;
}

/* Text areas and inputs */
.stTextArea, .stTextInput {
    direction: rtl;
    text-align: right;
    font-family: 'Tajawal', 'Arial', sans-serif;
    font-size: 16px;
}

/* Buttons */
.stButton > button {
    font-family: 'Tajawal', 'Arial', sans-serif;
    font-size: 16px;
    padding: 10px 20px;
}

/* Metrics */
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 10px 0;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.risk-low { background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%); }
.risk-medium { background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%); }
.risk-high { background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); }
.risk-very-high { background: linear-gradient(135deg, #8e44ad 0%, #2c3e50 100%); }

/* Feature importance chart */
.feature-bar {
    margin: 5px 0;
}

.explanation-box {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
    border-right: 4px solid #007bff;
    margin: 20px 0;
}

.language-badge {
    background-color: #6c757d;
    color: white;
    padding: 4px 8px;
    border-radius: 15px;
    font-size: 12px;
    margin: 0 5px;
}

.arabic-lang { background-color: #dc3545; }
.darija-lang { background-color: #fd7e14; }
.french-lang { background-color: #ffc107; color: black; }
.english-lang { background-color: #28a745; }
</style>
""", unsafe_allow_html=True)

# API configuration
API_BASE_URL = "http://localhost:5000"

def reshape_arabic_text(text):
    """Reshape Arabic text for proper display"""
    try:
        reshaped_text = arabic_reshaper.reshape(text)
        return bidi.algorithm.get_display(reshaped_text)
    except:
        return text

def get_risk_color(risk_level):
    """Get color based on risk level"""
    colors = {
        'very_low': '#2ecc71',
        'low': '#27ae60', 
        'medium': '#f39c12',
        'high': '#e74c3c',
        'very_high': '#8e44ad'
    }
    return colors.get(risk_level, '#95a5a6')

def get_risk_emoji(risk_level):
    """Get emoji based on risk level"""
    emojis = {
        'very_low': '✅',
        'low': '🟢',
        'medium': '🟡', 
        'high': '🟠',
        'very_high': '🔴'
    }
    return emojis.get(risk_level, '❓')

def create_risk_gauge(risk_score, risk_level):
    """Create a gauge chart for risk score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "نسبة المخاطرة"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': get_risk_color(risk_level)},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 60], 'color': "gray"},
                {'range': [60, 100], 'color': get_risk_color(risk_level)}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        font={'color': "darkblue", 'family': "Tajawal"}
    )
    
    return fig

def create_feature_importance_chart(feature_importance):
    """Create horizontal bar chart for feature importance"""
    if not feature_importance:
        return None
    
    # Prepare data
    features = [item['feature'] for item in feature_importance[:10]]
    importances = [item['importance'] for item in feature_importance[:10]]
    
    # Reshape Arabic text
    features_display = [reshape_arabic_text(f) for f in features]
    
    fig = go.Figure(go.Bar(
        x=importances,
        y=features_display,
        orientation='h',
        marker_color=[
            '#e74c3c' if imp > 0.7 else '#f39c12' if imp > 0.4 else '#2ecc71'
            for imp in importances
        ]
    ))
    
    fig.update_layout(
        title="أهمية الميزات",
        xaxis_title="الأهمية",
        yaxis_title="الميزة",
        height=400,
        font={'family': 'Tajawal'},
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def create_language_pie_chart(language_distribution):
    """Create pie chart for language distribution"""
    if not language_distribution:
        return None
    
    # Prepare data
    languages = list(language_distribution.keys())
    ratios = list(language_distribution.values())
    
    # Language labels in Arabic
    lang_labels = {
        'arabic': 'العربية الفصحى',
        'darija': 'الدارجة',
        'french': 'الفرنسية',
        'english': 'الإنجليزية',
        'unknown': 'غير معروف'
    }
    
    labels_display = [lang_labels.get(lang, lang) for lang in languages]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels_display,
        values=ratios,
        hole=0.3
    )])
    
    fig.update_layout(
        title="توزيع اللغات",
        font={'family': 'Tajawal'},
        height=300
    )
    
    return fig

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="rtl-text">
        <h1>🔍 Arabic/Darija Fake News Detection</h1>
        <p>نظام متقدم للكشف عن الأخبار الكاذبة باللغة العربية والدارجة</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ الإعدادات")
        
        # Text input
        st.markdown("#### 📝 أدخل النص للتحليل")
        input_text = st.text_area(
            "النص:",
            height=200,
            placeholder="أدخل النص المراد تحليله هنا...",
            key="input_text"
        )
        
        # Analysis options
        st.markdown("#### 🔧 خيارات التحليل")
        include_explanation = st.checkbox(
            "تضمين الشرح المفصل",
            value=True,
            help="قم بتضمين شرح مفصل للقرار"
        )
        
        use_fallback = st.checkbox(
            "استخدام النموذج الاحتياطي",
            value=True,
            help="استخدام نموذج XGBoost كاحتياطي عند فشل AraBERT"
        )
        
        # Analyze button
        analyze_button = st.button(
            "🔍 تحليل النص",
            type="primary",
            use_container_width=True
        )
        
        # Sample texts
        st.markdown("#### 📋 نصوص عينة")
        
        sample_texts = {
            "خبر حقيقي": "أعلنت وزارة الصحة اليوم عن نجاح حملة التطعيم ضد فيروس كورونا في عدة مناطق البلاد.",
            "خبر كاذب": "صدمة: كشف أطباء سر علاجاً سحرياً يقضي على السرطان في أسبوع واحد فقط! لا تصدقوا هذا الخبر الكاذب.",
            "نص دارجة": "كاين واحد كايقول ليك هاد الخبر صحيح، أنا ما عرفتش والو، داباا الزين ماشي مزيان.",
            "نص مختلط": "Breaking news! حادث خطير في الدار البيضاء، according to sources multiples, il y aurait des victimes."
        }
        
        for sample_name, sample_text in sample_texts.items():
            if st.button(sample_name):
                st.session_state.input_text = sample_text
    
    # Main content area
    if analyze_button or input_text:
        if not input_text.strip():
            st.error("⚠️ الرجاء إدخال نص للتحليل")
            return
        
        # Show loading
        with st.spinner("جاري التحليل..."):
            try:
                # Call API
                response = requests.post(
                    f"{API_BASE_URL}/analyze",
                    json={
                        "text": input_text,
                        "include_explanation": include_explanation,
                        "use_fallback": use_fallback
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if result.get('success'):
                        analysis_result = result.get('result', {})
                        display_analysis_results(analysis_result, input_text)
                    else:
                        st.error(f"❌ فشل التحليل: {result.get('error', 'خطأ غير معروف')}")
                        
                else:
                    st.error(f"❌ خطأ في الاتصال بالخادم: {response.status_code}")
                    
            except requests.exceptions.Timeout:
                st.error("⏰ انتهت مهلة الاتصال. الرجاء المحاولة مرة أخرى.")
            except Exception as e:
                st.error(f"❌ حدث خطأ غير متوقع: {str(e)}")
    
    # Footer info
    st.markdown("---")
    st.markdown("""
    <div class="rtl-text">
        <p><strong>🔬 عن النظام:</strong></p>
        <ul>
            <li>يدعم اللغات العربية الفصحى، الدارجة، الفرنسية، والإنجليزية</li>
            <li>يستخدم Haqiqa API مع تحليل متقدم للميزات</li>
            <li>يوفر شرح مفصل للقرارات باستخدام تقنيات LIME-like</li>
            <li>واجهة RTL عربية بالكامل</li>
        </ul>
        
        <p><strong>📊 كيفية الاستخدام:</strong></p>
        <ol>
            <li>أدخل النص المراد تحليله</li>
            <li>اختر خيارات التحليل المطلوبة</li>
            <li>اضغط على زر "تحليل النص"</li>
            <li>استعرض النتائج المفصلة</li>
        </ol>
        
        <p><strong>⚠️ ملاحظات:</strong></p>
        <ul>
            <li>النظام يعتمد على Haqiqa API للتنبؤ بالأخبار الكاذبة</li>
            <li>قد يستغرق التحليل بضع ثوانٍ حسب طول النص</li>
            <li>النتائج هي لأغراض المعلومات فقط ولا تغني عن اليقين المطلق</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def display_analysis_results(result, original_text):
    """Display comprehensive analysis results"""
    
    # Risk analysis section
    st.markdown("## 🎯 نتائج التحليل")
    
    # Risk score and level
    risk_analysis = result.get('risk_analysis', {})
    risk_score = risk_analysis.get('overall_risk_score', 0)
    risk_level = risk_analysis.get('risk_level', 'unknown')
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        risk_color = get_risk_color(risk_level)
        st.markdown(f"""
        <div class="metric-card risk-{risk_level}">
            <h3>{get_risk_emoji(risk_level)} مستوى المخاطرة</h3>
            <h2>{reshape_arabic_text(risk_level.upper())}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 درجة المخاطرة</h3>
            <h2>{risk_score:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        haqiqa_score = risk_analysis.get('haqiqa_score', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>🤖 Haqiqa</h3>
            <h2>{haqiqa_score:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        feature_score = risk_analysis.get('feature_score', 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 الميزات</h3>
            <h2>{feature_score:.3f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Risk gauge
    st.plotly_chart(create_risk_gauge(risk_score, risk_level), use_container_width=True)
    
    # Language analysis
    language_analysis = result.get('language_analysis', {})
    if language_analysis:
        st.markdown("### 🌍 تحليل اللغة")
        
        lang_col1, lang_col2 = st.columns(2)
        
        with lang_col1:
            primary_lang = language_analysis.get('primary_language', 'unknown')
            confidence = language_analysis.get('confidence', 0)
            
            # Language badge
            lang_class = f"{primary_lang}-lang"
            st.markdown(f"""
            <span class="language-badge {lang_class}">
                {reshape_arabic_text(primary_lang.upper())}
            </span>
            <span style="margin-right: 10px;">
                الثقة: {confidence:.1%}
            </span>
            """, unsafe_allow_html=True)
        
        with lang_col2:
            is_code_switched = language_analysis.get('is_code_switched', False)
            st.markdown(f"""
            <div class="explanation-box">
                <h4>🔄 Code-switching:</h4>
                <p>{'نعم' if is_code_switched else 'لا'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Language distribution chart
        lang_dist = language_analysis.get('language_distribution', {})
        if lang_dist:
            st.plotly_chart(create_language_pie_chart(lang_dist), use_container_width=True)
    
    # Feature analysis
    feature_analysis = result.get('feature_analysis', {})
    if feature_analysis:
        st.markdown("### 📊 تحليل الميزات")
        
        # Tabs for different feature types
        feature_tab1, feature_tab2, feature_tab3 = st.tabs(["📝 النص", "😀 المشاعر", "🔍 المعجم"])
        
        with feature_tab1:
            text_features = feature_analysis.get('text_features', {})
            if text_features:
                st.json(text_features)
        
        with feature_tab2:
            sentiment_features = feature_analysis.get('sentiment_features', {})
            if sentiment_features:
                # Sentiment metrics
                sent_col1, sent_col2, sent_col3 = st.columns(3)
                
                with sent_col1:
                    positive_score = sentiment_features.get('positive_score', 0)
                    st.metric("إيجابي", f"{positive_score:.3f}")
                
                with sent_col2:
                    negative_score = sentiment_features.get('negative_score', 0)
                    st.metric("سلبي", f"{negative_score:.3f}")
                
                with sent_col3:
                    subjectivity = sentiment_features.get('sentiment_subjectivity', 0)
                    st.metric("موضوعية", f"{subjectivity:.3f}")
                
                st.json(sentiment_features)
        
        with feature_tab3:
            lexicon_features = feature_analysis.get('lexicon_features', {})
            if lexicon_features:
                # Lexicon risk factors
                risk_factors = lexicon_features.get('overall_fake_news_risk', 0)
                st.metric("مخاطر المعجم", f"{risk_factors:.3f}")
                
                st.json(lexicon_features)
    
    # Haqiqa prediction
    haqiqa_prediction = result.get('haqiqa_prediction', {})
    if haqiqa_prediction:
        st.markdown("### 🤖 تنبؤ Haqiqa")
        
        haqiqa_col1, haqiqa_col2 = st.columns(2)
        
        with haqiqa_col1:
            prediction = haqiqa_prediction.get('prediction', 'Unknown')
            confidence = haqiqa_prediction.get('confidence', 0)
            
            st.markdown(f"""
            <div class="explanation-box">
                <h4>🎯 التنبؤ:</h4>
                <h3>{reshape_arabic_text(prediction)}</h3>
                <p>الثقة: {confidence:.1%}</p>
                <p>النموذج: {haqiqa_prediction.get('model_used', 'Unknown')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with haqiqa_col2:
            if haqiqa_prediction.get('fallback_used'):
                st.warning("⚠️ تم استخدام النموذج الاحتياطي (XGBoost)")
            
            # Probabilities
            real_prob = haqiqa_prediction.get('real_probability', 0)
            fake_prob = haqiqa_prediction.get('fake_probability', 0)
            
            st.markdown(f"""
            <div class="explanation-box">
                <h4>📊 الاحتمالات:</h4>
                <p>حقيقي: {real_prob:.1%}</p>
                <p>كاذب: {fake_prob:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Explanation
    explanation = result.get('explanation')
    if explanation and st.session_state.get('include_explanation', True):
        st.markdown("### 📝 الشرح المفصل")
        
        exp_col1, exp_col2 = st.columns(2)
        
        with exp_col1:
            st.markdown(f"""
            <div class="explanation-box">
                <h4>📋 ملخص الشرح:</h4>
                <p>{reshape_arabic_text(explanation.get('summary', ''))}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with exp_col2:
            key_factors = explanation.get('key_factors', [])
            if key_factors:
                st.markdown("**عوامل المخاطرة الرئيسية:**")
                for i, factor in enumerate(key_factors[:5]):
                    factor_name = factor.get('factor', 'unknown')
                    severity = factor.get('severity', 'unknown')
                    impact = factor.get('impact', 0)
                    
                    st.markdown(f"""
                    <div style="margin: 10px 0; padding: 10px; background-color: #f8f9fa; border-radius: 5px;">
                        <strong>{i+1}. {reshape_arabic_text(factor_name)}</strong><br>
                        <small>الشدة: {severity} | التأثير: {impact:.3f}</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Recommendations
        recommendations = explanation.get('recommendations', [])
        if recommendations:
            st.markdown("**📌 التوصيات:**")
            for rec in recommendations:
                st.markdown(f"- {reshape_arabic_text(rec)}")
    
    # Processing time
    processing_time = result.get('processing_time', 0)
    if processing_time:
        st.markdown(f"⏱️ وقت المعالجة: {processing_time:.2f} ثانية")
    
    # Feature importance chart
    if feature_analysis:
        lexicon_features = feature_analysis.get('lexicon_features', {})
        if lexicon_features:
            # Create feature importance from lexicon
            feature_importance = [
                {'feature': 'clickbait', 'importance': lexicon_features.get('clickbait_score', 0)},
                {'feature': 'عدم اليقين', 'importance': lexicon_features.get('uncertainty_score', 0)},
                {'feature': 'نظريات المؤامرة', 'importance': lexicon_features.get('conspiracy_score', 0)},
                {'feature': 'الدعاية', 'importance': lexicon_features.get('propaganda_score', 0)},
                {'feature': 'المعلومات المضللة', 'importance': lexicon_features.get('unreliable_source_score', 0)}
            ]
            
            feature_chart = create_feature_importance_chart(feature_importance)
            if feature_chart:
                st.plotly_chart(feature_chart, use_container_width=True)

if __name__ == "__main__":
    main()