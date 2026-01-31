import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS 
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .result-card {
        background-color: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.1);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        transition: transform 0.2s ease;
    }
    
    .warning-box {
        padding: 1rem;
        background-color: rgba(255, 193, 7, 0.1); 
        border-left: 4px solid #ffc107;
        border-radius: 8px;
        margin: 1rem 0;
    }

    /* Step Indicator */
    .step-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 2rem;
    }
    .step-item {
        flex: 1;
        text-align: center;
        font-size: 0.85rem;
        font-weight: 600;
        opacity: 0.6;
    }
    .step-item.active { opacity: 1; color: #3498db; font-weight: 700; }
    .step-item.completed { color: #2ecc71; opacity: 1; }
    
    </style>
""", unsafe_allow_html=True)

# Load model 
# --- GANTI BAGIAN INI DI app2.py ---

@st.cache_resource
def load_artifacts():
    # 1. Definisikan variabel dengan nilai awal None (PENTING AGAR TIDAK ERROR VARIABLE NOT FOUND)
    model = None
    scaler = None
    feature_names = None
    scaled_features_list = None

    try:
        # 2. Load artifacts utama
        model = joblib.load("logreg_model.pkl")
        scaler = joblib.load("scaler.pkl")
        feature_names = joblib.load("feature_names.pkl")
        
        # 3. Load artifacts tambahan (dengan try-except tersendiri)
        try:
            scaled_features_list = joblib.load("scaled_features_list.pkl")
        except:
            # Jika file list hilang/rusak, gunakan default hardcoded
            scaled_features_list = ["BMI", "MentHlth", "PhysHlth", "Age", "Education", "Income", "GenHlth"]
            
    except Exception as e:
        # 4. Tampilkan pesan error ASLI jika gagal load (misal karena beda versi)
        st.error(f"⚠️ Terjadi kesalahan saat memuat model: {e}")
        return None, None, None, None

    return model, scaler, feature_names, scaled_features_list

# Load dataset dashboard
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
    except Exception as e:
        print(f"Error membaca file: {e}")
        return None

model, scaler, feature_names, scaled_features_list = load_model()
df = load_dataset()

#fungsi pembantu
def map_age_to_ageg5yr(age):
    if age < 25: return 1
    elif age < 30: return 2
    elif age < 35: return 3
    elif age < 40: return 4
    elif age < 45: return 5
    elif age < 50: return 6
    elif age < 55: return 7
    elif age < 60: return 8
    elif age < 65: return 9
    elif age < 70: return 10
    elif age < 75: return 11
    elif age < 80: return 12
    else: return 13

def map_education(label):
    mapping = {"SD": 2, "SMP": 3, "SMA": 4, "D3/S1": 5, "S2/S3": 6, "Tidak Sekolah/SD": 2, "Sarjana+": 6}
    return mapping.get(label, 4)

def map_income_rp(rp):
    if rp < 15000000: return 1
    elif rp < 25000000: return 2
    elif rp < 35000000: return 3
    elif rp < 50000000: return 4
    elif rp < 75000000: return 5
    elif rp < 100000000: return 6
    elif rp < 150000000: return 7
    else: return 8

def get_risk_category(prob):
    # Threshold disesuaikan agar lebih sensitif
    if prob < 0.30: return "Sangat Rendah", "green", "😊"
    elif prob < 0.50: return "Rendah", "lightgreen", "🙂"
    elif prob < 0.70: return "Sedang", "orange", "😐"
    elif prob < 0.85: return "Tinggi", "darkorange", "😕"
    else: return "Sangat Tinggi", "red", "😞"

# Session State 
if 'page' not in st.session_state: st.session_state.page = 'dashboard'
if 'current_step' not in st.session_state: st.session_state.current_step = 1
if 'show_prediction' not in st.session_state: st.session_state.show_prediction = False

if 'form_data' not in st.session_state:
    st.session_state.form_data = {
        'age': 30, 'sex': 'Perempuan', 'weight': 60.0, 'height': 165.0, 'bmi': 22.0,
        'HighBP': False, 'HighChol': False, 'CholCheck': False, 'Smoker': False,
        'Stroke': False, 'HeartDiseaseorAttack': False, 'PhysActivity': False,
        'DiffWalk': False, 'Fruits': False, 'Veggies': False, 'HvyAlcoholConsump': False,
        'MentHlth': 0, 'PhysHlth': 0, 'GenHlth': 2, 'education': 'SMA',
        'income': 30000000, 'AnyHealthcare': True, 'NoDocbcCost': False
    }

def next_step(): st.session_state.current_step += 1
def prev_step(): st.session_state.current_step -= 1
def reset_form():
    st.session_state.current_step = 1
    st.session_state.show_prediction = False
    # Reset values to default
    st.session_state.form_data = {
        'age': 30, 'sex': 'Perempuan', 'weight': 60.0, 'height': 165.0, 'bmi': 22.0,
        'HighBP': False, 'HighChol': False, 'CholCheck': False, 'Smoker': False,
        'Stroke': False, 'HeartDiseaseorAttack': False, 'PhysActivity': False,
        'DiffWalk': False, 'Fruits': False, 'Veggies': False, 'HvyAlcoholConsump': False,
        'MentHlth': 0, 'PhysHlth': 0, 'GenHlth': 2, 'education': 'SMA',
        'income': 30000000, 'AnyHealthcare': True, 'NoDocbcCost': False
    }

def go_to_prediction(): st.session_state.page = 'prediction'; st.session_state.current_step = 1
def go_to_dashboard(): st.session_state.page = 'dashboard'
def go_to_limitations(): st.session_state.page = 'limitations'

# SIDEBAR 
with st.sidebar:
    st.title("🧭 Navigasi")
    if st.button("📊 Dashboard", use_container_width=True, type="primary" if st.session_state.page == 'dashboard' else "secondary"):
        go_to_dashboard(); st.rerun()
    if st.button("🩺 Prediksi Risiko", use_container_width=True, type="primary" if st.session_state.page == 'prediction' else "secondary"):
        go_to_prediction(); st.rerun()
    if st.button("⚠️ Keterbatasan", use_container_width=True, type="primary" if st.session_state.page == 'limitations' else "secondary"):
        go_to_limitations(); st.rerun()
    st.title("ℹ️ Tentang Aplikasi")
    
    st.write("""
        Aplikasi ini memprediksi **risiko diabetes** berdasarkan:
        - Kondisi kesehatan
        - Kebiasaan hidup
        - Faktor sosial
        
        Model dilatih menggunakan data survei kesehatan **BRFSS 2015 (Amerika Serikat)** dengan **70,692 responden**.
    """)
    
    st.warning("""
        ⚠️ **PENTING**: 
        - Model berbasis data populasi AS
        - Hasil bersifat **estimasi kasar**
        - **BUKAN** diagnosis medis
        - Baca tab "Keterbatasan" sebelum menggunakan
    """)
    
    if st.session_state.page == 'prediction' and st.session_state.current_step <= 4:
        st.markdown("---")
        st.markdown("### 📍 Progress")
        step_names = ["Data Pribadi", "Riwayat Kesehatan", "Gaya Hidup", "Faktor Sosial"]
        for i, name in enumerate(step_names, 1):
            if i < st.session_state.current_step: st.success(f"✅ {name}")
            elif i == st.session_state.current_step: st.info(f"▶️ {name}")
            else: st.write(f"⚪ {name}")
        
        if st.button("🔄 Reset Form", use_container_width=True):
            reset_form(); st.rerun()

#  CONTENT UTAMA

if st.session_state.page == 'limitations':
    st.markdown('<p class="main-header">⚠️ Keterbatasan & Disclaimer</p>', unsafe_allow_html=True)
    st.markdown("""
    ## 🎯 Tujuan Aplikasi
    
    Aplikasi ini dibuat untuk **tujuan edukasi dan demonstrasi** kemampuan machine learning dalam analisis risiko kesehatan.
    
    Aplikasi ini **BUKAN** alat diagnostik medis dan **TIDAK BOLEH** digunakan sebagai pengganti konsultasi medis profesional.
    """)
    

    st.markdown("""
     📞 Yang Harus Anda Lakukan
    
    **Jika hasil menunjukkan risiko tinggi:**
    1. 🏥 **Konsultasi dengan dokter** - ini yang paling penting!
    2. 🩸 **Lakukan tes gula darah** yang proper (HbA1c, GDP, GD2PP)
    3. 💪 **Mulai perbaiki gaya hidup** - tapi dengan bimbingan profesional
    4. 📊 **Monitor kesehatan secara rutin**
    
    **Jika hasil menunjukkan risiko rendah:**
    1. ✅ **Tetap jaga pola hidup sehat**
    2. 🏃 **Rutin olahraga dan makan bergizi**
    3. 🏥 **Medical check-up rutin tetap penting**
    4. ⚠️ **Jangan jadikan ini alasan untuk lengah**
    """)
    if st.button("Saya Mengerti - Lanjut ke Prediksi", type="primary"):
        go_to_prediction(); st.rerun()

elif st.session_state.page == 'dashboard':
    st.title("📊 Dashboard Kesehatan")
    st.caption("Analisis Data BRFSS 2015 (70,692 Responden)")
    
    if df is not None:
        # KEY METRICS
        col1, col2, col3, col4 = st.columns(4)
        
        total_responden = len(df)
        diabetes_count = df['Diabetes_binary'].sum()
        avg_bmi = df['BMI'].mean()
        high_bp_rate = (df['HighBP'].sum() / total_responden) * 100
        
        with col1:
            st.metric("Total Responden", f"{total_responden:,}")
        with col2:
            st.metric("Data Diabetes", f"{int(diabetes_count):,}", help="Jumlah sampel positif di dataset")
        with col3:
            st.metric("Rata-rata BMI", f"{avg_bmi:.1f}", delta="Overweight" if avg_bmi >= 25 else "Normal", delta_color="inverse")
        with col4:
            st.metric("Hipertensi", f"{high_bp_rate:.1f}%", help="Persentase responden dengan darah tinggi")

    

        # CHART UTAMA ---
        col_left, col_right = st.columns([1, 2])
        color_map = {0: '#2a9d8f', 1: '#e76f51'}
        
        with col_left:
            # Donut Chart
            diabetes_dist = df['Diabetes_binary'].value_counts()
            fig_diabetes = go.Figure(data=[go.Pie(
                labels=['Sehat', 'Diabetes'],
                values=[diabetes_dist[0], diabetes_dist[1]],
                hole=0.6,
                marker=dict(colors=['#2a9d8f', '#e76f51']),
                textinfo='percent',
                hoverinfo='label+value'
            )])
            fig_diabetes.update_layout(
                title=dict(text="Komposisi Dataset", font=dict(size=14)),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                margin=dict(t=40, b=0, l=0, r=0),
                height=300,
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_diabetes, use_container_width=True)
        
        with col_right:
            # Horizontal Bar Chart
            risk_data = {
                'Tekanan Darah Tinggi': (df['HighBP'].sum() / total_responden) * 100,
                'Kolesterol Tinggi': (df['HighChol'].sum() / total_responden) * 100,
                'Perokok': (df['Smoker'].sum() / total_responden) * 100,
                'Aktivitas Fisik Kurang': ((total_responden - df['PhysActivity'].sum()) / total_responden) * 100,
                'Obesitas (BMI>30)': (len(df[df['BMI'] >= 30]) / total_responden) * 100
            }
            # Sort data
            risk_sorted = dict(sorted(risk_data.items(), key=lambda item: item[1]))
            
            fig_risk = go.Figure(data=[go.Bar(
                x=list(risk_sorted.values()),
                y=list(risk_sorted.keys()),
                orientation='h',
                marker=dict(
                    color='#457b9d', 
                    opacity=0.9,
                    line=dict(width=0)
                ),
                text=[f"{v:.1f}%" for v in risk_sorted.values()],
                textposition='outside', 
            )])
            fig_risk.update_layout(
                title=dict(text="Faktor Risiko Dominan (%)", font=dict(size=14)),
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False),
                margin=dict(t=40, b=0, l=0, r=0),
                height=300,
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_risk, use_container_width=True)

    

        # AGE & BMI DISTRIBUTION 
        st.subheader("Tren Usia & BMI")
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            fig_bmi = go.Figure()
            fig_bmi.add_trace(go.Histogram(
                x=df[df['Diabetes_binary'] == 0]['BMI'],
                name='Sehat',
                marker_color='#2a9d8f', opacity=0.6
            ))
            fig_bmi.add_trace(go.Histogram(
                x=df[df['Diabetes_binary'] == 1]['BMI'],
                name='Diabetes',
                marker_color='#e76f51', opacity=0.6
            ))
            fig_bmi.update_layout(
                title="Distribusi BMI (Berat Badan)",
                barmode='overlay',
                xaxis_title="Skor BMI",
                yaxis_title="Jumlah Orang",
                yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(x=0.75, y=0.9),
                height=380
            )
            st.plotly_chart(fig_bmi, use_container_width=True)
            # Insight BMI
            st.info("ℹ️ **Insight:** Orang dengan BMI di atas 30 (Obesitas) memiliki populasi penderita diabetes (warna oranye) yang jauh lebih tebal dibanding BMI normal.")

        with col_chart2:
            # GRAFIK USIA-
            
            # Dictionary untuk menerjemahkan kode 1-13 ke Rentang Usia
            age_map = {
                1: "18-24 Thn", 2: "25-29 Thn", 3: "30-34 Thn", 4: "35-39 Thn", 5: "40-44 Thn",
                6: "45-49 Thn", 7: "50-54 Thn", 8: "55-59 Thn", 9: "60-64 Thn", 10: "65-69 Thn",
                11: "70-74 Thn", 12: "75-79 Thn", 13: "80+ Thn"
            }
            
            # Hitung Rata-rata Risiko per Kategori Usia
            age_risk = df.groupby('Age')['Diabetes_binary'].mean() * 100
            
            # Buat List Label agar urut sesuai index 1-13
            age_labels = [age_map.get(x, str(x)) for x in age_risk.index]
            
            fig_age = go.Figure()
            fig_age.add_trace(go.Scatter(
                x=age_labels,    
                y=age_risk.values,
                mode='lines+markers',
                line=dict(color='#e76f51', width=4, shape='spline'), 
                marker=dict(size=10, color='white', line=dict(width=2, color='#e76f51')),
                hovertemplate='<b>Usia: %{x}</b><br>Risiko Rata-rata: %{y:.1f}%<extra></extra>' 
            ))
            fig_age.update_layout(
                title="Peningkatan Risiko Berdasarkan Usia",
                xaxis_title="Kelompok Usia",
                yaxis_title="Persentase Penderita Diabetes (%)",
                yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
                xaxis=dict(tickangle=-45),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=380
            )
            st.plotly_chart(fig_age, use_container_width=True)
            max_risk_age = age_labels[age_risk.argmax()]
            st.markdown(f"""
            <div style="background-color: rgba(231, 111, 81, 0.1); padding: 10px; border-radius: 8px; border-left: 4px solid #e76f51;">
                <small><strong>📈 Analisis Tren:</strong> Grafik menunjukkan korelasi positif yang kuat antara usia dan diabetes. 
                Risiko mulai menanjak signifikan setelah usia <b>40-44 Tahun</b> dan mencapai puncaknya pada kelompok usia <b>{max_risk_age}</b>.</small>
            </div>
            """, unsafe_allow_html=True)

        # GAYA HIDUP & FAKTOR KESEHATAN
        st.subheader("🏃🏼‍♂️ Gaya Hidup & Kesehatan Mental")
        st.caption("Analisis hubungan antara kebiasaan sehari-hari, kesehatan mental, dan risiko diabetes.")

        col_life1, col_life2 = st.columns([1, 1])

        with col_life1:
            # GenHlth: 1 (Excellent) -> 5 (Poor) Mapping Label agar mudah dibaca
            health_labels = {1: "Sangat Baik", 2: "Baik Sekali", 3: "Baik", 4: "Cukup", 5: "Buruk"}
            df['GenHlth_Label'] = df['GenHlth'].map(health_labels)
            # Hitung proporsi
            health_risk = df.groupby('GenHlth_Label')['Diabetes_binary'].mean().reset_index()
            # Sort  (Sangat Baik -> Buruk)
            sorter = ["Sangat Baik", "Baik Sekali", "Baik", "Cukup", "Buruk"]
            health_risk['GenHlth_Label'] = pd.Categorical(health_risk['GenHlth_Label'], categories=sorter, ordered=True)
            health_risk = health_risk.sort_values('GenHlth_Label')

            fig_genhlth = go.Figure()
            fig_genhlth.add_trace(go.Bar(
                x=health_risk['GenHlth_Label'],
                y=health_risk['Diabetes_binary'] * 100,
                marker_color=['#2a9d8f', '#2a9d8f', '#e9c46a', '#f4a261', '#e76f51'],
                text=[f"{v:.1f}%" for v in health_risk['Diabetes_binary'] * 100],
                textposition='auto',
            ))
            fig_genhlth.update_layout(
                title="Persentase Diabetes berdasarkan Persepsi Kesehatan",
                xaxis_title="Persepsi Kesehatan Diri Sendiri",
                yaxis_title="% Penderita Diabetes",
                yaxis=dict(showgrid=False),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=350
            )
            st.plotly_chart(fig_genhlth, use_container_width=True)
            
            st.info("🧠 **Fakta Menarik:** Responden yang merasa kesehatannya 'Buruk' memiliki prevalensi diabetes yang sangat tinggi (>40%), menunjukkan kesadaran diri yang kuat akan kondisi tubuh.")

        with col_life2:
            # 2. Dampak Kebiasaan Buruk vs Baik
            habits = {
                'Perokok Aktif': df[df['Smoker'] == 1]['Diabetes_binary'].mean(),
                'Bukan Perokok': df[df['Smoker'] == 0]['Diabetes_binary'].mean(),
                'Aktif Olahraga': df[df['PhysActivity'] == 1]['Diabetes_binary'].mean(),
                'Jarang Olahraga': df[df['PhysActivity'] == 0]['Diabetes_binary'].mean(),
                'Makan Sayur Tiap Hari': df[df['Veggies'] == 1]['Diabetes_binary'].mean(),
                'Jarang Makan Sayur': df[df['Veggies'] == 0]['Diabetes_binary'].mean()
            }
            
            # Ubah ke Dataframe untuk plotting
            habit_df = pd.DataFrame(list(habits.items()), columns=['Kebiasaan', 'Risiko'])
            habit_df['Risiko'] = habit_df['Risiko'] * 100
            habit_df['Color'] = ['#e76f51', '#2a9d8f', '#2a9d8f', '#e76f51', '#2a9d8f', '#e76f51'] # Merah utk bad habit, Hijau utk good

            fig_habit = go.Figure(go.Bar(
                x=habit_df['Risiko'],
                y=habit_df['Kebiasaan'],
                orientation='h',
                marker_color=habit_df['Color'],
                text=[f"{v:.1f}%" for v in habit_df['Risiko']],
                textposition='inside',
                insidetextanchor='middle'
            ))
            
            fig_habit.update_layout(
                title="Perbandingan Risiko: Gaya Hidup Sehat vs Tidak",
                xaxis_title="Persentase Penderita Diabetes (%)",
                xaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.2)'),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=350,
                margin=dict(l=10)
            )
            st.plotly_chart(fig_habit, use_container_width=True)
            
            st.info("🏃‍♂️ **Insight:** Kurang olahraga (Sedentary Lifestyle) menjadi faktor risiko gaya hidup terbesar, bahkan dampaknya terlihat lebih signifikan dibandingkan pola makan sayur.")


        
        #  MENTAL & PHYSICAL HEALTH DAYS
        st.subheader("📅 Kualitas Hidup Bulanan")
        # Rata-rata hari sakit fisik & mental dalam 30 hari terakhir
        avg_health_days = df.groupby('Diabetes_binary')[['PhysHlth', 'MentHlth']].mean().reset_index()
        fig_days = go.Figure()
        # Bar Kesehatan Fisik
        fig_days.add_trace(go.Bar(
            name='Hari Sakit Fisik',
            x=['Non-Diabetes', 'Diabetes'],
            y=avg_health_days['PhysHlth'],
            marker_color='#457b9d',
            text=[f"{v:.1f} Hari" for v in avg_health_days['PhysHlth']],
            textposition='auto'
        ))
        
        # Bar Kesehatan Mental
        fig_days.add_trace(go.Bar(
            name='Hari Terganggu Mental',
            x=['Non-Diabetes', 'Diabetes'],
            y=avg_health_days['MentHlth'],
            marker_color='#a8dadc',
            text=[f"{v:.1f} Hari" for v in avg_health_days['MentHlth']],
            textposition='auto'
        ))

        fig_days.update_layout(
            title="Rata-rata Jumlah Hari 'Kurang Sehat' dalam Sebulan (30 Hari)",
            yaxis_title="Jumlah Hari",
            barmode='group', # Grouped bar chart
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=300
        )
        st.plotly_chart(fig_days, use_container_width=True)

    

        # KORELASI
        with st.container(border=True):
            
            st.subheader("🔗 Hubungan Antar Variabel (Korelasi)")
            st.caption("Analisis statistik untuk melihat hubungan sebab-akibat antar faktor risiko.")

            # HEATMAP
            selected_features = ['Diabetes_binary', 'HighBP', 'HighChol', 'BMI', 'Age', 'GenHlth', 'DiffWalk', 'HeartDiseaseorAttack']
            
            rename_map = {
                'Diabetes_binary': 'Diabetes',
                'HighBP': 'Darah Tinggi',
                'HighChol': 'Kolesterol',
                'GenHlth': 'Kesehatan Umum',
                'DiffWalk': 'Susah Jalan',
                'HeartDiseaseorAttack': 'Sakit Jantung',
                'Age': 'Usia'
            }
            
            corr_df = df[selected_features].rename(columns=rename_map).corr()
            
            # Buat Heatmap
            fig_corr = px.imshow(
                corr_df,
                text_auto='.2f', 
                aspect="auto",
                color_continuous_scale='RdBu_r', 
                origin='lower'
            )
            
            fig_corr.update_layout(
                title="Matriks Korelasi (Warna Merah = Hubungan Kuat)",
                margin=dict(t=30, l=0, r=0, b=0),
                height=350, 
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                coloraxis_showscale=False
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            st.markdown("##### 💡 Insight & Kesimpulan")
            
            # Logika Interpretasi
            corr_target = corr_df['Diabetes'].drop('Diabetes')
            top_3 = corr_target.sort_values(ascending=False).head(3)

            # Grid untuk insight
            col_insight1, col_insight2 = st.columns(2)

            with col_insight1:
                st.info(f"**Faktor Utama:**\n\nAnalisis menunjukkan bahwa **{top_3.index[0]}** adalah indikator terkuat (Koefisien: {top_3.values[0]:.2f}). Jika pasien memiliki kondisi ini, risiko diabetes melonjak drastis.")

            with col_insight2:
                st.warning(f"**Pola Komorbiditas:**\n\nTerlihat hubungan erat antara **Darah Tinggi** dan **Kolesterol**. Kedua penyakit ini sering menyerang bersamaan, memperburuk kondisi pasien secara eksponensial.")
            
            st.caption("*Catatan: Angka 1.00 berarti hubungan mutlak, 0.00 berarti tidak berhubungan.*")

        # CALL TO ACTION 
        st.write("")
        st.markdown("### 🩺 Mulai Pengecekan")
        with st.container():
            col_cta1, col_cta2 = st.columns([2, 1])
            with col_cta1:
                st.info("Gunakan fitur prediksi AI kami untuk mendapatkan estimasi risiko pribadi Anda berdasarkan pola hidup.")
            with col_cta2:
                if st.button("Mulai Prediksi Sekarang →", type="primary", use_container_width=True):
                    go_to_prediction()
                    st.rerun()

    else:
        st.error("Dataset tidak ditemukan.")
elif st.session_state.page == 'prediction':
    st.markdown('<p class="main-header">🩺 Estimasi Risiko Diabetes</p>', unsafe_allow_html=True)
    
    # Progress Bar
    progress = st.session_state.current_step / 4
    st.progress(progress)
    
    # DATA PRIBADI
    if st.session_state.current_step == 1:
        st.subheader("1️⃣ Data Pribadi")
        age = st.number_input("Umur", 18, 100, st.session_state.form_data['age'])
        sex = st.radio("Jenis Kelamin", ["Perempuan", "Laki-laki"], index=0 if st.session_state.form_data['sex'] == "Perempuan" else 1)
        
        c1, c2 = st.columns(2)
        weight = c1.number_input("Berat (kg)", 30.0, 200.0, st.session_state.form_data['weight'])
        height = c2.number_input("Tinggi (cm)", 100.0, 250.0, st.session_state.form_data['height'])
        
        bmi = weight / ((height/100)**2)
        st.info(f"📐 BMI Anda: **{bmi:.1f}** ({'Normal' if 18.5 <= bmi < 25 else 'Obesitas' if bmi >= 30 else 'Perlu Perhatian'})")
        
        if st.button("Lanjut →", use_container_width=True):
            st.session_state.form_data.update({'age':age, 'sex':sex, 'weight':weight, 'height':height, 'bmi':bmi})
            next_step(); st.rerun()

    # RIWAYAT KESEHATAN 
    elif st.session_state.current_step == 2:
        st.subheader("2️⃣ Riwayat Kesehatan")
        st.caption("Centang jika Anda pernah/sedang mengalami kondisi ini:")
        
        c1, c2 = st.columns(2)
        with c1:
            HighBP = st.checkbox("Tekanan Darah Tinggi", st.session_state.form_data['HighBP'])
            HighChol = st.checkbox("Kolesterol Tinggi", st.session_state.form_data['HighChol'])
            CholCheck = st.checkbox("Cek Kolesterol (5 thn terakhir)", st.session_state.form_data['CholCheck'])
            Stroke = st.checkbox("Pernah Stroke", st.session_state.form_data['Stroke'])
        with c2:
            HeartDiseaseorAttack = st.checkbox("Penyakit Jantung / Serangan Jantung", st.session_state.form_data['HeartDiseaseorAttack'])
            DiffWalk = st.checkbox("Kesulitan Berjalan / Naik Tangga", st.session_state.form_data['DiffWalk'])
        
        st.markdown("---")
        GenHlth = st.select_slider("Bagaimana kondisi kesehatan Anda secara umum?", options=[1,2,3,4,5], value=st.session_state.form_data['GenHlth'],
                                   format_func=lambda x: {1:"Sangat Baik", 2:"Baik", 3:"Cukup", 4:"Buruk", 5:"Sangat Buruk"}[x])
        
        c1, c2 = st.columns(2)
        MentHlth = c1.slider("Hari Stres/Depresi (30 hari terakhir)", 0, 30, st.session_state.form_data['MentHlth'])
        PhysHlth = c2.slider("Hari Sakit Fisik (30 hari terakhir)", 0, 30, st.session_state.form_data['PhysHlth'])
        
        c1, c2 = st.columns(2)
        if c1.button("← Kembali", use_container_width=True): prev_step(); st.rerun()
        if c2.button("Lanjut →", use_container_width=True):
            st.session_state.form_data.update({'HighBP':HighBP, 'HighChol':HighChol, 'CholCheck':CholCheck, 'Stroke':Stroke, 'HeartDiseaseorAttack':HeartDiseaseorAttack, 'DiffWalk':DiffWalk, 'GenHlth':GenHlth, 'MentHlth':MentHlth, 'PhysHlth':PhysHlth})
            next_step(); st.rerun()

    # GAYA HIDUP 
    elif st.session_state.current_step == 3:
        st.subheader("3️⃣ Kebiasaan & Gaya Hidup")
        
        PhysActivity = st.checkbox("Olahraga Rutin (Min. 30 menit/hari)", st.session_state.form_data['PhysActivity'])
        c1, c2 = st.columns(2)
        Fruits = c1.checkbox("Makan Buah Tiap Hari", st.session_state.form_data['Fruits'])
        Veggies = c2.checkbox("Makan Sayur Tiap Hari", st.session_state.form_data['Veggies'])
        
        st.markdown("---")
        c1, c2 = st.columns(2)
        Smoker = c1.checkbox("Perokok Aktif (Min. 100 batang seumur hidup)", st.session_state.form_data['Smoker'])
        HvyAlcoholConsump = c2.checkbox("Minum Alkohol Berlebihan", st.session_state.form_data['HvyAlcoholConsump'])
        
        c1, c2 = st.columns(2)
        if c1.button("← Kembali", use_container_width=True): prev_step(); st.rerun()
        if c2.button("Lanjut →", use_container_width=True):
            st.session_state.form_data.update({'PhysActivity':PhysActivity, 'Fruits':Fruits, 'Veggies':Veggies, 'Smoker':Smoker, 'HvyAlcoholConsump':HvyAlcoholConsump})
            next_step(); st.rerun()

    # SOSIAL & HASIL
    elif st.session_state.current_step == 4:
        if not st.session_state.show_prediction:
            st.subheader("4️⃣ Faktor Sosial & Ekonomi")
            
            education = st.selectbox("Pendidikan Terakhir", ["SD", "SMP", "SMA", "D3/S1", "S2/S3"], index=2)
            income = st.number_input("Pendapatan Tahunan (Rp)", 0, 1000000000, int(st.session_state.form_data['income']), step=1000000)
            
            c1, c2 = st.columns(2)
            AnyHealthcare = c1.checkbox("Punya Akses Layanan Kesehatan / BPJS", st.session_state.form_data['AnyHealthcare'])
            NoDocbcCost = c2.checkbox("Pernah Batal ke Dokter karena Biaya", st.session_state.form_data['NoDocbcCost'])
            
            c1, c2 = st.columns(2)
            if c1.button("← Kembali", use_container_width=True): prev_step(); st.rerun()
            if c2.button("🔍 Analisis Risiko Sekarang", use_container_width=True, type="primary"):
                st.session_state.form_data.update({'education':education, 'income':income, 'AnyHealthcare':AnyHealthcare, 'NoDocbcCost':NoDocbcCost})
                st.session_state.show_prediction = True
                st.rerun()
        
        else:
            # LOGIC PREDIKSI & CLINICAL GUARDRAILS
            st.subheader("📊 Hasil Analisis")
            
            if model is None:
                st.error("Model AI belum dimuat. Jalankan train_model.py dulu.")
            else:
                # Persiapan Data
                fd = st.session_state.form_data
                input_data = {
                    "HighBP": int(fd['HighBP']), "HighChol": int(fd['HighChol']), "CholCheck": int(fd['CholCheck']),
                    "BMI": fd['bmi'], "Smoker": int(fd['Smoker']), "Stroke": int(fd['Stroke']),
                    "HeartDiseaseorAttack": int(fd['HeartDiseaseorAttack']), "PhysActivity": int(fd['PhysActivity']),
                    "Fruits": int(fd['Fruits']), "Veggies": int(fd['Veggies']), "HvyAlcoholConsump": int(fd['HvyAlcoholConsump']),
                    "AnyHealthcare": int(fd['AnyHealthcare']), "NoDocbcCost": int(fd['NoDocbcCost']),
                    "GenHlth": fd['GenHlth'], "MentHlth": fd['MentHlth'], "PhysHlth": fd['PhysHlth'],
                    "DiffWalk": int(fd['DiffWalk']), "Sex": 1 if fd['sex'] == "Laki-laki" else 0,
                    "Age": map_age_to_ageg5yr(fd['age']),
                    "Education": map_education(fd['education']),
                    "Income": map_income_rp(fd['income'])
                }
                
                # DataFrame & Scaling
                X_input = pd.DataFrame([input_data])
                if feature_names:
                    X_input = X_input[feature_names]
                if scaler and scaled_features_list:
                    try:
                        cols_to_scale = [c for c in scaled_features_list if c in X_input.columns]
                        X_input[cols_to_scale] = scaler.transform(X_input[cols_to_scale])
                    except Exception as e:
                        st.warning(f"Note Scaling: {e}")

                # 3. Prediksi Awal
                raw_prob = model.predict_proba(X_input)[0][1]
                risk_score = 0
                
                # Faktor Kritis (Bobot Besar)
                if input_data['HighBP'] == 1: risk_score += 2
                if input_data['HighChol'] == 1: risk_score += 2
                if input_data['BMI'] >= 30: risk_score += 3 
                elif input_data['BMI'] >= 25: risk_score += 1 
                if input_data['HeartDiseaseorAttack'] == 1: risk_score += 3
                if input_data['Stroke'] == 1: risk_score += 3
                
                # Faktor Tambahan
                if input_data['GenHlth'] >= 4: risk_score += 2 
                if input_data['DiffWalk'] == 1: risk_score += 1
                if input_data['Age'] >= 9: risk_score += 1 

                # Logika Override Probabilitas
                final_prob = raw_prob
                override_msg = ""
                
                if risk_score >= 6: # Sangat Berbahaya 
                    final_prob = max(raw_prob, 0.86) # Paksa ke Sangat Tinggi
                    override_msg = "⚠️ Risiko dikoreksi naik karena komplikasi multi-faktor."
                elif risk_score >= 4: # Berbahaya
                    final_prob = max(raw_prob, 0.72) 
                elif risk_score >= 2: # Waspada
                    final_prob = max(raw_prob, 0.55)
                
                # Tampilkan Hasil
                label, color, icon = get_risk_category(final_prob)
                
                st.markdown(f"""
                <div class="result-card" style="border-top: 5px solid {color};">
                    <h2 style="color: {color}; margin: 0;">{icon} Risiko {label}</h2>
                    <h1 style="font-size: 3.5rem; margin: 10px 0;">{final_prob:.1%}</h1>
                    <div style="background: #eee; height: 10px; border-radius: 5px; width: 100%;">
                        <div style="background: {color}; width: {final_prob*100}%; height: 100%; border-radius: 5px;"></div>
                    </div>
                    <p style="margin-top: 10px; color: #666;">Probabilitas: {final_prob:.4f} (Raw AI: {raw_prob:.4f})</p>
                    {f'<p style="color: #e67e22; font-size: 0.9rem;">{override_msg}</p>' if override_msg else ''}
                </div>
                """, unsafe_allow_html=True)
                
                # Rekomendasi
                c1, c2 = st.columns([3, 2])
                with c1:
                    st.subheader("📝 Saran Kesehatan")
                    if final_prob > 0.5:
                        st.error("""
                        **PERHATIAN MEDIS DIPERLUKAN:**
                        1. Segera cek **Gula Darah Puasa** & **HbA1c**.
                        2. Konsultasi dokter untuk manajemen Hipertensi/Kolesterol (jika ada).
                        3. Targetkan penurunan berat badan 5-10% jika BMI > 25.
                        """)
                    else:
                        st.success("""
                        **PERTAHANKAN KONDISI:**
                        1. Lanjutkan olahraga rutin 150 menit/minggu.
                        2. Jaga pola makan seimbang (kurangi gorengan/manis).
                        3. Check-up kesehatan minimal 1x setahun.
                        """)
                
                with c2:
                    st.subheader("🔍 Faktor Terdeteksi")
                    detected = []
                    if input_data['HighBP']: detected.append("Tekanan Darah Tinggi")
                    if input_data['HighChol']: detected.append("Kolesterol Tinggi")
                    if input_data['BMI'] >= 30: detected.append(f"Obesitas (BMI {input_data['BMI']:.1f})")
                    if input_data['Smoker']: detected.append("Perokok")
                    if input_data['HeartDiseaseorAttack']: detected.append("Riwayat Jantung")
                    
                    if detected:
                        for d in detected: st.write(f"• {d}")
                    else:
                        st.write("✅ Tidak ada faktor risiko mayor.")
            
            st.markdown("---")
            c1, c2 = st.columns(2)
            if c1.button("← Edit Data", use_container_width=True): 
                st.session_state.show_prediction = False; st.rerun()
            if c2.button("🔄 Mulai Ulang", use_container_width=True): 

                reset_form(); st.rerun()
