import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import io

# Configuração da página
st.set_page_config(page_title="GammaFlip Analysis", layout="wide", initial_sidebar_state="expanded")

# Título principal
st.title("📊 GammaFlip Analysis Tool")
st.markdown("Análise de Gamma Exposure para Opções")

# ===== FUNÇÕES =====

def calcGammaEx(S, K, vol, T, r, q, optType, OI):
    """Calcula a exposição de Gamma para opções europeias"""
    if T == 0 or vol == 0:
        return 0
    
    dp = (np.log(S/K) + (r - q + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    dm = dp - vol*np.sqrt(T)
    
    if optType == 'call':
        gamma = np.exp(-q*T) * norm.pdf(dp) / (S * vol * np.sqrt(T))
        return OI * 100 * S * S * 0.01 * gamma
    else:
        gamma = K * np.exp(-r*T) * norm.pdf(dm) / (S * S * vol * np.sqrt(T))
        return OI * 100 * S * S * 0.01 * gamma

def isThirdFriday(d):
    """Verifica se a data é a terceira sexta-feira do mês"""
    return d.weekday() == 4 and 15 <= d.day <= 21

def parse_csv_file(file_content):
    """Parse do arquivo CSV CBOE"""
    lines = file_content.split('\n')
    
    # Remover linhas vazias no início
    while len(lines) > 0 and lines[0].strip() == '':
        lines.pop(0)
    
    # Extrair preço spot (primeira linha com conteúdo)
    spot_line = lines[0]
    spot_price = float(spot_line.split('Last:')[1].split(',')[0].strip())
    
    # Extrair data (segunda linha)
    date_line = lines[1]
    date_str = date_line.split('Date:')[1].split(',')[0].strip()
    
    # Parse da data
    try:
        today_date = pd.to_datetime(date_str, format='%d de %B de %Y às %H:%M %Z')
    except:
        try:
            today_date = pd.to_datetime(date_str, format='%d %B %Y')
        except:
            today_date = datetime.now()
    
    # Ler dados das opções (começando na linha 4)
    df = pd.read_csv(io.StringIO('\n'.join(lines[4:])), sep=",")
    
    return df, spot_price, today_date

def process_data(df, spot_price, today_date):
    """Processa os dados das opções"""
    # Renomear colunas corretamente baseado no arquivo CBOE
    df.columns = ['ExpirationDate', 'Calls', 'CallLastSale', 'CallNet', 'CallBid', 'CallAsk', 'CallVol',
                  'CallIV', 'CallDelta', 'CallGamma', 'CallOpenInt', 'StrikePrice', 'Puts', 'PutLastSale',
                  'PutNet', 'PutBid', 'PutAsk', 'PutVol', 'PutIV', 'PutDelta', 'PutGamma', 'PutOpenInt']
    
    # Converter tipos de dados
    df['StrikePrice'] = pd.to_numeric(df['StrikePrice'], errors='coerce')
    df['CallIV'] = pd.to_numeric(df['CallIV'], errors='coerce')
    df['PutIV'] = pd.to_numeric(df['PutIV'], errors='coerce')
    df['CallGamma'] = pd.to_numeric(df['CallGamma'], errors='coerce')
    df['PutGamma'] = pd.to_numeric(df['PutGamma'], errors='coerce')
    df['CallOpenInt'] = pd.to_numeric(df['CallOpenInt'], errors='coerce')
    df['PutOpenInt'] = pd.to_numeric(df['PutOpenInt'], errors='coerce')
    df['CallDelta'] = pd.to_numeric(df['CallDelta'], errors='coerce')
    df['PutDelta'] = pd.to_numeric(df['PutDelta'], errors='coerce')
    
    # Remover linhas com valores NaN críticos
    df = df.dropna(subset=['StrikePrice', 'CallOpenInt', 'PutOpenInt'])
    
    # Calcular Gamma Exposure
    df['CallGEX'] = df['CallGamma'] * df['CallOpenInt'] * 100 * spot_price * spot_price * 0.01
    df['PutGEX'] = df['PutGamma'] * df['PutOpenInt'] * 100 * spot_price * spot_price * 0.01 * -1
    df['TotalGamma'] = (df['CallGEX'] + df['PutGEX']) / 10**9
    
    # Agregar por Strike - IMPORTANTE: especificar apenas colunas numéricas
    dfAgg = df.groupby(['StrikePrice'], as_index=True).agg({
        'CallGEX': 'sum',
        'PutGEX': 'sum',
        'TotalGamma': 'sum',
        'CallDelta': 'sum',
        'PutDelta': 'sum',
        'CallOpenInt': 'sum',
        'PutOpenInt': 'sum'
    })
    
    strikes = dfAgg.index.values
    
    return df, dfAgg, strikes, spot_price

def create_chart1(dfAgg, strikes, spot_price, df):
    """Cria o gráfico 1: Absolute Gamma Exposure"""
    x_data = strikes
    y_data = dfAgg['TotalGamma'].to_numpy()
    
    from_strike = 0.8 * spot_price
    to_strike = 1.2 * spot_price
    
    fig = go.Figure(
        go.Bar(
            x=x_data,
            y=y_data,
            width=0.6,  # Reduzido de 6 para 0.6
            marker_color='rgb(26, 118, 255)',
            marker_line_color='black',
            marker_line_width=0.15,
            name='Gamma Exposure'
        )
    )
    
    if len(y_data) > 0:
        fig.add_shape(
            type='line',
            x0=spot_price,
            y0=min(y_data),
            x1=spot_price,
            y1=max(y_data),
            line=dict(color='red', width=2, dash='dash')
        )
    
    fig.update_layout(
        title={
            'text': f"Total Gamma: ${df['TotalGamma'].sum():,.2f} Bn per 1% Move",
            'font': {'size': 20, 'family': 'Arial Black'}
        },
        xaxis_title='Strike',
        yaxis_title='Spot Gamma Exposure ($ billions/1% move)',
        xaxis=dict(range=[from_strike, to_strike]),
        yaxis=dict(tickformat='$,.2f'),
        plot_bgcolor='white',
        font=dict(family='Arial', size=12, color='black'),
        height=700,
        hovermode='x unified'
    )
    
    return fig

def create_chart2(dfAgg, strikes, spot_price, df):
    """Cria o gráfico 2: Gamma Exposure by Calls and Puts"""
    from_strike = 0.8 * spot_price
    to_strike = 1.2 * spot_price
    
    fig = go.Figure()
    fig.add_bar(x=strikes, y=dfAgg['CallGEX'].to_numpy() / 10**9, width=0.35, name="Call Gamma", marker_color='rgb(0, 100, 200)')
    fig.add_bar(x=strikes, y=dfAgg['PutGEX'].to_numpy() / 10**9, width=0.35, name="Put Gamma", marker_color='rgb(200, 50, 50)')
    
    fig.update_xaxes(range=[from_strike, to_strike])
    chart_title = f"Total Gamma: ${df['TotalGamma'].sum():,.2f} Bn per 1% Move"
    
    fig.update_layout(
        title_text=chart_title,
        title_font=dict(size=20, family="Arial Black"),
        xaxis_title="Strike",
        yaxis_title="Spot Gamma Exposure ($ billions/1% move)",
        plot_bgcolor='white',
        font=dict(family='Arial', size=12, color='black'),
        height=700,
        hovermode='x unified',
        barmode='group'
    )
    
    if len(dfAgg) > 0:
        max_call_gex = max(dfAgg['CallGEX'].to_numpy() / 10**9)
        fig.add_shape(
            dict(
                type="line",
                x0=spot_price,
                y0=0,
                x1=spot_price,
                y1=max_call_gex,
                line=dict(color="red", width=2, dash='dash'),
            )
        )
    
    return fig

def create_chart3(df, today_date, spot_price):
    """Cria o gráfico 3: Gamma Exposure Profile"""
    from_strike = 0.8 * spot_price
    to_strike = 1.2 * spot_price
    
    levels = np.linspace(from_strike, to_strike, 60)
    
    # Converter ExpirationDate para datetime se não estiver
    df['ExpirationDate'] = pd.to_datetime(df['ExpirationDate'], errors='coerce')
    
    # Calcular dias até expiração
    df['daysTillExp'] = df['ExpirationDate'].apply(
        lambda x: 1/262 if pd.isna(x) or (np.busday_count(today_date.date(), x.date())) == 0 
        else np.busday_count(today_date.date(), x.date())/262
    )
    
    next_expiry = df['ExpirationDate'].min()
    df['IsThirdFriday'] = df['ExpirationDate'].apply(lambda x: isThirdFriday(x) if pd.notna(x) else False)
    third_fridays = df.loc[df['IsThirdFriday'] == True]
    
    if len(third_fridays) > 0:
        next_monthly_exp = third_fridays['ExpirationDate'].min()
    else:
        next_monthly_exp = next_expiry
    
    total_gamma = []
    total_gamma_ex_next = []
    total_gamma_ex_fri = []
    
    # Para cada nível de preço, calcular gamma exposure
    for level in levels:
        df['callGammaEx'] = df.apply(
            lambda row: calcGammaEx(level, row['StrikePrice'], row['CallIV'], 
                                   row['daysTillExp'], 0, 0, "call", row['CallOpenInt']), 
            axis=1
        )
        df['putGammaEx'] = df.apply(
            lambda row: calcGammaEx(level, row['StrikePrice'], row['PutIV'], 
                                   row['daysTillExp'], 0, 0, "put", row['PutOpenInt']), 
            axis=1
        )
        
        total_gamma.append(df['callGammaEx'].sum() - df['putGammaEx'].sum())
        
        ex_nxt = df.loc[df['ExpirationDate'] != next_expiry]
        total_gamma_ex_next.append(ex_nxt['callGammaEx'].sum() - ex_nxt['putGammaEx'].sum())
        
        ex_fri = df.loc[df['ExpirationDate'] != next_monthly_exp]
        total_gamma_ex_fri.append(ex_fri['callGammaEx'].sum() - ex_fri['putGammaEx'].sum())
    
    total_gamma = np.array(total_gamma) / 10**9
    total_gamma_ex_next = np.array(total_gamma_ex_next) / 10**9
    total_gamma_ex_fri = np.array(total_gamma_ex_fri) / 10**9
    
    # Encontrar Gamma Flip Point
    zero_cross_idx = np.where(np.diff(np.sign(total_gamma)))[0]
    
    if len(zero_cross_idx) > 0:
        neg_gamma = total_gamma[zero_cross_idx]
        pos_gamma = total_gamma[zero_cross_idx+1]
        neg_strike = levels[zero_cross_idx]
        pos_strike = levels[zero_cross_idx+1]
        zero_gamma = pos_strike - ((pos_strike - neg_strike) * pos_gamma/(pos_gamma-neg_gamma))
        zero_gamma = zero_gamma[0]
    else:
        zero_gamma = spot_price
    
    # Criar figura
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=levels, y=total_gamma, mode='lines', name='All Expiries', line=dict(width=2)))
    fig.add_trace(go.Scatter(x=levels, y=total_gamma_ex_next, mode='lines', name='Ex-Next Expiry', line=dict(width=2)))
    fig.add_trace(go.Scatter(x=levels, y=total_gamma_ex_fri, mode='lines', name='Ex-Next Monthly Expiry', line=dict(width=2)))
    
    chart_title = f"Gamma Exposure Profile, {today_date.strftime('%d %b %Y')}"
    fig.update_layout(
        title=chart_title,
        title_font=dict(size=20, family="Arial Black"),
        xaxis_title='Index Price',
        yaxis_title='Gamma Exposure ($ billions/1% move)',
        plot_bgcolor='white',
        font=dict(family='Arial', size=12, color='black'),
        height=600,
        hovermode='x unified'
    )
    
    # Adicionar linhas verticais
    if len(total_gamma) > 0:
        fig.add_shape(
            dict(
                type="line",
                x0=spot_price,
                y0=min(total_gamma),
                x1=spot_price,
                y1=max(total_gamma),
                line=dict(color="red", width=2, dash='dash'),
            )
        )
        
        fig.add_shape(
            dict(
                type="line",
                x0=zero_gamma,
                y0=min(total_gamma),
                x1=zero_gamma,
                y1=max(total_gamma),
                line=dict(color="green", width=2, dash='dash'),
            )
        )
        
        # Adicionar anotações
        fig.add_annotation(
            x=spot_price,
            y=max(total_gamma),
            text=f"Spot: {spot_price:,.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="red",
            ax=40,
            ay=-40,
            bgcolor="rgba(255,0,0,0.7)",
            font=dict(color="white", size=10)
        )
        
        fig.add_annotation(
            x=zero_gamma,
            y=max(total_gamma),
            text=f"Flip: {zero_gamma:,.2f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="green",
            ax=-40,
            ay=-40,
            bgcolor="rgba(0,128,0,0.7)",
            font=dict(color="white", size=10)
        )
        
        fig.update_xaxes(range=[from_strike, to_strike])
    
    return fig, zero_gamma

def create_chart4(dfAgg, strikes, spot_price):
    """Cria o gráfico 4: Total Delta por Strike"""
    from_strike = 0.8 * spot_price
    to_strike = 1.2 * spot_price
    
    fig = go.Figure()
    fig.add_bar(
        x=strikes, 
        y=dfAgg['CallDelta'].to_numpy(), 
        width=0.35, 
        name="Call Delta",
        marker_color='rgb(0, 100, 200)'
    )
    fig.add_bar(
        x=strikes, 
        y=dfAgg['PutDelta'].to_numpy(), 
        width=0.35, 
        name="Put Delta",
        marker_color='rgb(200, 50, 50)'
    )
    
    fig.update_xaxes(range=[from_strike, to_strike])
    
    fig.update_layout(
        title_text="Total Delta por Strike",
        title_font=dict(size=20, family="Arial Black"),
        xaxis_title="Strike",
        yaxis_title="Delta Total",
        plot_bgcolor='white',
        font=dict(family='Arial', size=12, color='black'),
        height=700,
        hovermode='x unified',
        barmode='group'
    )
    
    fig.add_shape(
        dict(
            type="line",
            x0=spot_price,
            y0=min(dfAgg['PutDelta'].to_numpy()),
            x1=spot_price,
            y1=max(dfAgg['CallDelta'].to_numpy()),
            line=dict(color="red", width=2, dash='dash'),
        )
    )
    
    return fig

def create_chart5(dfAgg, strikes, spot_price):
    """Cria o gráfico 5: Total Open Interest por Strike"""
    from_strike = 0.8 * spot_price
    to_strike = 1.2 * spot_price
    
    fig = go.Figure()
    fig.add_bar(
        x=strikes, 
        y=dfAgg['CallOpenInt'].to_numpy(), 
        width=0.35, 
        name="Call Open Interest",
        marker_color='rgb(0, 150, 100)'
    )
    fig.add_bar(
        x=strikes, 
        y=dfAgg['PutOpenInt'].to_numpy(), 
        width=0.35, 
        name="Put Open Interest",
        marker_color='rgb(200, 100, 50)'
    )
    
    fig.update_xaxes(range=[from_strike, to_strike])
    
    fig.update_layout(
        title_text="Total Open Interest por Strike",
        title_font=dict(size=20, family="Arial Black"),
        xaxis_title="Strike",
        yaxis_title="Open Interest Total",
        plot_bgcolor='white',
        font=dict(family='Arial', size=12, color='black'),
        height=700,
        hovermode='x unified',
        barmode='group'
    )
    
    fig.add_shape(
        dict(
            type="line",
            x0=spot_price,
            y0=0,
            x1=spot_price,
            y1=max(max(dfAgg['CallOpenInt'].to_numpy()), max(dfAgg['PutOpenInt'].to_numpy())),
            line=dict(color="red", width=2, dash='dash'),
        )
    )
    
    return fig

# ===== INTERFACE STREAMLIT ====

# Sidebar para upload
st.sidebar.header("📁 Upload de Arquivo")
uploaded_file = st.sidebar.file_uploader("Selecione um arquivo CSV", type=['csv'])

if uploaded_file is not None:
    try:
        # Ler arquivo
        file_content = uploaded_file.read().decode('utf-8')
        
        # Parse do arquivo
        df, spot_price, today_date = parse_csv_file(file_content)
        
        # Processar dados
        df, dfAgg, strikes, spot_price = process_data(df, spot_price, today_date)
        
        # Seletor de data de vencimento na sidebar
        st.sidebar.header("📅 Filtros")
        
        # Garantir que ExpirationDate é datetime
        df['ExpirationDate'] = pd.to_datetime(df['ExpirationDate'], errors='coerce')
        
        # Obter datas únicas de vencimento
        expiration_dates = sorted(df['ExpirationDate'].dropna().unique())
        expiration_dates_str = [pd.Timestamp(d).strftime('%d/%m/%Y') for d in expiration_dates]
        
        # Seletor de data
        selected_date_str = st.sidebar.selectbox(
            "Selecione a data de vencimento:",
            expiration_dates_str,
            index=0
        )
        
        # Converter de volta para datetime
        selected_date = pd.to_datetime(selected_date_str, format='%d/%m/%Y')
        
        # Filtrar dados pela data selecionada
        df_filtered = df[df['ExpirationDate'].dt.date == selected_date.date()].copy()
        
        # Recalcular agregação para a data selecionada
        if len(df_filtered) > 0:
            df_filtered['CallGEX'] = df_filtered['CallGamma'] * df_filtered['CallOpenInt'] * 100 * spot_price * spot_price * 0.01
            df_filtered['PutGEX'] = df_filtered['PutGamma'] * df_filtered['PutOpenInt'] * 100 * spot_price * spot_price * 0.01 * -1
            df_filtered['TotalGamma'] = (df_filtered['CallGEX'] + df_filtered['PutGEX']) / 10**9
            
            dfAgg_filtered = df_filtered.groupby(['StrikePrice'], as_index=True).agg({
                'CallGEX': 'sum',
                'PutGEX': 'sum',
                'TotalGamma': 'sum',
                'CallDelta': 'sum',
                'PutDelta': 'sum',
                'CallOpenInt': 'sum',
                'PutOpenInt': 'sum'
            })
            
            strikes_filtered = dfAgg_filtered.index.values
        else:
            dfAgg_filtered = dfAgg
            strikes_filtered = strikes
            df_filtered = df
        
        # Informações principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Preço Spot", f"${spot_price:,.2f}")
        
        with col2:
            st.metric("Total Gamma", f"${df_filtered['TotalGamma'].sum():,.2f}B")
        
        with col3:
            st.metric("Strikes", f"{len(strikes_filtered)}")
        
        with col4:
            st.metric("Vencimento", selected_date.strftime('%d/%m/%Y'))
        
        st.divider()
        
        # Gráfico 1
        st.subheader("📊 Gráfico 1: Gamma Exposure Absoluto")
        fig1 = create_chart1(dfAgg_filtered, strikes_filtered, spot_price, df_filtered)
        st.plotly_chart(fig1, use_container_width=True)
        
        # Gráfico 2
        st.subheader("📊 Gráfico 2: Gamma Exposure por Calls e Puts")
        fig2 = create_chart2(dfAgg_filtered, strikes_filtered, spot_price, df_filtered)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Gráfico 3
        st.subheader("📊 Gráfico 3: Gamma Exposure Profile")
        with st.spinner("Calculando Gamma Profile..."):
            fig3, zero_gamma = create_chart3(df_filtered, selected_date, spot_price)
        
        st.plotly_chart(fig3, use_container_width=True)
        
        # Gráfico 4
        st.subheader("📊 Gráfico 4: Total Delta por Strike")
        fig4 = create_chart4(dfAgg_filtered, strikes_filtered, spot_price)
        st.plotly_chart(fig4, use_container_width=True)
        
        # Gráfico 5
        st.subheader("📊 Gráfico 5: Total Open Interest por Strike")
        fig5 = create_chart5(dfAgg_filtered, strikes_filtered, spot_price)
        st.plotly_chart(fig5, use_container_width=True)
        
        # Informações adicionais
        st.divider()
        st.subheader("📈 Informações Adicionais")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Gamma Flip Point", f"${zero_gamma:,.2f}")
        
        with col2:
            distancia = ((zero_gamma - spot_price) / spot_price) * 100
            st.metric("Distância do Spot", f"{distancia:+.2f}%")
        
        with col3:
            call_gex = dfAgg_filtered['CallGEX'].sum() / 10**9
            st.metric("Call GEX Total", f"${call_gex:,.2f}B")
        
        # Tabela de dados
        st.subheader("📋 Dados por Strike")
        
        display_df = dfAgg_filtered[['CallGEX', 'PutGEX', 'TotalGamma', 'CallDelta', 'PutDelta', 'CallOpenInt', 'PutOpenInt']].copy()
        display_df['CallGEX'] = display_df['CallGEX'] / 10**9
        display_df['PutGEX'] = display_df['PutGEX'] / 10**9
        display_df.columns = ['Call GEX (B)', 'Put GEX (B)', 'Total Gamma (B)', 'Call Delta', 'Put Delta', 'Call OI', 'Put OI']
        
        st.dataframe(display_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Erro ao processar arquivo: {str(e)}")
        import traceback
        st.info(f"Detalhes do erro:\n```\n{traceback.format_exc()}\n```")

else:
    st.info("👈 Faça upload de um arquivo CSV para começar a análise")
    
    # Mostrar exemplo de formato esperado
    st.subheader("📝 Formato Esperado do Arquivo")
    st.markdown("""
    O arquivo deve estar no formato CBOE com:
    - Linha 1: Informações do ativo com "Last: XXX.XX"
    - Linha 2: Data e horário com "Date: ..."
    - Linha 3: Cabeçalhos das colunas
    - Linha 4+: Dados das opções
    
    **Colunas esperadas (22 colunas):**
    - Expiration Date
    - Calls, Last Sale, Net, Bid, Ask, Volume, IV, Delta, Gamma, Open Interest
    - Strike Price
    - Puts, Last Sale, Net, Bid, Ask, Volume, IV, Delta, Gamma, Open Interest
    """)
