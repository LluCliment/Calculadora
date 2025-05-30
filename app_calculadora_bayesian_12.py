import streamlit as st
import pandas as pd
import numpy as np
import time

# --- Funciones de Mapeo (sin cambios) ---
def mapear_edad_a_grupo(edad):
    if 18 <= edad <= 29: return 'De 18 a 29 años'
    elif 30 <= edad <= 39: return 'De 30 a 39 años'
    elif 40 <= edad <= 49: return 'De 40 a 49 años'
    return None

def mapear_tamaño_municipio(habitantes):
    if habitantes <= 10000: return 'Hasta 10000 habitantes'
    elif 10001 <= habitantes <= 50000: return 'De 10001 a 50000 habitantes'
    elif 50001 <= habitantes <= 500000: return 'De 50001 a 500000 habitantes'
    elif habitantes > 500000: return 'Más de 500000 habitantes'
    return None

# --- Constantes (sin cambios mayores aquí, ya que la base del cálculo es P(Test VIH)) ---
CODIFICACION_LECTURA_LIMPIO = 'latin-1'
SEPARADOR = ';'
COLUMNA_RECUENTOS = 'Total' # Se usa para los archivos originales de test VIH y para el nuevo de ETS
COLUMNA_ESTADO_PRUEBA_VIH = 'Realización específica de la prueba del VIH/sida' # Para los datos de test VIH
VALOR_TESTADO_VIH = 'Alguna vez'
VALOR_NO_TESTADO_VIH = 'Nunca'
VALOR_TOTAL_FILA_CARACT = 'Total' # Para las filas de totales en los CSV originales
ANIMATION_SLEEP_TIME = 0.05

# Nombres de columnas para el nuevo archivo de ETS diagnosticadas
COL_SEXO_ETS = 'Sexo'
COL_CCAA_ETS = 'Comunidad autónoma'
COL_DIAGNOSTICADOS_ETS = 'Total' # Asumimos que 'Total' en este archivo son los diagnosticados

COLS_CARACTERISTICAS = { # Se mantiene para el cálculo de P(Test VIH)
    'df_municipio_base': ['Sexo', 'Grupo de edad', 'Tamaño del municipio de residencia'],
    'df_estudios_adj': ['Sexo', 'Grupo de edad', 'Estudios'],
    'df_alcohol_adj': ['Sexo', 'Frecuencia de consumo de alcohol en los últimos 12 meses'],
    'df_pernoctacion_adj': ['Sexo', 'Frecuencia de pernoctación fuera de casa por trabajo/estudio en últimos 12 meses'],
    'df_salidas_adj': ['Sexo', 'Frecuencia de salidas nocturnas en los últimos 12 meses'],
    'df_convivencia_adj': ['Sexo', 'Personas con las que ha convivido en los últimos 12 meses'],
    'df_ccaa_adj': ['Sexo', 'Comunidad autónoma'] # Este df_ccaa_adj será usado para el denominador del factor
}

@st.cache_data
def cargar_dataframes_test_vih(): # Renombrado para claridad
    dfs = {}
    file_mapping = {
        'df_municipio_base': 'va_tamaño_municipio_edad.csv',
        'df_estudios_adj': 'va_estudios_edad.csv',
        'df_alcohol_adj': 'va_consumo_alcohol.csv',
        'df_pernoctacion_adj': 'va_pernoctacion_fuera_de_casa.csv',
        'df_salidas_adj': 'va_salidas_nocturnas.csv',
        'df_convivencia_adj': 'va_convivencia.csv',
        'df_ccaa_adj': 'va_comunidad_autonoma.csv' # Este es el importante para el denominador del factor
    }
    error_ocurrido_carga = False
    for df_key, filename in file_mapping.items():
        try:
            df_temp = pd.read_csv(filename, sep=SEPARADOR, encoding=CODIFICACION_LECTURA_LIMPIO, decimal=',', thousands='.')
            
            # Comprobaciones para los archivos de test VIH
            if COLUMNA_RECUENTOS not in df_temp.columns or COLUMNA_ESTADO_PRUEBA_VIH not in df_temp.columns:
                st.error(f"Columnas '{COLUMNA_RECUENTOS}' o '{COLUMNA_ESTADO_PRUEBA_VIH}' faltantes en {filename} (datos Test VIH).")
                error_ocurrido_carga = True; break
            for col_caract_key in COLS_CARACTERISTICAS[df_key]:
                 if col_caract_key not in df_temp.columns:
                    st.error(f"Columna de característica '{col_caract_key}' faltante en {filename} (datos Test VIH).")
                    error_ocurrido_carga = True; break
            if error_ocurrido_carga: break
            dfs[df_key] = df_temp
        except Exception as e:
            st.error(f"Error al cargar o procesar {filename} (datos Test VIH): {e}")
            error_ocurrido_carga = True; break
    if error_ocurrido_carga: st.stop(); return None
    return dfs

@st.cache_data
def cargar_dataframe_ets_diagnosticados():
    filename_ets = 'va_ets_comunidad_autonoma.csv'
    try:
        df_ets = pd.read_csv(filename_ets, sep=SEPARADOR, encoding=CODIFICACION_LECTURA_LIMPIO, decimal=',', thousands='.')
        # Comprobar columnas necesarias para el archivo de ETS diagnosticados
        if not all(col in df_ets.columns for col in [COL_SEXO_ETS, COL_CCAA_ETS, COL_DIAGNOSTICADOS_ETS]):
            st.error(f"Columnas '{COL_SEXO_ETS}', '{COL_CCAA_ETS}' o '{COL_DIAGNOSTICADOS_ETS}' faltantes en {filename_ets}.")
            return None
        # Asegurar que la columna de recuentos sea numérica
        if not pd.api.types.is_numeric_dtype(df_ets[COL_DIAGNOSTICADOS_ETS]):
            try:
                df_ets[COL_DIAGNOSTICADOS_ETS] = pd.to_numeric(df_ets[COL_DIAGNOSTICADOS_ETS].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False))
            except ValueError:
                st.error(f"No se pudo convertir la columna '{COL_DIAGNOSTICADOS_ETS}' a numérica en {filename_ets}.")
                return None
        return df_ets
    except Exception as e:
        st.error(f"Error al cargar o procesar {filename_ets}: {e}")
        return None

def get_recuentos_test_vih(df, filtro_dict): # Renombrado para claridad
    query_parts = []
    for col, val in filtro_dict.items():
        if col not in df.columns: return None
        valor_escapado = str(val).replace("'", "''")
        query_parts.append(f"`{col}` == '{valor_escapado}'")
    if not query_parts: return 0
    query_str = " & ".join(query_parts)
    try: resultado = df.query(query_str)
    except Exception: return None
    if resultado.empty: return 0
    if not pd.api.types.is_numeric_dtype(resultado[COLUMNA_RECUENTOS]):
        print(f"ADVERTENCIA: La columna '{COLUMNA_RECUENTOS}' no es numérica. Query: {query_str}. Dtype: {resultado[COLUMNA_RECUENTOS].dtype}")
        return None
    return resultado[COLUMNA_RECUENTOS].sum()

def calcular_likelihoods_caracteristica_test_vih( # Renombrado
    df_adj, sexo_usr, valor_caracteristica_usr, nombre_col_caracteristica_df,
    grupo_edad_map=None):
    filtros_base_demograficos = {'Sexo': sexo_usr}
    if grupo_edad_map and 'Grupo de edad' in df_adj.columns:
        filtros_base_demograficos['Grupo de edad'] = grupo_edad_map
    
    n_E_H = get_recuentos_test_vih(df_adj, {**filtros_base_demograficos, nombre_col_caracteristica_df: valor_caracteristica_usr, COLUMNA_ESTADO_PRUEBA_VIH: VALOR_TESTADO_VIH})
    if n_E_H is None: return None, None
    
    n_E_notH = get_recuentos_test_vih(df_adj, {**filtros_base_demograficos, nombre_col_caracteristica_df: valor_caracteristica_usr, COLUMNA_ESTADO_PRUEBA_VIH: VALOR_NO_TESTADO_VIH})
    if n_E_notH is None: return None, None
    
    n_H_total_cat = get_recuentos_test_vih(df_adj, {**filtros_base_demograficos, nombre_col_caracteristica_df: VALOR_TOTAL_FILA_CARACT, COLUMNA_ESTADO_PRUEBA_VIH: VALOR_TESTADO_VIH})
    if n_H_total_cat is None: return None, None
    
    n_notH_total_cat = get_recuentos_test_vih(df_adj, {**filtros_base_demograficos, nombre_col_caracteristica_df: VALOR_TOTAL_FILA_CARACT, COLUMNA_ESTADO_PRUEBA_VIH: VALOR_NO_TESTADO_VIH})
    if n_notH_total_cat is None: return None, None

    p_E_dado_H = n_E_H / n_H_total_cat if n_H_total_cat > 0 else 0.0
    p_E_dado_notH = n_E_notH / n_notH_total_cat if n_notH_total_cat > 0 else 0.0
    
    return p_E_dado_H, p_E_dado_notH

def calcular_prob_test_vih( # Renombrado
    sexo_usr, edad_usr, tam_municipio_usr, estudios_usr, consumo_alcohol_usr,
    pernoctacion_usr, salidas_nocturnas_usr, convivencia_usr, comunidad_autonoma_usr, # CCAA se usa aquí para P(Test VIH)
    df_mun_b, df_est_a, df_alc_a, df_pern_a, df_sal_a, df_conv_a, df_ca_a ):
    grupo_edad_map = mapear_edad_a_grupo(edad_usr)
    tam_municipio_map = mapear_tamaño_municipio(tam_municipio_usr)
    if not grupo_edad_map or not tam_municipio_map: return None
    
    n_testados_base = get_recuentos_test_vih(df_mun_b, {COLS_CARACTERISTICAS['df_municipio_base'][0]: sexo_usr, COLS_CARACTERISTICAS['df_municipio_base'][1]: grupo_edad_map, COLS_CARACTERISTICAS['df_municipio_base'][2]: tam_municipio_map, COLUMNA_ESTADO_PRUEBA_VIH: VALOR_TESTADO_VIH})
    n_total_grupo_base = get_recuentos_test_vih(df_mun_b, {COLS_CARACTERISTICAS['df_municipio_base'][0]: sexo_usr, COLS_CARACTERISTICAS['df_municipio_base'][1]: grupo_edad_map, COLS_CARACTERISTICAS['df_municipio_base'][2]: tam_municipio_map, COLUMNA_ESTADO_PRUEBA_VIH: VALOR_TOTAL_FILA_CARACT})
    if n_testados_base is None or n_total_grupo_base is None: return None
    
    p_base_test_vih = n_testados_base / n_total_grupo_base if n_total_grupo_base > 0 else 0.0
    if not (0 <= p_base_test_vih <= 1): return None
    
    log_odds_acumulado = np.log(p_base_test_vih / (1 - p_base_test_vih)) if p_base_test_vih > 0 and p_base_test_vih < 1 else (-np.inf if p_base_test_vih == 0 else np.inf)
    
    caracteristicas_config_list = [
        ('Estudios', df_est_a, estudios_usr, COLS_CARACTERISTICAS['df_estudios_adj'][2], grupo_edad_map),
        ('Consumo Alcohol', df_alc_a, consumo_alcohol_usr, COLS_CARACTERISTICAS['df_alcohol_adj'][1], None),
        ('Pernoctación', df_pern_a, pernoctacion_usr, COLS_CARACTERISTICAS['df_pernoctacion_adj'][1], None),
        ('Salidas Nocturnas', df_sal_a, salidas_nocturnas_usr, COLS_CARACTERISTICAS['df_salidas_adj'][1], None),
        ('Convivencia', df_conv_a, convivencia_usr, COLS_CARACTERISTICAS['df_convivencia_adj'][1], None),
        ('CCAA', df_ca_a, comunidad_autonoma_usr, COLS_CARACTERISTICAS['df_ccaa_adj'][1], None) ] # CCAA como característica para P(Test VIH)
        
    for _, df_adj, valor_usr, col_caracteristica_df, edad_map_para_caract in caracteristicas_config_list:
        opciones_validas_df = [str(opt).lower() for opt in df_adj[col_caracteristica_df].unique() if str(opt).lower() != VALOR_TOTAL_FILA_CARACT.lower() and str(opt).lower() != 'no consta']
        if str(valor_usr).lower() not in opciones_validas_df: continue
        
        p_E_dado_Testado, p_E_dado_NoTestado = calcular_likelihoods_caracteristica_test_vih(
            df_adj, sexo_usr, valor_usr, col_caracteristica_df,
            grupo_edad_map=edad_map_para_caract )

        if p_E_dado_Testado is not None and p_E_dado_NoTestado is not None:
            log_likelihood_ratio = 0.0
            if p_E_dado_NoTestado == 0 and p_E_dado_Testado == 0: log_likelihood_ratio = 0.0
            elif p_E_dado_NoTestado == 0: log_likelihood_ratio = np.inf if p_E_dado_Testado > 0 else 0.0
            elif p_E_dado_Testado == 0: log_likelihood_ratio = -np.inf
            else: log_likelihood_ratio = np.log(p_E_dado_Testado / p_E_dado_NoTestado)
            
            current_log_odds_sign = np.sign(log_odds_acumulado) if np.isfinite(log_odds_acumulado) else log_odds_acumulado
            lr_sign = np.sign(log_likelihood_ratio) if np.isfinite(log_likelihood_ratio) else log_likelihood_ratio
            if np.isinf(current_log_odds_sign) and np.isinf(lr_sign) and current_log_odds_sign != lr_sign: log_odds_acumulado = 0.0
            else: log_odds_acumulado += log_likelihood_ratio
            if np.isposinf(current_log_odds_sign) and np.isposinf(lr_sign): log_odds_acumulado = np.inf
            if np.isneginf(current_log_odds_sign) and np.isneginf(lr_sign): log_odds_acumulado = -np.inf
            
    if np.isneginf(log_odds_acumulado): return 0.0
    elif np.isposinf(log_odds_acumulado): return 1.0
    elif np.isnan(log_odds_acumulado): return None
    else: return np.exp(log_odds_acumulado) / (1 + np.exp(log_odds_acumulado))

def calcular_factor_ets(sexo_usr, ccaa_usr, df_ets_diagnosticados, df_ccaa_test_vih_data):
    """
    Calcula el factor: N(Diagnosticados ETS) / N(Con Test VIH) para un sexo y CCAA.
    df_ccaa_test_vih_data es el df_ccaa_adj de los datos de test VIH.
    """
    if df_ets_diagnosticados is None or df_ccaa_test_vih_data is None:
        st.warning("Datos para el factor ETS no disponibles.")
        return None

    # Numerador: N(Personas Diagnosticadas ETS en Sexo, CCAA)
    fila_ets = df_ets_diagnosticados[
        (df_ets_diagnosticados[COL_SEXO_ETS] == sexo_usr) &
        (df_ets_diagnosticados[COL_CCAA_ETS] == ccaa_usr)
    ]
    if fila_ets.empty:
        st.warning(f"No hay datos de ETS diagnosticados para {sexo_usr}, {ccaa_usr}.")
        return None # O podrías devolver 0 o un valor por defecto
    
    # Asegurar que la columna de recuentos es numérica antes de sumar
    if not pd.api.types.is_numeric_dtype(fila_ets[COL_DIAGNOSTICADOS_ETS]):
         st.error(f"La columna de diagnosticados ETS '{COL_DIAGNOSTICADOS_ETS}' no es numérica tras la carga.")
         return None
    
    n_diagnosticados_ets = fila_ets[COL_DIAGNOSTICADOS_ETS].sum() # Usar sum() por si hay múltiples filas (no debería si Sexo y CCAA son únicos)


    # Denominador: N(Personas con Test VIH en Sexo, CCAA)
    # Usamos get_recuentos_test_vih que ya maneja la estructura de df_ccaa_test_vih_data
    n_con_test_vih = get_recuentos_test_vih(df_ccaa_test_vih_data, {
        'Sexo': sexo_usr,
        'Comunidad autónoma': ccaa_usr,
        COLUMNA_ESTADO_PRUEBA_VIH: VALOR_TESTADO_VIH # Solo los que se hicieron el test
    })
    
    if n_con_test_vih is None: # Error en get_recuentos_test_vih
        st.warning(f"No se pudieron obtener recuentos de tests VIH para {sexo_usr}, {ccaa_usr}.")
        return None
    if n_con_test_vih == 0:
        st.warning(f"Cero personas con test VIH para {sexo_usr}, {ccaa_usr}. El factor ETS no se puede calcular (división por cero).")
        return 0 # O manejar de otra forma, e.g., devolver None o un valor pequeño si n_diagnosticados_ets > 0

    if n_diagnosticados_ets > n_con_test_vih :
        st.warning(f"Advertencia: Más diagnosticados ETS ({n_diagnosticados_ets}) que personas con test VIH ({n_con_test_vih}) para {sexo_usr}, {ccaa_usr}. El factor será > 1.")
        # Esto puede ser un problema de datos o interpretación. El factor podría ser >1.
        # Considera si quieres 'capar' el factor a 1.0.
        # Por ahora, se permite que sea > 1, lo que puede llevar a P_final > 1 si P_test_VIH es alta.

    factor = n_diagnosticados_ets / n_con_test_vih
    return factor

# --- INTERFAZ DE STREAMLIT ---
st.set_page_config(page_title="Calculadora Probabilidad Diagnóstico ETS", layout="centered")
st.title("📊 Estimador de Probabilidad de Diagnóstico de ETS") # CAMBIO
st.markdown("""
Esta herramienta estima la probabilidad de haber sido diagnosticado/a con alguna Enfermedad de Transmisión Sexual (ETS).
Se basa en la probabilidad de haberse realizado un test de VIH (calculada según tu perfil completo)
y un factor de ajuste según tu sexo y comunidad autónoma basado en datos de diagnósticos de ETS y tests de VIH.
**Importante:** Esto es una estimación y no un diagnóstico médico. Consulta siempre a un profesional de la salud.
""")

# Cargar todos los dataframes necesarios
dfs_test_vih = cargar_dataframes_test_vih()
df_ets_diagnosticados_data = cargar_dataframe_ets_diagnosticados()

if dfs_test_vih and df_ets_diagnosticados_data is not None: # Asegurarse que ambos cargaron
    df_municipio_base = dfs_test_vih['df_municipio_base']
    df_estudios_adj = dfs_test_vih['df_estudios_adj']
    df_alcohol_adj = dfs_test_vih['df_alcohol_adj']
    df_pernoctacion_adj = dfs_test_vih['df_pernoctacion_adj']
    df_salidas_adj = dfs_test_vih['df_salidas_adj']
    df_convivencia_adj = dfs_test_vih['df_convivencia_adj']
    df_ccaa_adj = dfs_test_vih['df_ccaa_adj'] # Para el cálculo de P(Test VIH) y el denominador del Factor ETS

    st.sidebar.header("👤 Tu Perfil")
    user_sexo_options = df_municipio_base[COLS_CARACTERISTICAS['df_municipio_base'][0]].unique().tolist() if COLS_CARACTERISTICAS['df_municipio_base'][0] in df_municipio_base.columns else ["(No hay opciones)"]
    user_sexo = st.sidebar.selectbox("Sexo:", options=user_sexo_options, index=0)
    user_edad = st.sidebar.number_input("Edad (18-49 años):", min_value=18, max_value=49, value=25, step=1)
    user_tam_municipio_hab = st.sidebar.number_input("Habitantes en tu municipio:", min_value=1, value=25000, step=1000, help="Introduce el número de habitantes de tu municipio de residencia.")

    def get_unique_options_for_sidebar(df, col_name):
        if col_name not in df.columns: return ["(No disponible)"]
        excluded_values = [VALOR_TOTAL_FILA_CARACT.lower(), 'no consta']
        options = [opt for opt in df[col_name].unique() if str(opt).lower() not in excluded_values]
        return options if options else ["(Sin opciones)"]

    user_estudios = st.sidebar.selectbox("Nivel de estudios:", options=get_unique_options_for_sidebar(df_estudios_adj, COLS_CARACTERISTICAS['df_estudios_adj'][2]))
    user_consumo_alcohol = st.sidebar.selectbox("Consumo de alcohol:", options=get_unique_options_for_sidebar(df_alcohol_adj, COLS_CARACTERISTICAS['df_alcohol_adj'][1]), help="Últimos 12 meses")
    user_pernoctacion = st.sidebar.selectbox("Pernoctación fuera de casa:", options=get_unique_options_for_sidebar(df_pernoctacion_adj, COLS_CARACTERISTICAS['df_pernoctacion_adj'][1]), help="Trabajo/estudio, últimos 12 meses")
    user_salidas_nocturnas = st.sidebar.selectbox("Salidas nocturnas:", options=get_unique_options_for_sidebar(df_salidas_adj, COLS_CARACTERISTICAS['df_salidas_adj'][1]), help="Últimos 12 meses")
    user_convivencia = st.sidebar.selectbox("Convivencia:", options=get_unique_options_for_sidebar(df_convivencia_adj, COLS_CARACTERISTICAS['df_convivencia_adj'][1]), help="Últimos 12 meses")
    
    # Las opciones de CCAA deben ser consistentes entre los dataframes
    # Usaremos las de df_ccaa_adj que se usa para P(Test VIH) y para el denominador del factor
    opciones_ccaa_sidebar = get_unique_options_for_sidebar(df_ccaa_adj, COLS_CARACTERISTICAS['df_ccaa_adj'][1])
    user_comunidad_autonoma = st.sidebar.selectbox("Comunidad autónoma:", options=opciones_ccaa_sidebar)


    if 'show_result' not in st.session_state: st.session_state.show_result = False
    if 'prob_result_ets' not in st.session_state: st.session_state.prob_result_ets = 0.0
    if 'error_calculo' not in st.session_state: st.session_state.error_calculo = False

    if st.sidebar.button("Calcular Estimación de Diagnóstico ETS", type="primary", use_container_width=True): # CAMBIO TEXTO BOTON
        st.session_state.error_calculo = False
        st.session_state.show_result = False
        valid_input = True
        all_user_options = [user_estudios, user_consumo_alcohol, user_pernoctacion, user_salidas_nocturnas, user_convivencia, user_comunidad_autonoma, user_sexo]
        if any(opt in ["(No disponible)", "(Sin opciones)", "(No hay opciones)"] for opt in all_user_options) :
            st.sidebar.error("Una o más selecciones no son válidas.")
            valid_input = False
        if mapear_edad_a_grupo(user_edad) is None: st.sidebar.error("Edad fuera de rango (18-49)."); valid_input = False
        if mapear_tamaño_municipio(user_tam_municipio_hab) is None: st.sidebar.error("Tamaño de municipio no mapeable."); valid_input = False
        
        if valid_input:
            with st.spinner("Calculando tu estimación..."):
                # 1. Calcular P(Test VIH | Perfil_Completo)
                prob_test_vih_perfil = calcular_prob_test_vih(
                    user_sexo, user_edad, user_tam_municipio_hab, user_estudios,
                    user_consumo_alcohol, user_pernoctacion, user_salidas_nocturnas,
                    user_convivencia, user_comunidad_autonoma,
                    df_municipio_base, df_estudios_adj, df_alcohol_adj,
                    df_pernoctacion_adj, df_salidas_adj, df_convivencia_adj, df_ccaa_adj )

                if prob_test_vih_perfil is None:
                    st.session_state.error_calculo = True
                    st.error("No se pudo calcular la probabilidad base de hacerse un test de VIH.")
                else:
                    # 2. Calcular Factor_ETS(Sexo, CCAA)
                    factor_ets = calcular_factor_ets(user_sexo, user_comunidad_autonoma,
                                                     df_ets_diagnosticados_data, df_ccaa_adj) # df_ccaa_adj es el df_ccaa_test_vih_data
                    
                    if factor_ets is None:
                        st.session_state.error_calculo = True
                        st.error("No se pudo calcular el factor de ajuste ETS para la CCAA y sexo seleccionados.")
                    else:
                        # 3. Calcular P_Estimada_ETS
                        prob_estimada_ets = prob_test_vih_perfil * factor_ets
                        
                        # Es importante que la probabilidad final no exceda 100%
                        # Si factor_ets puede ser > 1, prob_estimada_ets podría ser > 1
                        prob_estimada_ets = min(prob_estimada_ets, 1.0) # Capar a 1.0 (100%)
                        prob_estimada_ets = max(prob_estimada_ets, 0.0) # Asegurar que no sea negativa

                        st.session_state.prob_result_ets = prob_estimada_ets
                        st.session_state.show_result = True
            
            if st.session_state.error_calculo:
                st.session_state.show_result = True # Para mostrar el mensaje de error
        else:
            st.session_state.show_result = False

    if st.session_state.show_result:
        if st.session_state.error_calculo:
            # Los mensajes de error específicos ya se muestran arriba
            st.error("No se pudo completar la estimación. Revisa las entradas o los mensajes de advertencia/error anteriores.")
        else:
            st.subheader("Tu Estimación de Probabilidad de Diagnóstico de ETS:") # CAMBIO
            prob_value = st.session_state.prob_result_ets
            progress_bar_placeholder = st.empty()
            status_text_placeholder = st.empty()
            percentage_text_style = "text-align: center; font-size: 2.8em; font-weight: bold; color: #28a745; margin-top: 10px; margin-bottom: 10px;"
            
            # Animación
            current_progress = 0.0
            target_progress = int(prob_value * 100)
            
            for i in range(target_progress + 1):
                time.sleep(ANIMATION_SLEEP_TIME)
                progress_bar_placeholder.progress(i / 100)
                status_text_placeholder.markdown(f"<p style='{percentage_text_style}'>{i}%</p>", unsafe_allow_html=True)
            
            # Mostrar valor final con decimal si es necesario
            status_text_placeholder.markdown(f"<p style='{percentage_text_style}'>{prob_value*100:.1f}%</p>", unsafe_allow_html=True)
            progress_bar_placeholder.progress(prob_value)

            # Mostrar desglose si es útil para depuración o transparencia
            # st.markdown(f"Debug: P(Test VIH | Perfil) = {prob_test_vih_perfil:.4f}, Factor ETS = {factor_ets:.4f}")

    else:
        st.info("Completa tu perfil en el panel lateral y presiona 'Calcular Estimación de Diagnóstico ETS'.")
else:
    st.error("La carga de uno o más DataFrames esenciales falló. La aplicación no puede continuar.")
    if not dfs_test_vih:
        st.error("Específicamente, los datos para calcular la probabilidad de test VIH no se cargaron.")
    if df_ets_diagnosticados_data is None:
        st.error("Específicamente, los datos de ETS diagnosticados no se cargaron.")

st.markdown("---")
st.caption("Esta es una herramienta desarrollada con fines ilustrativos y educativos, no reemplaza el consejo médico profesional ni un diagnóstico.")
st.caption("Datos obtenidos de: Encuesta Nacional de Salud y Hábitos Sexuales (2003), INE.") # <--- LÍNEA AÑADIDA
st.caption("Autores: Lluc Climent Navarro, Raúl Company Martínez, Aleix Torró Abad, Hugo Cuartero Álvarez.")