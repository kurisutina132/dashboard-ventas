import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="Dashboard de Predicción de Ventas",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS personalizado
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
        font-weight: bold;
        padding: 10px;
        border-radius: 10px;
        border: none;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #764ba2;
    }
    </style>
""", unsafe_allow_html=True)

# Función para cargar datos y modelo
@st.cache_resource
def cargar_modelo():
    try:
        modelo = joblib.load('models/modelo_final.joblib')
        return modelo
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        return None

@st.cache_data
def cargar_datos():
    try:
        df = pd.read_csv('data/processed/inferencia_df_transformado.csv')
        df['fecha'] = pd.to_datetime(df['fecha'])
        return df
    except Exception as e:
        st.error(f"❌ Error al cargar los datos: {e}")
        return None

# Función para hacer predicciones recursivas
def predecir_recursivo(df_producto, modelo, ajuste_descuento, escenario_competencia):
    """
    Realiza predicciones recursivas día a día actualizando los lags.
    """
    df_pred = df_producto.copy()
    df_pred = df_pred.sort_values('fecha').reset_index(drop=True)
    
    # Ajustar precio_venta según descuento
    factor_descuento = 1 + (ajuste_descuento / 100)
    df_pred['precio_venta'] = df_pred['precio_base'] * factor_descuento
    
    # Ajustar precio_competencia según escenario
    factor_competencia = 1 + (escenario_competencia / 100)
    df_pred['Amazon'] = df_pred['Amazon'] * factor_competencia
    df_pred['Decathlon'] = df_pred['Decathlon'] * factor_competencia
    df_pred['Deporvillage'] = df_pred['Deporvillage'] * factor_competencia
    df_pred['precio_competencia'] = df_pred[['Amazon', 'Decathlon', 'Deporvillage']].mean(axis=1)
    
    # Recalcular variables derivadas
    df_pred['descuento_pct'] = ((df_pred['precio_base'] - df_pred['precio_venta']) / df_pred['precio_base']) * 100
    df_pred['ratio_precio'] = df_pred['precio_venta'] / df_pred['precio_competencia']
    
    # Obtener columnas del modelo
    feature_cols = modelo.feature_names_in_
    
    # Lista para almacenar predicciones
    predicciones = []
    
    # Predecir día a día
    for i in range(len(df_pred)):
        # Asegura que todas las columnas de feature_cols estén presentes en df_pred
        for col in feature_cols:
            if col not in df_pred.columns:
                df_pred[col] = 0
        # Reordena las columnas para que coincidan exactamente
        df_pred = df_pred.reindex(columns=list(df_pred.columns) if set(feature_cols).issubset(df_pred.columns) else list(df_pred.columns) + [col for col in feature_cols if col not in df_pred.columns])
        # Preparar features para predicción
        X_pred = df_pred.iloc[[i]][feature_cols]

        # Hacer predicción
        pred = modelo.predict(X_pred)[0]
        pred = max(0, pred)  # No permitir ventas negativas
        predicciones.append(pred)

        # Actualizar lags para el siguiente día (si no es el último día)
        if i < len(df_pred) - 1:
            # Desplazar lags hacia adelante
            for lag in range(7, 1, -1):
                if f'unidades_vendidas_lag{lag}' in df_pred.columns:
                    df_pred.loc[i + 1, f'unidades_vendidas_lag{lag}'] = df_pred.loc[i, f'unidades_vendidas_lag{lag-1}']

            # Actualizar lag_1 con la predicción actual
            if 'unidades_vendidas_lag1' in df_pred.columns:
                df_pred.loc[i + 1, 'unidades_vendidas_lag1'] = pred

            # Actualizar media móvil de 7 días
            if 'unidades_vendidas_mm7' in df_pred.columns:
                # Usar las últimas 7 predicciones (o menos si no hay suficientes)
                ultimas_predicciones = predicciones[-min(7, len(predicciones)):]
                df_pred.loc[i + 1, 'unidades_vendidas_mm7'] = np.mean(ultimas_predicciones)
    
    df_pred['unidades_predichas'] = predicciones
    df_pred['ingresos_proyectados'] = df_pred['unidades_predichas'] * df_pred['precio_venta']
    
    return df_pred

# Cargar modelo y datos
modelo = cargar_modelo()
df_inferencia = cargar_datos()

if modelo is None or df_inferencia is None:
    st.stop()

# SIDEBAR - Controles de simulación
st.sidebar.title("🎛️ Controles de Simulación")

# Obtener lista de productos únicos
productos = df_inferencia.sort_values('nombre')['nombre'].unique()
producto_seleccionado = st.sidebar.selectbox(
    "📦 Seleccionar Producto",
    productos,
    index=0
)

# Slider de ajuste de descuento
ajuste_descuento = st.sidebar.slider(
    "💰 Ajuste de Descuento",
    min_value=-50,
    max_value=50,
    value=0,
    step=5,
    format="%d%%",
    help="Ajusta el descuento sobre el precio base"
)

# Selector de escenario de competencia
escenario = st.sidebar.radio(
    "Escenario de competencia",
    ["Actual (0%)", "Competencia -5%", "Competencia +5%"],
    index=0,
    label_visibility="collapsed"
)

# Mapear escenario a valor numérico
escenarios_map = {
    "Actual (0%)": 0,
    "Competencia -5%": -5,
    "Competencia +5%": 5
}
escenario_valor = escenarios_map[escenario]

# Botón de simulación
simular = st.sidebar.button("🚀 Simular Ventas", type="primary")

# ZONA PRINCIPAL
st.title(f"📊 Dashboard de Predicción de Ventas - Noviembre 2025")
st.markdown(f"### Producto: **{producto_seleccionado}**")
st.markdown("---")

if simular:
    with st.spinner("🔄 Calculando predicciones recursivas..."):
        # Filtrar datos del producto seleccionado
        df_producto = df_inferencia[df_inferencia['nombre'] == producto_seleccionado].copy()
        
        if len(df_producto) == 0:
            st.error("❌ No hay datos para el producto seleccionado")
            st.stop()
        
        # Hacer predicciones recursivas
        df_resultado = predecir_recursivo(df_producto, modelo, ajuste_descuento, escenario_valor)
        
        # KPIs DESTACADOS
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            unidades_totales = df_resultado['unidades_predichas'].sum()
            st.metric(
                label="📦 Unidades Totales",
                value=f"{unidades_totales:,.0f}",
                delta=f"{unidades_totales/30:.1f} unid/día"
            )
        
        with col2:
            ingresos_totales = df_resultado['ingresos_proyectados'].sum()
            st.metric(
                label="💰 Ingresos Proyectados",
                value=f"€{ingresos_totales:,.2f}",
                delta=f"€{ingresos_totales/30:.2f}/día"
            )
        
        with col3:
            precio_promedio = df_resultado['precio_venta'].mean()
            st.metric(
                label="💵 Precio Promedio",
                value=f"€{precio_promedio:.2f}",
                delta=f"{ajuste_descuento:+d}% descuento"
            )
        
        with col4:
            descuento_promedio = df_resultado['descuento_pct'].mean()
            st.metric(
                label="🏷️ Descuento Promedio",
                value=f"{descuento_promedio:.1f}%",
                delta=None
            )
        
        st.markdown("---")
        
        # GRÁFICO DE PREDICCIÓN DIARIA
        st.subheader("📈 Predicción Diaria de Ventas")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Usar seaborn para el gráfico
        sns.set_style("whitegrid")
        df_resultado['dia'] = df_resultado['fecha'].dt.day
        
        # Línea de predicción
        sns.lineplot(
            data=df_resultado,
            x='dia',
            y='unidades_predichas',
            ax=ax,
            color='#667eea',
            linewidth=2.5,
            marker='o',
            markersize=6
        )
        
        # Marcar Black Friday (día 28)
        black_friday_idx = df_resultado[df_resultado['dia'] == 28].index[0]
        black_friday_ventas = df_resultado.loc[black_friday_idx, 'unidades_predichas']
        
        ax.axvline(x=28, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Black Friday')
        ax.plot(28, black_friday_ventas, 'ro', markersize=12, zorder=5)
        ax.annotate(
            f'Black Friday\n{black_friday_ventas:.0f} unidades',
            xy=(28, black_friday_ventas),
            xytext=(28, black_friday_ventas * 1.15),
            fontsize=11,
            fontweight='bold',
            color='red',
            ha='center',
            arrowprops=dict(arrowstyle='->', color='red', lw=2)
        )
        
        ax.set_xlabel('Día de Noviembre', fontsize=12, fontweight='bold')
        ax.set_ylabel('Unidades Vendidas', fontsize=12, fontweight='bold')
        ax.set_title('Predicción de Ventas Diarias - Noviembre 2025', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, 31, 2))
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("---")
        
        # TABLA DETALLADA
        st.subheader("📋 Detalle Diario de Predicciones")
        
        # Preparar tabla para mostrar
        df_tabla = df_resultado[['fecha', 'precio_venta', 'precio_competencia', 'descuento_pct', 'unidades_predichas', 'ingresos_proyectados']].copy()
        df_tabla['dia_semana'] = df_resultado['fecha'].dt.day_name()
        df_tabla['fecha'] = df_tabla['fecha'].dt.strftime('%d/%m/%Y')
        
        # Renombrar columnas
        df_tabla = df_tabla.rename(columns={
            'fecha': 'Fecha',
            'dia_semana': 'Día',
            'precio_venta': 'Precio Venta (€)',
            'precio_competencia': 'Precio Competencia (€)',
            'descuento_pct': 'Descuento (%)',
            'unidades_predichas': 'Unidades Predichas',
            'ingresos_proyectados': 'Ingresos (€)'
        })
        
        # Formatear números
        df_tabla['Precio Venta (€)'] = df_tabla['Precio Venta (€)'].apply(lambda x: f"€{x:.2f}")
        df_tabla['Precio Competencia (€)'] = df_tabla['Precio Competencia (€)'].apply(lambda x: f"€{x:.2f}")
        df_tabla['Descuento (%)'] = df_tabla['Descuento (%)'].apply(lambda x: f"{x:.1f}%")
        df_tabla['Unidades Predichas'] = df_tabla['Unidades Predichas'].apply(lambda x: f"{x:.0f}")
        df_tabla['Ingresos (€)'] = df_tabla['Ingresos (€)'].apply(lambda x: f"€{x:.2f}")
        
        # Añadir emoji para Black Friday
        df_tabla['Fecha'] = df_tabla['Fecha'].apply(lambda x: f"🔥 {x}" if x.startswith('28') else x)
        
        # Reordenar columnas
        df_tabla = df_tabla[['Fecha', 'Día', 'Precio Venta (€)', 'Precio Competencia (€)', 'Descuento (%)', 'Unidades Predichas', 'Ingresos (€)']]
        
        st.dataframe(df_tabla, use_container_width=True, height=400)
        
        st.markdown("---")
        
        # COMPARATIVA DE ESCENARIOS
        st.subheader("🔄 Comparativa de Escenarios de Competencia")
        st.markdown(f"*Manteniendo descuento al {ajuste_descuento:+d}%*")
        
        # Calcular predicciones para los 3 escenarios
        escenarios_resultados = {}
        
        for nombre_esc, valor_esc in escenarios_map.items():
            df_esc = predecir_recursivo(df_producto, modelo, ajuste_descuento, valor_esc)
            escenarios_resultados[nombre_esc] = {
                'unidades': df_esc['unidades_predichas'].sum(),
                'ingresos': df_esc['ingresos_proyectados'].sum()
            }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📊 Actual (0%)")
            st.metric(
                "Unidades Totales",
                f"{escenarios_resultados['Actual (0%)']['unidades']:,.0f}"
            )
            st.metric(
                "Ingresos Totales",
                f"€{escenarios_resultados['Actual (0%)']['ingresos']:,.2f}"
            )
        
        with col2:
            st.markdown("### 📉 Competencia -5%")
            delta_unidades = escenarios_resultados['Competencia -5%']['unidades'] - escenarios_resultados['Actual (0%)']['unidades']
            delta_ingresos = escenarios_resultados['Competencia -5%']['ingresos'] - escenarios_resultados['Actual (0%)']['ingresos']
            
            st.metric(
                "Unidades Totales",
                f"{escenarios_resultados['Competencia -5%']['unidades']:,.0f}",
                delta=f"{delta_unidades:+,.0f}"
            )
            st.metric(
                "Ingresos Totales",
                f"€{escenarios_resultados['Competencia -5%']['ingresos']:,.2f}",
                delta=f"€{delta_ingresos:+,.2f}"
            )
        
        with col3:
            st.markdown("### 📈 Competencia +5%")
            delta_unidades = escenarios_resultados['Competencia +5%']['unidades'] - escenarios_resultados['Actual (0%)']['unidades']
            delta_ingresos = escenarios_resultados['Competencia +5%']['ingresos'] - escenarios_resultados['Actual (0%)']['ingresos']
            
            st.metric(
                "Unidades Totales",
                f"{escenarios_resultados['Competencia +5%']['unidades']:,.0f}",
                delta=f"{delta_unidades:+,.0f}"
            )
            st.metric(
                "Ingresos Totales",
                f"€{escenarios_resultados['Competencia +5%']['ingresos']:,.2f}",
                delta=f"€{delta_ingresos:+,.2f}"
            )
        
        st.success("✅ Simulación completada exitosamente")

else:
    # Mensaje inicial
    st.info("👈 Configura los parámetros en el panel lateral y pulsa **'Simular Ventas'** para ver las predicciones")
    
    # Mostrar información del dataset
    st.subheader("📊 Información del Dataset")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Productos", len(df_inferencia['nombre'].unique()))
    
    with col2:
        st.metric("Total Registros", len(df_inferencia))
    
    with col3:
        st.metric("Días de Predicción", 30)
