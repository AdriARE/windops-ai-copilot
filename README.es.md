🇬🇧 [English](README.md) | 🇪🇸 [Español](README.es.md) | 🇧🇷 [Português](README.pt.md)

# WindOps AI Copilot

> Sistema de mantenimiento predictivo explicable para flotas de aerogeneradores basado en una evaluación híbrida del riesgo, detección de anomalías y un copiloto de mantenimiento impulsado por un LLM.

---

# Descripción

WindOps AI Copilot es una aplicación de inteligencia artificial de extremo a extremo que simula el entorno SCADA de un parque eólico, detecta comportamientos anómalos en los aerogeneradores, prioriza las intervenciones de mantenimiento y genera recomendaciones explicables mediante un agente desarrollado con LangGraph.

El proyecto combina generación de datos SCADA sintéticos, detección de anomalías, evaluación híbrida del riesgo e IA generativa para transformar datos operativos en decisiones de mantenimiento transparentes, justificables y respaldadas por evidencias.

---

# Características

- Generación de datos SCADA sintéticos con escenarios de fallo configurables.
- Pipeline de ingeniería de variables para calcular indicadores de salud de los aerogeneradores.
- Detección de anomalías mediante Isolation Forest.
- Evaluación híbrida del riesgo combinando umbrales de ingeniería y el comportamiento relativo de la flota.
- Priorización del mantenimiento de toda la flota.
- Copiloto de mantenimiento basado en LangGraph.
- Cambio automático a Demo Mode.
- Live Mode protegido mediante contraseña.
- Dashboard interactivo desarrollado con Streamlit.
- Interfaz de línea de comandos (CLI).
- Exportación de informes en formatos JSON, CSV y PDF.

---

# Arquitectura

```text
                    Raw SCADA Data
                          │
                          ▼
               Hourly Aggregation
                          │
                          ▼
              Feature Engineering
                          │
                          ▼
           Isolation Forest Detection
                          │
                          ▼
              Hybrid Risk Scoring
                          │
                          ▼
              Fleet Prioritisation
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
   LangGraph AI Agent              Streamlit Dashboard
          │                               │
          └───────────────┬───────────────┘
                          ▼
             Maintenance Action Plans
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
        JSON             CSV             PDF
```

La evaluación híbrida del riesgo se compone de:

- **70 % de riesgo absoluto**, calculado a partir de umbrales de ingeniería.
- **30 % de riesgo relativo**, normalizado con respecto al comportamiento del resto de la flota.

Este enfoque evita que un deterioro generalizado del parque oculte el comportamiento anómalo de un aerogenerador concreto.

---

# Estructura del proyecto

```text
windops-ai-copilot/

├── app/
│   ├── agent.py                 # LangGraph agent
│   ├── app.py                   # Streamlit dashboard
│   └── cli.py                   # Command-line interface
│
├── notebooks/
│   ├── 01_eda_demo_data.ipynb
│   ├── 02_features_and_scoring.ipynb
│   └── 03_agent_decisions.ipynb
│
├── reports/
│
├── src/
│   ├── anomaly.py
│   ├── config.py
│   ├── data_generation.py
│   ├── expected_power.py
│   ├── features.py
│   ├── impact.py
│   ├── io.py                    # CSV, Markdown and PDF export
│   ├── prioritization.py
│   └── risk.py
│
├── requirements.txt
└── README.md
```

---

# Escenarios de demostración

| Escenario | Descripción |
|----------|-------------|
| `green` | Flota en condiciones normales |
| `gearbox` | Degradación de la caja multiplicadora |
| `pitch` | Fallo del sistema de pitch |
| `yaw` | Desalineación del sistema de yaw |
| `mixed` | Múltiples fallos simultáneos |
| `red` | Degradación severa de la flota |

| Fallo | Riesgo predominante | Comportamiento habitual |
|--------|---------------------|-------------------------|
| Gearbox degradation | Mecánico | Elevada pérdida de potencia acompañada de un aumento de la temperatura de la caja multiplicadora |
| Pitch malfunction | Aerodinámico | Reducción de la producción pese a disponer de condiciones de viento normales |
| Yaw misalignment | Aerodinámico | Pérdida de potencia persistente provocada por la desalineación del rotor |
| Sensor drift | Anomalía | Comportamiento anómalo de los sensores detectado mediante Isolation Forest |

---

# Primeros pasos

## Requisitos

- Python 3.10 o superior.

Instala las dependencias del proyecto:

```bash
pip install -r requirements.txt
```

Crea un archivo `.env`:

```text
ANTHROPIC_API_KEY=your_api_key
LIVE_MODE_PASSWORD=your_password
```

Si no se proporciona una clave de la API de Anthropic, la aplicación se ejecuta íntegramente en **Demo Mode**, utilizando el motor de diagnóstico determinista integrado basado en reglas.

Incluso cuando se dispone de una `ANTHROPIC_API_KEY` válida, la aplicación de Streamlit siempre arranca en **Demo Mode**. Para habilitar las llamadas reales a la API es necesario desbloquear explícitamente el **Live Mode** desde la barra lateral mediante `LIVE_MODE_PASSWORD`, evitando así el consumo accidental de tokens.

---

# Dashboard de Streamlit

Inicia el dashboard:

```bash
streamlit run app/app.py
```

El dashboard incluye:

- Estado general de la flota.
- Inspección individual de aerogeneradores.
- Recomendaciones de mantenimiento generadas por IA.
- Sección desplegable **Why**, donde se explica el razonamiento de cada plan de acción.
- Traza completa de ejecución del agente.
- Exportación de informes en PDF.
- Indicador del modo de ejecución (Demo Mode / Live Mode).

---

# Interfaz de línea de comandos

Run the escenario por defecto:

```bash
python -m app.cli
```

Ejecuta un escenario diferente:

```bash
python -m app.cli --scenario red
```

Analiza cinco aerogeneradores:

```bash
python -m app.cli --top-n 5
```

Exporta los informes:

```bash
python -m app.cli --scenario mixed --top-n 5 --export
```

Ejecuta únicamente el pipeline analítico:

```bash
python -m app.cli --no-agent
```

Muestra la ayuda:

```bash
python -m app.cli --help
```

> **Nota**
>
> La CLI debe ejecutarse como un módulo de Python (`python -m app.cli`) y no como un script. De este modo se conserva la estructura del paquete y se evitan problemas de importación relacionados con `sys.path`.

---

# Notebooks

El proyecto incluye tres notebooks que documentan todo el proceso de desarrollo.

| Notebook | Objetivo |
|-----------|----------|
| **01** | Análisis exploratorio de los datos SCADA sintéticos |
| **02** | Ingeniería de variables, detección de anomalías y evaluación híbrida del riesgo |
| **03** | Flujo de trabajo del agente LangGraph y generación de recomendaciones de mantenimiento |

Se recomienda ejecutarlos en orden para seguir paso a paso el desarrollo completo del proyecto.

---

# Evaluación híbrida del riesgo

La puntuación final de riesgo de cada aerogenerador se obtiene combinando tres subíndices interpretables.

| Subíndice | Peso | Variables principales |
|-----------|------|-----------------------|
| Aerodinámico | 50 % | Power gap, yaw proxy e inestabilidad del pitch |
| Mecánico | 30 % | Temperatura del aceite de la caja multiplicadora y tendencia de vibraciones |
| Anomalías | 20 % | Puntuación de Isolation Forest y persistencia de las anomalías |

Cada subíndice combina umbrales de ingeniería con el comportamiento relativo de la flota:

```text
Risk = 0.70 × Absolute Component
     + 0.30 × Relative Component
```

La prioridad final de mantenimiento se calcula combinando:

- **50 % Puntuación de riesgo**
- **35 % Pérdida de energía estimada**
- **15 % Criticidad del activo**

Todos estos pesos pueden modificarse desde `src/config.py`.

---

# Agente LangGraph

El copiloto de IA sigue un flujo de trabajo de tipo ReAct implementado con LangGraph.

Herramientas disponibles:

- `get_priority_ranking`
- `get_turbine_details`
- `submit_action_plan`

La aplicación selecciona automáticamente el modo de ejecución adecuado:

| Modo | Comportamiento |
|------|----------------|
| **Live Mode** | Utiliza Claude a través de la API de Anthropic |
| **Demo Mode** | Utiliza el motor de diagnóstico determinista integrado |

Si la API de Anthropic no está disponible o falla la autenticación, la aplicación cambia automáticamente a **Demo Mode**, manteniendo exactamente el mismo flujo de trabajo y el mismo formato de salida.

Cada recomendación de mantenimiento incluye los subíndices de riesgo que la respaldan, las señales operativas relevantes, las llamadas a las herramientas y la traza completa de ejecución, permitiendo comprender cómo se ha generado cada recomendación.

---

# Ejemplo de plan de acción

El agente de IA genera recomendaciones de mantenimiento estructuradas en formato JSON.

```json
{
  "turbine_id": "WTG-02",
  "urgency": "high",
  "fault_hypothesis": "Degradación de la caja multiplicadora o del tren de potencia",
  "recommended_action": "1. Programar una toma de muestras del aceite de la caja multiplicadora y un análisis de vibraciones en las próximas 24 horas. 2. Supervisar la temperatura de la caja multiplicadora durante el siguiente ciclo de operación. 3. Inspeccionar el sistema de lubricación y los rodamientos durante la próxima ventana de mantenimiento.",
  "rationale": "El riesgo mecánico es el predominante debido al aumento de la temperatura del aceite de la caja multiplicadora y a una pérdida sostenida de potencia, lo que apunta a una probable degradación del tren de potencia."
}
```

Las recomendaciones se muestran en el dashboard de Streamlit y también pueden exportarse como informes en formato JSON, CSV y PDF.

---

# Demo Mode vs Live Mode

| Funcionalidad | Demo Mode | Live Mode |
|---------------|-----------|-----------|
| Pipeline analítico | ✅ | ✅ |
| Evaluación del riesgo | ✅ | ✅ |
| Priorización | ✅ | ✅ |
| Dashboard | ✅ | ✅ |
| Exportación a PDF | ✅ | ✅ |
| Traza del agente | ✅ | ✅ |
| API de Claude | ❌ | ✅ |

El **Demo Mode** utiliza el motor de diagnóstico determinista basado en reglas incluido en el proyecto.

El **Live Mode** sustituye ese motor por Claude a través de la API de Anthropic, manteniendo el mismo flujo de trabajo y la misma estructura de salida.

---


# Limitaciones conocidas

## Datos

- Los datos SCADA utilizados en el proyecto son sintéticos.
- El comportamiento de las señales es físicamente coherente, pero no refleja toda la complejidad de un parque eólico real.
- Actualmente solo se simula un modelo genérico de aerogenerador de 3 MW.

## Analítica

- Los umbrales utilizados en la evaluación del riesgo se basan en criterios de ingeniería y no han sido aprendidos a partir de históricos reales de fallos.
- Isolation Forest se entrena y evalúa sobre el mismo conjunto de datos sintético.
- Los registros históricos de mantenimiento todavía no se incorporan al proceso de priorización.

## Agente de IA

- El Live Mode requiere una clave válida de la API de Anthropic.
- El Live Mode consume créditos de la API de Anthropic.
- El dashboard de Streamlit siempre se inicia en **Demo Mode** y requiere `LIVE_MODE_PASSWORD` para habilitar el Live Mode.
- **La CLI todavía no aplica esta protección.** Si detecta una clave válida de la API de Anthropic, se ejecuta automáticamente en Live Mode. Alinear este comportamiento con el dashboard sigue siendo una tarea pendiente.
- Las recomendaciones están diseñadas para apoyar la toma de decisiones de ingeniería, no para sustituir el criterio técnico de un ingeniero.
- Actualmente no existe integración con plataformas CMMS (SAP PM, IBM Maximo, etc.).

---

# Stack tecnológico

| Capa | Tecnología |
|--------|------------|
| Lenguaje | Python |
| Procesamiento de datos | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Detección de anomalías | Isolation Forest |
| Framework de IA | LangGraph, LangChain |
| LLM | Claude (Anthropic) |
| Dashboard | Streamlit |
| Visualización | Matplotlib, Seaborn |
| Informes | JSON, CSV, PDF |
| Entorno | python-dotenv |
| Desarrollo | Jupyter Notebook |

---

# Próximas mejoras

Los siguientes hitos previstos para el proyecto son:

- Aplicar la misma protección mediante `LIVE_MODE_PASSWORD` a la CLI.
- Implementar la ingestión de datos SCADA reales a través de `src/ingesta.py`.
- Validar el pipeline utilizando el dataset del parque eólico Kelmarsh Wind Farm.
- Incorporar un mecanismo de respaldo automático a DeepSeek cuando Anthropic no esté disponible.
- Incrementar la cobertura de pruebas automatizadas mediante `pytest`.
- Persistir los datasets simulados para evitar regenerar los escenarios en cada ejecución.
- Desplegar la aplicación en Streamlit Community Cloud.
- Grabar una demostración completa que muestre el dashboard, el pipeline analítico y el flujo de trabajo del copiloto de IA.

---

# Autor

**Adrián Rodríguez Estévez**

Ingeniero en transición hacia Data Science e IA, con experiencia profesional en instalación, puesta en marcha de parques eólicos y gestión de proyectos en Europa y LATAM.

Este proyecto combina experiencia en el sector de las energías renovables con Data Science, Machine Learning e IA Generativa para desarrollar un flujo de trabajo de mantenimiento predictivo explicable aplicado a flotas de aerogeneradores.

**Conecta conmigo**

- GitHub: https://github.com/AdriARE
- LinkedIn: https://www.linkedin.com/in/arestevez/

---

# Licencia

Este proyecto se publica con fines educativos y como parte de mi portfolio profesional.

Siéntete libre de explorar el código, reproducir los análisis y utilizar el proyecto como referencia para aprendizaje, investigación o inspiración en desarrollos similares.