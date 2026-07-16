🇬🇧 [English](README.md) | 🇪🇸 [Español](README.es.md) | 🇧🇷 [Português](README.pt.md)

# WindOps AI Copilot

> Sistema de manutenção preditiva explicável para frotas de turbinas eólicas, baseado em avaliação híbrida de risco, detecção de anomalias e um copiloto de manutenção alimentado por um LLM.

---

# Visão Geral

O WindOps AI Copilot é uma aplicação de inteligência artificial de ponta a ponta que simula o ambiente SCADA de um parque eólico, detecta comportamentos anômalos em turbinas eólicas, prioriza intervenções de manutenção e gera recomendações explicáveis por meio de um agente desenvolvido com LangGraph.

O projeto combina geração de dados SCADA sintéticos, detecção de anomalias, avaliação híbrida de risco e IA generativa para transformar dados operacionais em decisões de manutenção transparentes, justificáveis e baseadas em evidências.

---

# Principais Recursos

- Geração de dados SCADA sintéticos com cenários de falha configuráveis.
- Pipeline de engenharia de atributos para calcular indicadores de integridade das turbinas eólicas.
- Detecção de anomalias utilizando Isolation Forest.
- Avaliação híbrida de risco combinando limites de engenharia com o comportamento relativo da frota.
- Priorização das atividades de manutenção em toda a frota.
- Copiloto de manutenção baseado em LangGraph.
- Alternância automática para o Demo Mode.
- Live Mode protegido por senha.
- Dashboard interativo desenvolvido com Streamlit.
- Interface de linha de comando (CLI).
- Exportação de relatórios nos formatos JSON, CSV e PDF.

---

# Arquitetura

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
          ┌────────┴─────────┐
          │                              │
          ▼                              ▼
   LangGraph AI Agent             Streamlit Dashboard
          │                              │
          └────────┬─────────┘
                         ▼
              Maintenance Action Plans
                         │
          ┌────────┼────────┐
          ▼             ▼              ▼
        JSON             CSV            PDF
```

A avaliação híbrida de risco é composta por:

- **70% de risco absoluto**, calculado a partir de limites de engenharia.
- **30% de risco relativo**, normalizado em relação ao comportamento da frota.

Essa abordagem evita que uma degradação generalizada do parque eólico esconda o comportamento anômalo de uma turbina específica.

---

# Estrutura do Projeto

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

# Cenários de Demonstração

| Cenário | Descrição |
|----------|-----------|
| `green` | Frota operando normalmente |
| `gearbox` | Degradação da caixa multiplicadora |
| `pitch` | Falha no sistema de pitch |
| `yaw` | Desalinhamento do sistema de yaw |
| `mixed` | Múltiplas falhas simultâneas |
| `red` | Degradação severa da frota |

| Falha | Risco predominante | Comportamento típico |
|--------|--------------------|----------------------|
| Gearbox degradation | Mecânico | Elevada perda de potência acompanhada pelo aumento da temperatura da caixa multiplicadora |
| Pitch malfunction | Aerodinâmico | Redução da geração de energia mesmo com condições normais de vento |
| Yaw misalignment | Aerodinâmico | Perda persistente de potência causada pelo desalinhamento do rotor |
| Sensor drift | Anomalia | Comportamento anormal dos sensores detectado pelo Isolation Forest |

---

# Primeiros Passos

## Requisitos

- Python 3.10 ou superior.

Instale as dependências do projeto:

```bash
pip install -r requirements.txt
```

Crie um arquivo `.env`:

```text
ANTHROPIC_API_KEY=your_api_key
LIVE_MODE_PASSWORD=your_password
```

Sem uma chave da API da Anthropic, a aplicação é executada integralmente em **Demo Mode**, utilizando o mecanismo determinístico de diagnóstico baseado em regras incorporado ao projeto.

Mesmo com uma `ANTHROPIC_API_KEY` válida, a aplicação Streamlit sempre inicia em **Demo Mode**. Para habilitar chamadas reais à API, é necessário desbloquear manualmente o **Live Mode** na barra lateral utilizando `LIVE_MODE_PASSWORD`, evitando o consumo acidental de tokens.

---

# Dashboard do Streamlit

Inicie o dashboard:

```bash
streamlit run app/app.py
```

O dashboard oferece:

- Visão geral da frota.
- Inspeção individual das turbinas.
- Recomendações de manutenção geradas por IA.
- Seção expansível **Why**, que explica o raciocínio por trás de cada plano de ação.
- Registro completo da execução do agente.
- Exportação de relatórios em PDF.
- Indicador do modo de execução (Demo Mode / Live Mode).

---

# Interface de Linha de Comando

Execute o cenário padrão:

```bash
python -m app.cli
```

Execute um cenário diferente:

```bash
python -m app.cli --scenario red
```

Analise cinco turbinas:

```bash
python -m app.cli --top-n 5
```

Exporte os relatórios:

```bash
python -m app.cli --scenario mixed --top-n 5 --export
```

Execute apenas o pipeline analítico:

```bash
python -m app.cli --no-agent
```

Exiba a ajuda:

```bash
python -m app.cli --help
```

> **Observação**
>
> A CLI deve ser executada como um módulo Python (`python -m app.cli`), e não como um script. Dessa forma, a estrutura do pacote é preservada e problemas de importação relacionados ao `sys.path` são evitados.

---

# Notebooks

O projeto inclui três notebooks que documentam todo o processo de desenvolvimento.

| Notebook | Objetivo |
|-----------|----------|
| **01** | Análise exploratória dos dados SCADA sintéticos |
| **02** | Engenharia de atributos, detecção de anomalias e avaliação híbrida de risco |
| **03** | Fluxo de trabalho do agente LangGraph e geração de recomendações de manutenção |

Recomenda-se executá-los em sequência para acompanhar todas as etapas do desenvolvimento do projeto.

---

# Avaliação Híbrida de Risco

A pontuação final de risco de cada turbina é calculada a partir da combinação de três subíndices interpretáveis.

| Subíndice | Peso | Principais indicadores |
|------------|------|------------------------|
| Aerodinâmico | 50% | Power gap, yaw proxy e instabilidade do pitch |
| Mecânico | 30% | Temperatura do óleo da caixa multiplicadora e tendência de vibração |
| Anomalias | 20% | Pontuação do Isolation Forest e persistência das anomalias |

Cada subíndice combina limites de engenharia com o comportamento relativo da frota:

```text
Risk = 0.70 × Absolute Component
     + 0.30 × Relative Component
```

A prioridade final de manutenção combina:

- **50% Pontuação de risco**
- **35% Perda estimada de energia**
- **15% Criticidade do ativo**

Todos esses pesos podem ser configurados em `src/config.py`.

---

# Agente LangGraph

O copiloto de IA segue um fluxo de trabalho do tipo ReAct implementado com LangGraph.

Ferramentas disponíveis:

- `get_priority_ranking`
- `get_turbine_details`
- `submit_action_plan`

A aplicação seleciona automaticamente o modo de execução mais adequado:

| Modo | Comportamento |
|------|---------------|
| **Live Mode** | Utiliza o Claude por meio da API da Anthropic |
| **Demo Mode** | Utiliza o mecanismo determinístico de diagnóstico incorporado ao projeto |

Se a API da Anthropic estiver indisponível ou ocorrer uma falha de autenticação, a aplicação muda automaticamente para o **Demo Mode**, preservando exatamente o mesmo fluxo de trabalho e o mesmo formato de saída.

Cada recomendação de manutenção inclui os subíndices de risco que a sustentam, os principais sinais operacionais, as chamadas às ferramentas e o registro completo da execução, permitindo entender como cada recomendação foi gerada.

---

# Exemplo de Plano de Ação

O agente de IA gera recomendações estruturadas de manutenção em formato JSON.

```json
{
  "turbine_id": "WTG-02",
  "urgency": "high",
  "fault_hypothesis": "Degradação da caixa multiplicadora ou do trem de potência",
  "recommended_action": "1. Programar uma coleta de óleo da caixa multiplicadora e uma análise de vibração nas próximas 24 horas. 2. Monitorar a temperatura da caixa multiplicadora durante o próximo ciclo de operação. 3. Inspecionar o sistema de lubrificação e os rolamentos na próxima janela de manutenção.",
  "rationale": "O risco mecânico é predominante devido ao aumento da temperatura do óleo da caixa multiplicadora e à perda sustentada de potência, indicando uma provável degradação do trem de potência."
}
```

As recomendações são exibidas no dashboard do Streamlit e também podem ser exportadas em formatos JSON, CSV e PDF.

---

# Demo Mode vs Live Mode

| Funcionalidade | Demo Mode | Live Mode |
|----------------|-----------|-----------|
| Pipeline analítico | ✅ | ✅ |
| Avaliação de risco | ✅ | ✅ |
| Priorização | ✅ | ✅ |
| Dashboard | ✅ | ✅ |
| Exportação em PDF | ✅ | ✅ |
| Registro do agente | ✅ | ✅ |
| API do Claude | ❌ | ✅ |

O **Demo Mode** utiliza o mecanismo determinístico de diagnóstico baseado em regras incluído no projeto.

O **Live Mode** substitui esse mecanismo pelo Claude por meio da API da Anthropic, mantendo o mesmo fluxo de trabalho e a mesma estrutura de saída.

---
# Limitações Conhecidas

## Dados

- Os dados SCADA são gerados de forma sintética.
- O comportamento dos sinais é fisicamente plausível, mas não representa toda a complexidade de um parque eólico real.
- Atualmente, o projeto simula apenas um modelo genérico de turbina eólica de 3 MW.

## Análise

- Os limites utilizados na avaliação de risco são heurísticas de engenharia, e não parâmetros aprendidos a partir de falhas históricas.
- O Isolation Forest é treinado e avaliado utilizando o mesmo conjunto de dados sintéticos.
- O histórico de manutenção ainda não é incorporado ao processo de priorização.

## Agente de IA

- O Live Mode requer uma chave válida da API da Anthropic.
- O Live Mode consome créditos da API da Anthropic.
- O dashboard do Streamlit sempre inicia em **Demo Mode** e exige `LIVE_MODE_PASSWORD` para desbloquear o Live Mode.
- **A CLI ainda não aplica essa proteção.** Se uma chave válida da API da Anthropic for detectada, ela será executada automaticamente em Live Mode. Alinhar esse comportamento ao dashboard é uma melhoria prevista para as próximas versões.
- As recomendações foram desenvolvidas para apoiar a tomada de decisão em engenharia, e não para substituir o julgamento técnico dos engenheiros.
- Atualmente não há integração com plataformas CMMS (SAP PM, IBM Maximo, etc.).

---

# Stack Tecnológico

| Camada | Tecnologia |
|---------|------------|
| Linguagem | Python |
| Processamento de Dados | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Detecção de Anomalias | Isolation Forest |
| Framework de IA | LangGraph, LangChain |
| LLM | Claude (Anthropic) |
| Dashboard | Streamlit |
| Visualização | Matplotlib, Seaborn |
| Relatórios | JSON, CSV, PDF |
| Ambiente | python-dotenv |
| Desenvolvimento | Jupyter Notebook |

---

# Próximas Melhorias

Os próximos marcos planejados para o projeto são:

- Aplicar a mesma proteção por `LIVE_MODE_PASSWORD` à CLI.
- Implementar a ingestão de dados SCADA reais por meio de `src/ingesta.py`.
- Validar o pipeline utilizando o conjunto de dados do Kelmarsh Wind Farm.
- Adicionar fallback automático para o DeepSeek quando a Anthropic não estiver disponível.
- Ampliar a cobertura de testes automatizados com `pytest`.
- Persistir os conjuntos de dados simulados para evitar a regeneração dos cenários a cada execução.
- Publicar a aplicação no Streamlit Community Cloud.
- Gravar um vídeo de demonstração completo apresentando o dashboard, o pipeline analítico e o fluxo de trabalho do AI Copilot.

---

# Autor

**Adrián Rodríguez Estévez**

Engenheiro em transição para as áreas de Data Science e AI Engineering, com experiência profissional em instalação, comissionamento de parques eólicos e gerenciamento de projetos na Europa e na América Latina.

Este projeto reúne experiência no setor de energias renováveis com Data Science, Machine Learning e IA generativa para construir um fluxo de manutenção preditiva explicável para frotas de turbinas eólicas.

**Vamos nos conectar**

- GitHub: https://github.com/AdriARE
- LinkedIn: https://www.linkedin.com/in/arestevez/

---

# Licença

Este projeto foi desenvolvido para fins educacionais e de portfólio.

Sinta-se à vontade para explorar o código, reproduzir as análises e utilizar o projeto como referência para estudos, aprendizado ou pesquisa.
