# FarmTech MVP -- Sistema de Monitoramento e Inteligência Agrícola

Um MVP completo para monitoramento de sensores agrícolas, registrar
irrigação, fertilização e gerar previsões de produtividade usando
Machine Learning.

------------------------------------------------------------------------

## 🚀 Configuração em 5 Minutos

### 1. Criar Banco de Dados & Estrutura

Entre na pasta **backend**:

``` bash
cd backend
pip install -r requirements.txt
```

Inicialize o banco:

``` bash
python init_db.py
```

Isso criará o arquivo **farmtech.db** com a tabela `sensors`.

------------------------------------------------------------------------

## 2. Iniciar o Backend

Dentro da pasta **backend**:

``` bash
python app.py
```

O servidor ficará disponível em:

    http://localhost:5000

------------------------------------------------------------------------

## 3. Abrir o Frontend

Em outra janela do terminal:

``` bash
cd frontend
python -m http.server 8000
```

Ou simplesmente abra o arquivo **index.html** no navegador.

Acesse:

    http://localhost:8000

------------------------------------------------------------------------

## 📸 Como Demonstrar (Passo a Passo)

### 1. Enviar Dados de Sensores

Use qualquer cliente HTTP (Thunder Client, Insomnia, cURL):

``` json
POST /sensor-data
{
  "sensor_id": "sensor_01",
  "soil_moisture": 30,
  "soil_ph": 6.5,
  "air_temp": 28,
  "humidity": 70,
  "irrigation_ml": 200,
  "fertilizer_kg": 0.5
}
```

O backend salva automaticamente no banco.

------------------------------------------------------------------------

### 2. Ver Histórico no Dashboard

Abra o frontend → ele carrega:

-   Umidade do solo em tempo real\
-   pH\
-   Temperatura\
-   Irrigação aplicada\
-   Fertilização\
-   Tabela com histórico completo

------------------------------------------------------------------------

### 3. Prever Produtividade (Machine Learning)

O backend treina o modelo automaticamente quando há dados suficientes.

Faça:

    GET /predict?sensor_id=sensor_01

Se existir histórico, retorna:

``` json
{
  "predicted_yield": 3.87
}
```

------------------------------------------------------------------------

## 🐛 Troubleshooting

### Backend não inicia

Certifique‑se de instalar:

``` bash
pip install -r requirements.txt
```

E se o banco estiver faltando:

``` bash
python init_db.py
```

### Dashboard não carrega dados

Abra no navegador o console (F12 → Console).

Erros comuns:

-   CORS bloqueando → certifique‑se de que `flask-cors` está instalado.
-   backend parado → cheque http://localhost:5000/health
-   frontend fora do ar → abra o index.html diretamente.

------------------------------------------------------------------------

## 📊 Estrutura do Projeto

    farmtech/
    ├── backend/
    │   ├── app.py              ← API principal
    │   ├── init_db.py          ← Criação do banco de dados
    │   ├── model.py            ← Treinamento de Machine Learning (RandomForest)
    │   ├── db.sqlite           ← Banco de dados
    │   ├── requirements.txt    ← Dependências Python
    │   └── utils/              ← Funções auxiliares
    ├── frontend/
    │   ├── index.html          ← Interface do Dashboard
    │   ├── style.css           ← Tema visual
    │   ├── app.js              ← Lógica do dashboard
    └── README.md               ← Este arquivo

------------------------------------------------------------------------

## 🔑 Principais Endpoints

### Registrar dados de sensores

`POST /sensor-data`

### Listar histórico

`GET /history?sensor_id=XYZ`

### Prever produtividade

`GET /predict?sensor_id=XYZ`

### Checar status do sistema

`GET /health`

------------------------------------------------------------------------

## 🌱 Sobre o Projeto

Este MVP simula um sistema completo de agricultura inteligente,
integrando:

-   Telemetria de sensores
-   Banco de dados completo
-   Dashboard web
-   API em Flask
-   Modelo de Machine Learning
-   Previsões automáticas

Serve como base sólida para projetos FIAP, TCCs e demonstrações
profissionais.

------------------------------------------------------------------------

Pronto! Agora o projeto está documentado com clareza e pronto para
apresentação.
