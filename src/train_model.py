import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import os

# --- CONFIGURAÇÃO ---
CSV_FILE = "../data/simulated_sensor_data.csv"

# Variáveis Alvo para Regressão:
TARGETS = {
    'yield_ton_per_ha': "Estimativa de Rendimento (ton/ha)",
    'irrigation_ml': "Sugestão de Volume de Irrigação (ml)",
    'fertilizer_kg': "Sugestão de Fertilização (kg)"
}

# Modelos a serem comparados
MODELS = {
    'Regressão Linear Múltipla': LinearRegression(),
    'Regressão Ridge': Ridge(alpha=1.0, random_state=42),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
}
# --- FIM CONFIGURAÇÃO ---

def load_data():
    """Carrega e prepara os dados."""
    if not os.path.exists(CSV_FILE):
        print(f"ERRO: Arquivo {CSV_FILE} não encontrado. Abortando a análise.")
        sys.exit(1)
        
    df = pd.read_csv(CSV_FILE)
    
    # Colunas a serem usadas como features preditoras (sem o alvo atual)
    FEATURES = ['soil_moisture', 'soil_ph', 'air_temp', 'humidity']
    
    # Remove colunas não necessárias e linhas com valores ausentes
    df = df.drop(columns=['sensor_id', 'timestamp'], errors='ignore')
    df = df.dropna()

    return df, FEATURES

def evaluate_model(y_test, y_pred):
    """Calcula e retorna as métricas de avaliação."""
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

def run_predictive_analysis(df, features):
    """Treina e avalia todos os modelos para todas as variáveis alvo."""
    
    results = {}
    
    for target_name, target_desc in TARGETS.items():
        print(f"\n[ANALISANDO ALVO: {target_desc}]")
        
        # Define X e Y para o alvo atual
        X = df[features]
        y = df[target_name]
        
        # Divisão dos dados
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        target_results = {}
        
        for model_name, model in MODELS.items():
            
            # Treinamento
            model.fit(X_train, y_train)
            
            # Previsão
            y_pred = model.predict(X_test)
            
            # Avaliação
            metrics = evaluate_model(y_test, y_pred)
            target_results[model_name] = metrics
            
            print(f"  > Modelo {model_name}: R² = {metrics['R2']:.4f}, RMSE = {metrics['RMSE']:.4f}")
            
        results[target_name] = target_results
        
    return results

def present_recommendations(results):
    """Interpreta os resultados e sugere ações de manejo."""
    
    print("\n\n" + "="*50)
    print("      RECOMENDAÇÕES DE MANEJO BASEADAS EM REGRESSÃO")
    print("="*50)
    
    # Encontrando o melhor modelo para prever Rendimento (yield_ton_per_ha)
    yield_results = results['yield_ton_per_ha']
    best_model_name = max(yield_results, key=lambda name: yield_results[name]['R2'])
    best_r2 = yield_results[best_model_name]['R2']
    
    print(f"\n[Foco Principal: Rendimento Estimado]")
    print(f"O modelo mais eficiente para prever o Rendimento (R²) é o: {best_model_name} (R²: {best_r2:.4f})")
    
    if best_r2 < 0.70:
        print("\nRECOMENDAÇÃO 1 (MODELAGEM): O R² está baixo (< 0.70). Sugere-se adicionar mais features, como o Índice NPK total e interações entre umidade e temperatura do ar, para melhorar a previsão de rendimento.")
    else:
        print("\nRECOMENDAÇÃO 1 (MODELAGEM): O R² está satisfatório. O modelo Random Forest é frequentemente melhor para dados não lineares.")
        
    
    # Sugestões de Irrigação (Foco no MAE)
    irrigation_results = results['irrigation_ml']
    best_mae_irrigation_name = min(irrigation_results, key=lambda name: irrigation_results[name]['MAE'])
    best_mae_irrigation = irrigation_results[best_mae_irrigation_name]['MAE']
    
    print(f"\n[Foco Secundário: Volume de Irrigação Sugerido]")
    print(f"O melhor modelo para prever a Irrigação (menor MAE) é o: {best_mae_irrigation_name} (MAE: {best_mae_irrigation:.2f} ml)")
    print(f"RECOMENDAÇÃO 2 (IRRIGAÇÃO): O erro médio absoluto (MAE) é de {best_mae_irrigation:.2f} ml. Isso significa que, em média, a sugestão do modelo pode ter um erro de {best_mae_irrigation:.2f} ml no volume ideal. Isso é aceitável para um controle automatizado, mas requer calibração no campo.")
    
    # Sugestões de Fertilização (Foco no MAE)
    fertilizer_results = results['fertilizer_kg']
    best_mae_fertilizer_name = min(fertilizer_results, key=lambda name: fertilizer_results[name]['MAE'])
    best_mae_fertilizer = fertilizer_results[best_mae_fertilizer_name]['MAE']

    print(f"\n[Foco Secundário: Necessidade de Fertilização Sugerida]")
    print(f"O melhor modelo para prever a Fertilização (menor MAE) é o: {best_mae_fertilizer_name} (MAE: {best_mae_fertilizer:.4f} kg)")
    print(f"RECOMENDAÇÃO 3 (MANEJO): A precisão na predição de fertilizantes é vital. Um MAE de {best_mae_fertilizer:.4f} kg indica que o modelo tem alta precisão para sugerir o volume de fertilizante, auxiliando na otimização de custos e na prevenção de excesso de nutrientes.")

    
    # Apresentação tabular dos resultados
    print("\n\n[AVALIAÇÃO COMPARATIVA DETALHADA - R²]")
    data_r2 = {target: {model: metrics['R2'] for model, metrics in res.items()} for target, res in results.items()}
    df_r2 = pd.DataFrame(data_r2).transpose().rename(columns=TARGETS)
    print(df_r2)

    print("\n[AVALIAÇÃO COMPARATIVA DETALHADA - RMSE]")
    data_rmse = {target: {model: metrics['RMSE'] for model, metrics in res.items()} for target, res in results.items()}
    df_rmse = pd.DataFrame(data_rmse).transpose().rename(columns=TARGETS)
    print(df_rmse)


if __name__ == "__main__":
    df, features = load_data()
    
    if not df.empty:
        results = run_predictive_analysis(df, features)
        present_recommendations(results)
    else:
        print("Análise abortada devido à falha no carregamento dos dados.")