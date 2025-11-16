import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import joblib  # For saving and loading models
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'  
plt.rcParams['mathtext.fontset'] = 'stix'  
plt.rcParams['axes.unicode_minus'] = False  

class NPPVPredictor:
    def __init__(self, data_path=None):
        """
        Initialize NPPV predictor
        """
        if data_path:
            self.data = pd.read_excel(data_path)
        else:
            self.data = None
            
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = ['thetao', 'kd', 'chl', 'so', 'spco2']  # Removed sithick
        self.target_name = 'nppv'
        
    def preprocess_data(self):
        """
        Data preprocessing
        """
        print("Starting data preprocessing...")
        
        # Ensure all columns exist
        for col in self.feature_names + [self.target_name]:
            if col not in self.data.columns:
                print(f"Warning: Column {col} does not exist in the data")
                return False
        
        # Extract features and target
        self.X = self.data[self.feature_names]
        self.y = self.data[self.target_name]
        
        print(f"Data shape: {self.X.shape}")
        print(f"Target variable shape: {self.y.shape}")
        
        # Check for missing values
        print(f"Feature missing values: {self.X.isnull().sum().sum()}")
        print(f"Target variable missing values: {self.y.isnull().sum()}")
        
        return True
    
    def prepare_features(self):
        """
        Prepare feature data
        """
        # Split training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Standardize features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
    
    def initialize_models(self):
        """
        Initialize multiple machine learning models
        """
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Support Vector Regression': SVR(kernel='rbf', C=1.0),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        }
    
    def train_and_evaluate_models(self):
        """
        Train and evaluate all models
        """
        print("\nStarting model training and evaluation...")
        print("="*50)
        
        feature_names = self.X.columns.tolist()
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            

            X_train_used = self.X_train
            X_test_used = self.X_test
            
            # Train model
            model.fit(X_train_used, self.y_train)
            
            # Predict
            y_pred = model.predict(X_test_used)
            
            # Calculate evaluation metrics
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_used, self.y_train, 
                                      cv=5, scoring='r2')
            
            # Store results
            self.results[name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'uses_scaled_data': name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Support Vector Regression', 'Neural Network']
            }
            
            # Print results
            print(f"{name} results:")
            print(f"  R²: {r2:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  Cross-validation R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            
            # For linear models, print coefficients
            if hasattr(model, 'coef_'):
                coefficients = model.coef_
                print(f"  Feature coefficients:")
                for i, (feature, coef) in enumerate(zip(feature_names, coefficients)):
                    print(f"    {feature}: {coef:.6f}")
    
    def find_best_model(self):
        """
        Find the best model
        """
        best_model_name = None
        best_r2 = -float('inf')
        
        for name, result in self.results.items():
            if result['r2'] > best_r2:
                best_r2 = result['r2']
                best_model_name = name
        
        self.best_model_name = best_model_name
        self.best_model = self.results[best_model_name]['model']
        self.best_model_uses_scaled = self.results[best_model_name]['uses_scaled_data']
        
        return best_model_name, self.results[best_model_name]
    
    def predict(self, new_data):
        """
        Make predictions using the best model
        
        Parameters:
        new_data: New feature data, DataFrame or numpy array
        
        Returns:
        predictions: Prediction results
        """
        if self.best_model is None:
            print("Error: Please train the model first")
            return None
        
        # Ensure input data format is correct
        if isinstance(new_data, pd.DataFrame):
            # Check if feature columns match
            if not all(col in new_data.columns for col in self.feature_names):
                print(f"Error: Input data must contain the following features: {self.feature_names}")
                return None
            X_new = new_data[self.feature_names]
        else:
            # Assume it's a numpy array
            X_new = new_data
        
        # Decide whether to standardize based on model type
        if self.best_model_uses_scaled:
            X_new_processed = self.scaler.transform(X_new)
        else:
            X_new_processed = X_new
        
        # Make predictions
        predictions = self.best_model.predict(X_new_processed)
        
        return predictions
    
    def predict_single(self, thetao, kd, chl, so, spco2):
        """
        Predict NPPV value for a single sample
        
        Parameters:
        thetao, kd, chl, so, spco2: Environmental variable values
        
        Returns:
        prediction: Predicted NPPV value
        """
        # Create input data
        input_data = pd.DataFrame({
            'thetao': [thetao],
            'kd': [kd],
            'chl': [chl],
            'so': [so],
            'spco2': [spco2]
        })
        
        return self.predict(input_data)[0]
    
    def save_model(self, filepath='nppv_predictor_model.pkl'):
        """
        Save the trained model and scaler
        
        Parameters:
        filepath: Save path
        """
        if self.best_model is None:
            print("Error: No trained model to save")
            return False
        
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'best_model_uses_scaled': self.best_model_uses_scaled,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'target_name': self.target_name
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to: {filepath}")
        return True
    
    def load_model(self, filepath='nppv_predictor_model.pkl'):
        """
        Load a saved model
        
        Parameters:
        filepath: Model file path
        """
        try:
            model_data = joblib.load(filepath)
            
            self.best_model = model_data['best_model']
            self.best_model_name = model_data['best_model_name']
            self.best_model_uses_scaled = model_data['best_model_uses_scaled']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.target_name = model_data['target_name']
            
            print(f"Model loaded from {filepath}")
            print(f"Best model: {self.best_model_name}")
            return True
            
        except FileNotFoundError:
            print(f"Error: Model file {filepath} not found")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def plot_results(self):
        """
        Plot result charts
        """
        # 创建第一个图：模型性能比较
        fig1, axes1 = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. 模型性能比较
        models_names = list(self.results.keys())
        r2_scores = [self.results[name]['r2'] for name in models_names]
        rmse_scores = [self.results[name]['rmse'] for name in models_names]
        
        axes1[0].bar(models_names, r2_scores, color='skyblue')
        axes1[0].set_title('Model R² Score Comparison')
        axes1[0].set_ylabel('R² Score')
        axes1[0].tick_params(axis='x', rotation=45)
        
        axes1[1].bar(models_names, rmse_scores, color='lightcoral')
        axes1[1].set_title('Model RMSE Comparison')
        axes1[1].set_ylabel('RMSE')
        axes1[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # 创建第二个图：所有模型的预测值与实际值对比
        fig2, axes2 = plt.subplots(2, 4, figsize=(20, 10))
        axes2 = axes2.flatten()
        
        # 为每个模型绘制预测值与实际值对比图
        for i, (name, result) in enumerate(self.results.items()):
            y_pred = result['predictions']
            
            axes2[i].scatter(self.y_test, y_pred, alpha=0.6)
            axes2[i].plot([self.y_test.min(), self.y_test.max()], 
                         [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            axes2[i].set_xlabel('Actual Values')
            axes2[i].set_ylabel('Predicted Values')
            axes2[i].set_title(f'{name}\nR² = {result["r2"]:.4f}, RMSE = {result["rmse"]:.4f}')
            
            # 添加对角线参考线
            axes2[i].grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(self.results), len(axes2)):
            axes2[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # 创建第三个图：特征重要性
        fig3, axes3 = plt.subplots(2, 2, figsize=(15, 12))
        axes3 = axes3.flatten()
        
        # 为有特征重要性的模型绘制特征重要性图
        feature_importance_plots = 0
        
        # Random Forest 特征重要性
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']
            feature_importance = rf_model.feature_importances_
            feature_names = self.X.columns
            
            indices = np.argsort(feature_importance)[::-1]
            
            axes3[feature_importance_plots].bar(range(len(feature_importance)), 
                          feature_importance[indices])
            axes3[feature_importance_plots].set_title('Random Forest Feature Importance')
            axes3[feature_importance_plots].set_xticks(range(len(feature_importance)))
            axes3[feature_importance_plots].set_xticklabels([feature_names[i] for i in indices], rotation=45)
            feature_importance_plots += 1
        
        # Gradient Boosting 特征重要性
        if 'Gradient Boosting' in self.models:
            gb_model = self.models['Gradient Boosting']
            feature_importance = gb_model.feature_importances_
            feature_names = self.X.columns
            
            indices = np.argsort(feature_importance)[::-1]
            
            axes3[feature_importance_plots].bar(range(len(feature_importance)), 
                          feature_importance[indices], color='orange')
            axes3[feature_importance_plots].set_title('Gradient Boosting Feature Importance')
            axes3[feature_importance_plots].set_xticks(range(len(feature_importance)))
            axes3[feature_importance_plots].set_xticklabels([feature_names[i] for i in indices], rotation=45)
            feature_importance_plots += 1
        
        # 线性模型系数（作为特征重要性）
        linear_models = ['Linear Regression', 'Ridge Regression', 'Lasso Regression']
        for model_name in linear_models:
            if model_name in self.models and feature_importance_plots < len(axes3):
                model = self.models[model_name]
                if hasattr(model, 'coef_'):
                    coefficients = np.abs(model.coef_)
                    feature_names = self.X.columns
                    
                    indices = np.argsort(coefficients)[::-1]
                    
                    axes3[feature_importance_plots].bar(range(len(coefficients)), 
                                  coefficients[indices], color='green')
                    axes3[feature_importance_plots].set_title(f'{model_name} Coefficients (abs)')
                    axes3[feature_importance_plots].set_xticks(range(len(coefficients)))
                    axes3[feature_importance_plots].set_xticklabels([feature_names[i] for i in indices], rotation=45)
                    feature_importance_plots += 1
        
        # 隐藏多余的子图
        for i in range(feature_importance_plots, len(axes3)):
            axes3[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def print_relationship_equations(self):
        """
        Print relationship equations
        """
        print("\n" + "="*60)
        print("NPPV Relationship Equations with Environmental Variables")
        print("="*60)
        
        feature_names = self.X.columns.tolist()
        
        for name, result in self.results.items():
            model = result['model']
            
            print(f"\n{name}:")
            print(f"R²: {result['r2']:.4f}")
            
            if hasattr(model, 'intercept_') and hasattr(model, 'coef_'):
                # Linear model equations
                equation = f"nppv = {model.intercept_:.6f}"
                for i, (feature, coef) in enumerate(zip(feature_names, model.coef_)):
                    sign = '+' if coef >= 0 else '-'
                    equation += f" {sign} {abs(coef):.6f}×{feature}"
                print(f"Equation: {equation}")
            
            elif hasattr(model, 'feature_importances_'):
                # Tree model feature importance
                print("Feature Importance:")
                importances = model.feature_importances_
                for feature, importance in zip(feature_names, importances):
                    print(f"  {feature}: {importance:.4f}")
    
    def run_complete_analysis(self):
        """
        Run complete analysis pipeline
        """
        print("NPPV and Environmental Variables Relationship Analysis")
        print("="*50)
        
        # Data preprocessing
        if not self.preprocess_data():
            print("Data preprocessing failed, please check data file")
            return
        
        # Prepare features
        self.prepare_features()
        
        # Initialize models
        self.initialize_models()
        
        # Train and evaluate models
        self.train_and_evaluate_models()
        
        # Find best model
        best_name, best_result = self.find_best_model()
        print(f"\n{'='*50}")
        print(f"Best Model: {best_name}")
        print(f"Best R² Score: {best_result['r2']:.4f}")
        print(f"Best RMSE: {best_result['rmse']:.4f}")
        print(f"{'='*50}")
        
        # Plot results
        self.plot_results()
        
        # Print equations
        self.print_relationship_equations()
        
        # Save model
        self.save_model()
        
        return best_name, best_result
    
    def interactive_prediction(self):
        """
        Interactive prediction function: provide predictions based on user input
        """
        if self.best_model is None:
            print("Error: No trained model, please run complete analysis or load a model first")
            return
        
        print("\n" + "="*50)
        print("Interactive NPPV Prediction")
        print("="*50)
        print("Please enter values for the following environmental variables:")
        
        while True:
            try:
                print("\n--- Please enter environmental variable values ---")
                thetao = float(input("Sea temperature (thetao): "))
                kd = float(input("Attenuation coefficient (kd): "))
                chl = float(input("Chlorophyll concentration (chl): "))
                so = float(input("Salinity (so): "))
                spco2 = float(input("Surface CO2 partial pressure (spco2): "))
                
                # Make prediction
                prediction = self.predict_single(thetao, kd, chl, so, spco2)
                
                print(f"\nPrediction result: NPPV = {prediction:.4f}")
                print(f"Model used: {self.best_model_name}")
                
                # Ask if continue prediction
                continue_pred = input("\nContinue prediction? (y/n): ").strip().lower()
                if continue_pred != 'y':
                    print("Exiting interactive prediction")
                    break
                    
            except ValueError:
                print("Error: Please enter valid numbers")
            except Exception as e:
                print(f"Error during prediction: {e}")

# Usage example
if __name__ == "__main__":
    # Initialize predictor
    predictor = NPPVPredictor('precise_bohai_sea_data.xlsx')
    
    # Run complete analysis
    best_model, best_results = predictor.run_complete_analysis()
    
    # Additional analysis: correlation matrix
    print("\n" + "="*50)
    print("Environmental Variables Correlation Analysis")
    print("="*50)
    
    # Calculate correlation matrix
    correlation_matrix = predictor.data[['nppv', 'thetao', 'kd', 'chl', 'so', 'spco2']].corr()  # Removed sithick
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.3f')
    plt.title('Environmental Variables Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Print correlations with NPPV
    nppv_correlations = correlation_matrix['nppv'].sort_values(ascending=False)
    print("\nCorrelations with NPPV:")
    for var, corr in nppv_correlations.items():
        print(f"  {var}: {corr:.4f}")
    
    # Demonstrate prediction function
    print("\n" + "="*50)
    print("Prediction Function Demonstration")
    print("="*50)
    
    # Use test set data for prediction demonstration
    test_sample = predictor.X_test.iloc[:5]  # Take first 5 test samples
    predictions = predictor.predict(test_sample)
    actual_values = predictor.y_test.iloc[:5].values
    
    print("Test sample prediction results:")
    for i, (pred, actual) in enumerate(zip(predictions, actual_values)):
        print(f"Sample {i+1}: Predicted = {pred:.4f}, Actual = {actual:.4f}, Error = {abs(pred-actual):.4f}")
    
    # Demonstrate single sample prediction
    print("\nSingle sample prediction demonstration:")
    single_pred = predictor.predict_single(
        thetao=15.5, 
        kd=0.1, 
        chl=0.5, 
        so=34.5, 
        spco2=380
    )
    print(f"Single sample prediction result: {single_pred:.4f}")
    
    # Demonstrate model save and load
    print("\nModel save and load demonstration:")
    
    # Create new predictor instance and load model
    new_predictor = NPPVPredictor()
    if new_predictor.load_model():
        # Use loaded model for prediction
        loaded_predictions = new_predictor.predict(test_sample)
        print("Using loaded model for prediction:")
        for i, (pred, actual) in enumerate(zip(loaded_predictions, actual_values)):
            print(f"Sample {i+1}: Predicted = {pred:.4f}, Actual = {actual:.4f}")
    
    # Start interactive prediction
    print("\n" + "="*50)
    print("Starting Interactive Prediction")
    print("="*50)
    predictor.interactive_prediction()