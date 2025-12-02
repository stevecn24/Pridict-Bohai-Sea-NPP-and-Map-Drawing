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

# 设置全局字体参数，确保字符正确显示 - 移除不存在的字体
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.size'] = 12

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
            
            # Determine whether to use standardized data
            if name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Support Vector Regression', 'Neural Network']:
                X_train_used = self.X_train_scaled
                X_test_used = self.X_test_scaled
            else:
                X_train_used = self.X_train
                X_test_used = self.X_test
            
            # Train model
            model.fit(X_train_used, self.y_train)
            
            # Predict on training set and test set
            y_train_pred = model.predict(X_train_used)
            y_test_pred = model.predict(X_test_used)
            
            # Calculate training set evaluation metrics
            train_mse = mean_squared_error(self.y_train, y_train_pred)
            train_rmse = np.sqrt(train_mse)
            train_mae = mean_absolute_error(self.y_train, y_train_pred)
            train_r2 = r2_score(self.y_train, y_train_pred)
            
            # Calculate test set evaluation metrics
            test_mse = mean_squared_error(self.y_test, y_test_pred)
            test_rmse = np.sqrt(test_mse)
            test_mae = mean_absolute_error(self.y_test, y_test_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_used, self.y_train, 
                                      cv=5, scoring='r2')
            
            # Store results
            self.results[name] = {
                'model': model,
                'train_mse': train_mse,
                'train_rmse': train_rmse,
                'train_mae': train_mae,
                'train_r2': train_r2,
                'test_mse': test_mse,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_r2': test_r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'train_predictions': y_train_pred,
                'test_predictions': y_test_pred,
                'uses_scaled_data': name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression', 'Support Vector Regression', 'Neural Network']
            }
            
            # Print results
            print(f"{name} results:")
            print(f"  Training Set - R²: {train_r2:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
            print(f"  Test Set - R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")
            print(f"  Cross-validation R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            
            # For linear models, print coefficients
            if hasattr(model, 'coef_'):
                coefficients = model.coef_
                print(f"  Feature coefficients:")
                for i, (feature, coef) in enumerate(zip(feature_names, coefficients)):
                    print(f"    {feature}: {coef:.6f}")
    
    def find_best_model(self):
        """
        Find the best model based on test set R²
        """
        best_model_name = None
        best_r2 = -float('inf')
        
        for name, result in self.results.items():
            if result['test_r2'] > best_r2:
                best_r2 = result['test_r2']
                best_model_name = name
        
        self.best_model_name = best_model_name
        self.best_model = self.results[best_model_name]['model']
        self.best_model_uses_scaled = self.results[best_model_name]['uses_scaled_data']
        
        return best_model_name, self.results[best_model_name]
    
    def print_detailed_metrics(self):
        """
        Print detailed metrics for training and test sets
        """
        print("\n" + "="*80)
        print("DETAILED PERFORMANCE METRICS FOR ALL MODELS")
        print("="*80)
        
        # Create header
        header = f"{'Model':<25} {'Set':<10} {'R²':<10} {'RMSE':<12} {'MAE':<12} {'MSE':<12}"
        print(header)
        print("-" * 80)
        
        for name, result in self.results.items():
            # Training set metrics
            train_line = f"{name:<25} {'Train':<10} {result['train_r2']:<10.4f} {result['train_rmse']:<12.4f} {result['train_mae']:<12.4f} {result['train_mse']:<12.4f}"
            print(train_line)
            
            # Test set metrics
            test_line = f"{'':<25} {'Test':<10} {result['test_r2']:<10.4f} {result['test_rmse']:<12.4f} {result['test_mae']:<12.4f} {result['test_mse']:<12.4f}"
            print(test_line)
            
            # Cross-validation - 修复了格式化错误
            cv_mean_str = f"{result['cv_mean']:.4f}"
            cv_std_str = f"{result['cv_std']:.4f}"
            cv_line = f"{'':<25} {'CV-R²':<10} {cv_mean_str} ± {cv_std_str:<12}"
            print(cv_line)
            
            print("-" * 80)
    
    def plot_comparison_metrics(self):
        """
        Plot comparison charts for training and test set metrics - 优化版本
        """
        # 设置图表字体和大小 - 移除不存在的字体
        plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.size'] = 10  # 减小字体大小
        
        # 创建更大的图形，增加高度以容纳标签
        fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=600)
        
        models_names = list(self.results.keys())
        
        # 缩短模型名称用于显示
        short_names = {
            'Linear Regression': 'Linear',
            'Ridge Regression': 'Ridge',
            'Lasso Regression': 'Lasso', 
            'Random Forest': 'RF',
            'Gradient Boosting': 'GB',
            'Support Vector Regression': 'SVR',
            'Neural Network': 'NN'
        }
        display_names = [short_names.get(name, name) for name in models_names]
        
        # Training set R²
        train_r2_scores = [self.results[name]['train_r2'] for name in models_names]
        bars1 = axes[0, 0].bar(display_names, train_r2_scores, color='lightblue', alpha=0.7)
        axes[0, 0].set_title('Training Set R² Score Comparison', fontsize=12, pad=20)
        axes[0, 0].set_ylabel('R² Score', fontsize=11)
        axes[0, 0].tick_params(axis='x', rotation=45, labelsize=9)
        axes[0, 0].tick_params(axis='y', labelsize=9)
        # 添加值标签
        for bar in bars1:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Test set R²
        test_r2_scores = [self.results[name]['test_r2'] for name in models_names]
        bars2 = axes[0, 1].bar(display_names, test_r2_scores, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Test Set R² Score Comparison', fontsize=12, pad=20)
        axes[0, 1].set_ylabel('R² Score', fontsize=11)
        axes[0, 1].tick_params(axis='x', rotation=45, labelsize=9)
        axes[0, 1].tick_params(axis='y', labelsize=9)
        # 添加值标签
        for bar in bars2:
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Training set RMSE
        train_rmse_scores = [self.results[name]['train_rmse'] for name in models_names]
        bars3 = axes[1, 0].bar(display_names, train_rmse_scores, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('Training Set RMSE Comparison', fontsize=12, pad=20)
        axes[1, 0].set_ylabel('RMSE', fontsize=11)
        axes[1, 0].tick_params(axis='x', rotation=45, labelsize=9)
        axes[1, 0].tick_params(axis='y', labelsize=9)
        # 添加值标签
        for bar in bars3:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Test set RMSE
        test_rmse_scores = [self.results[name]['test_rmse'] for name in models_names]
        bars4 = axes[1, 1].bar(display_names, test_rmse_scores, color='orange', alpha=0.7)
        axes[1, 1].set_title('Test Set RMSE Comparison', fontsize=12, pad=20)
        axes[1, 1].set_ylabel('RMSE', fontsize=11)
        axes[1, 1].tick_params(axis='x', rotation=45, labelsize=9)
        axes[1, 1].tick_params(axis='y', labelsize=9)
        # 添加值标签
        for bar in bars4:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 调整子图间距
        plt.tight_layout(pad=3.0)
        plt.savefig('training_test_comparison_optimized.png', dpi=600, bbox_inches='tight')
        plt.show()
        
        # Create a separate plot for training vs test comparison
        fig, ax = plt.subplots(figsize=(12, 8), dpi=600)
        
        x = np.arange(len(display_names))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, train_r2_scores, width, label='Training R²', color='lightblue')
        bars2 = ax.bar(x + width/2, test_r2_scores, width, label='Test R²', color='lightcoral')
        
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('R² Score', fontsize=12)
        ax.set_title('Training vs Test Set R² Score Comparison', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=45)
        ax.legend(fontsize=12)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('training_vs_test_r2_comparison.png', dpi=600, bbox_inches='tight')
        plt.show()
    
    def plot_comparison_metrics_horizontal(self):
        """
        使用水平条形图避免标签重叠
        """
        plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=600)
        
        models_names = list(self.results.keys())
        
        # Training set R² - 水平条形图
        train_r2_scores = [self.results[name]['train_r2'] for name in models_names]
        bars1 = axes[0, 0].barh(models_names, train_r2_scores, color='lightblue', alpha=0.7)
        axes[0, 0].set_title('Training Set R² Score Comparison', fontsize=12)
        axes[0, 0].set_xlabel('R² Score', fontsize=11)
        # 添加值标签
        for bar in bars1:
            width = bar.get_width()
            axes[0, 0].text(width, bar.get_y() + bar.get_height()/2.,
                        f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        # Test set R² - 水平条形图
        test_r2_scores = [self.results[name]['test_r2'] for name in models_names]
        bars2 = axes[0, 1].barh(models_names, test_r2_scores, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Test Set R² Score Comparison', fontsize=12)
        axes[0, 1].set_xlabel('R² Score', fontsize=11)
        # 添加值标签
        for bar in bars2:
            width = bar.get_width()
            axes[0, 1].text(width, bar.get_y() + bar.get_height()/2.,
                        f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        # Training set RMSE - 水平条形图
        train_rmse_scores = [self.results[name]['train_rmse'] for name in models_names]
        bars3 = axes[1, 0].barh(models_names, train_rmse_scores, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title('Training Set RMSE Comparison', fontsize=12)
        axes[1, 0].set_xlabel('RMSE', fontsize=11)
        # 添加值标签
        for bar in bars3:
            width = bar.get_width()
            axes[1, 0].text(width, bar.get_y() + bar.get_height()/2.,
                        f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        # Test set RMSE - 水平条形图
        test_rmse_scores = [self.results[name]['test_rmse'] for name in models_names]
        bars4 = axes[1, 1].barh(models_names, test_rmse_scores, color='orange', alpha=0.7)
        axes[1, 1].set_title('Test Set RMSE Comparison', fontsize=12)
        axes[1, 1].set_xlabel('RMSE', fontsize=11)
        # 添加值标签
        for bar in bars4:
            width = bar.get_width()
            axes[1, 1].text(width, bar.get_y() + bar.get_height()/2.,
                        f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('training_test_comparison_horizontal.png', dpi=600, bbox_inches='tight')
        plt.show()
    
    def plot_metrics_table(self):
        """
        使用表格形式显示模型性能，避免图表重叠
        """
        import matplotlib.pyplot as plt
        from matplotlib.table import Table
        
        plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(12, 8), dpi=600)
        ax.axis('tight')
        ax.axis('off')
        
        # 准备表格数据
        table_data = []
        headers = ['Model', 'Train R²', 'Test R²', 'Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE']
        
        for name, result in self.results.items():
            row = [
                name,
                f"{result['train_r2']:.4f}",
                f"{result['test_r2']:.4f}", 
                f"{result['train_rmse']:.4f}",
                f"{result['test_rmse']:.4f}",
                f"{result['train_mae']:.4f}",
                f"{result['test_mae']:.4f}"
            ]
            table_data.append(row)
        
        # 创建表格
        table = ax.table(cellText=table_data, colLabels=headers, 
                        loc='center', cellLoc='center')
        
        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # 设置标题行样式
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4F81BD')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 设置交替行颜色
        for i in range(1, len(table_data) + 1):
            color = '#DCE6F1' if i % 2 == 0 else '#EBF1DE'
            for j in range(len(headers)):
                table[(i, j)].set_facecolor(color)
        
        plt.title('Model Performance Comparison Table', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig('model_performance_table.png', dpi=600, bbox_inches='tight')
        plt.show()
    
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
        Plot result charts including both training and test sets - 优化版本
        """
        # 设置图表字体 - 移除不存在的字体
        plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建第一个图：模型性能比较
        fig1, axes1 = plt.subplots(2, 2, figsize=(16, 14), dpi=600)
        
        # 1. 模型性能比较
        models_names = list(self.results.keys())
        
        # 缩短模型名称用于显示
        short_names = {
            'Linear Regression': 'Linear',
            'Ridge Regression': 'Ridge',
            'Lasso Regression': 'Lasso', 
            'Random Forest': 'RF',
            'Gradient Boosting': 'GB',
            'Support Vector Regression': 'SVR',
            'Neural Network': 'NN'
        }
        display_names = [short_names.get(name, name) for name in models_names]
        
        train_r2_scores = [self.results[name]['train_r2'] for name in models_names]
        test_r2_scores = [self.results[name]['test_r2'] for name in models_names]
        train_rmse_scores = [self.results[name]['train_rmse'] for name in models_names]
        test_rmse_scores = [self.results[name]['test_rmse'] for name in models_names]
        
        # Training R²
        bars1 = axes1[0, 0].bar(display_names, train_r2_scores, color='skyblue', alpha=0.7, label='Training')
        axes1[0, 0].set_title('Training Set R² Score Comparison', fontsize=12, pad=20)
        axes1[0, 0].set_ylabel('R² Score', fontsize=11)
        axes1[0, 0].tick_params(axis='x', rotation=45, labelsize=9)
        axes1[0, 0].tick_params(axis='y', labelsize=9)
        
        # Test R²
        bars2 = axes1[0, 1].bar(display_names, test_r2_scores, color='lightcoral', alpha=0.7, label='Test')
        axes1[0, 1].set_title('Test Set R² Score Comparison', fontsize=12, pad=20)
        axes1[0, 1].set_ylabel('R² Score', fontsize=11)
        axes1[0, 1].tick_params(axis='x', rotation=45, labelsize=9)
        axes1[0, 1].tick_params(axis='y', labelsize=9)
        
        # Training RMSE
        bars3 = axes1[1, 0].bar(display_names, train_rmse_scores, color='lightgreen', alpha=0.7, label='Training')
        axes1[1, 0].set_title('Training Set RMSE Comparison', fontsize=12, pad=20)
        axes1[1, 0].set_ylabel('RMSE', fontsize=11)
        axes1[1, 0].tick_params(axis='x', rotation=45, labelsize=9)
        axes1[1, 0].tick_params(axis='y', labelsize=9)
        
        # Test RMSE
        bars4 = axes1[1, 1].bar(display_names, test_rmse_scores, color='orange', alpha=0.7, label='Test')
        axes1[1, 1].set_title('Test Set RMSE Comparison', fontsize=12, pad=20)
        axes1[1, 1].set_ylabel('RMSE', fontsize=11)
        axes1[1, 1].tick_params(axis='x', rotation=45, labelsize=9)
        axes1[1, 1].tick_params(axis='y', labelsize=9)
        
        plt.tight_layout(pad=3.0)
        plt.savefig('model_performance_comparison_optimized.png', dpi=600, bbox_inches='tight')
        plt.show()
        
        # 创建第二个图：所有模型的预测值与实际值对比（测试集）
        fig2, axes2 = plt.subplots(2, 4, figsize=(20, 10), dpi=600)
        axes2 = axes2.flatten()
        
        # 为每个模型绘制预测值与实际值对比图（测试集）
        for i, (name, result) in enumerate(self.results.items()):
            y_test_pred = result['test_predictions']
            
            axes2[i].scatter(self.y_test, y_test_pred, alpha=0.6, s=20)
            axes2[i].plot([self.y_test.min(), self.y_test.max()], 
                         [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            axes2[i].set_xlabel('Actual Values', fontsize=10)
            axes2[i].set_ylabel('Predicted Values', fontsize=10)
            axes2[i].set_title(f'{short_names.get(name, name)}\nTest R² = {result["test_r2"]:.4f}', fontsize=11)
            
            # 添加对角线参考线
            axes2[i].grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(self.results), len(axes2)):
            axes2[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('predictions_vs_actuals_optimized.png', dpi=600, bbox_inches='tight')
        plt.show()
        
        # 创建第三个图：特征重要性
        fig3, axes3 = plt.subplots(2, 2, figsize=(15, 12), dpi=600)
        axes3 = axes3.flatten()
        
        # 为具有特征重要性的模型绘制特征重要性图
        feature_importance_plots = 0
        
        # Random Forest 特征重要性
        if 'Random Forest' in self.models:
            rf_model = self.models['Random Forest']
            feature_importance = rf_model.feature_importances_
            feature_names = self.X.columns
            
            indices = np.argsort(feature_importance)[::-1]
            
            axes3[feature_importance_plots].bar(range(len(feature_importance)), 
                          feature_importance[indices], color='skyblue')
            axes3[feature_importance_plots].set_title('Random Forest Feature Importance', fontsize=12)
            axes3[feature_importance_plots].set_xticks(range(len(feature_importance)))
            axes3[feature_importance_plots].set_xticklabels([feature_names[i] for i in indices], rotation=45, fontsize=10)
            axes3[feature_importance_plots].set_ylabel('Importance', fontsize=10)
            feature_importance_plots += 1
        
        # Gradient Boosting 特征重要性
        if 'Gradient Boosting' in self.models:
            gb_model = self.models['Gradient Boosting']
            feature_importance = gb_model.feature_importances_
            feature_names = self.X.columns
            
            indices = np.argsort(feature_importance)[::-1]
            
            axes3[feature_importance_plots].bar(range(len(feature_importance)), 
                          feature_importance[indices], color='orange')
            axes3[feature_importance_plots].set_title('Gradient Boosting Feature Importance', fontsize=12)
            axes3[feature_importance_plots].set_xticks(range(len(feature_importance)))
            axes3[feature_importance_plots].set_xticklabels([feature_names[i] for i in indices], rotation=45, fontsize=10)
            axes3[feature_importance_plots].set_ylabel('Importance', fontsize=10)
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
                                  coefficients[indices], color='lightgreen')
                    axes3[feature_importance_plots].set_title(f'{short_names.get(model_name, model_name)} Coefficients (abs)', fontsize=12)
                    axes3[feature_importance_plots].set_xticks(range(len(coefficients)))
                    axes3[feature_importance_plots].set_xticklabels([feature_names[i] for i in indices], rotation=45, fontsize=10)
                    axes3[feature_importance_plots].set_ylabel('Coefficient Value', fontsize=10)
                    feature_importance_plots += 1
        
        # 隐藏多余的子图
        for i in range(feature_importance_plots, len(axes3)):
            axes3[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('feature_importance_optimized.png', dpi=600, bbox_inches='tight')
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
            print(f"Training R²: {result['train_r2']:.4f}, Test R²: {result['test_r2']:.4f}")
            
            if hasattr(model, 'intercept_') and hasattr(model, 'coef_'):
                # Linear model equations
                equation = f"nppv = {model.intercept_:.6f}"
                for i, (feature, coef) in enumerate(zip(feature_names, model.coef_)):
                    sign = '+' if coef >= 0 else ''
                    equation += f" {sign}{coef:.6f}×{feature}"
                print(f"Equation: {equation}")
            
            elif hasattr(model, 'feature_importances_'):
                # Tree model feature importance
                print("Feature Importance:")
                importances = model.feature_importances_
                for feature, importance in zip(feature_names, importances):
                    print(f"  {feature}: {importance:.4f}")
    
    def run_complete_analysis(self):
        """
        Run complete analysis pipeline with optimized plotting
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
        print(f"Best Test R² Score: {best_result['test_r2']:.4f}")
        print(f"Best Test RMSE: {best_result['test_rmse']:.4f}")
        print(f"{'='*50}")
        
        # Print detailed metrics
        self.print_detailed_metrics()
        
        # 使用优化后的绘图函数
        self.plot_comparison_metrics()  # 优化的比较图
        self.plot_metrics_table()       # 新增的表格显示
        
        # Plot results
        self.plot_results()             # 优化的结果图
        
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
    
    # 设置图表字体 - 移除不存在的字体
    plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(10, 8), dpi=600)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.3f', annot_kws={"size": 10})
    plt.title('Environmental Variables Correlation Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=600, bbox_inches='tight')
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