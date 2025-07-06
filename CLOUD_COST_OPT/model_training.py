import pandas as pd
import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="viridis")


class TrafficForecastingModel:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = None
        self.feature_columns = []
        self.server_thresholds = {}
        self.target_column = None

        # Create necessary directories if they don't exist
        directories = ["output_graphs", "data", "config", "models", "cloud", "utils"]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)


    def load_preprocessed_data(self):
        """Load preprocessed data and configurations"""
        print("Loading preprocessed data...")

        # Load processed data
        try:
            self.df = pd.read_csv('data/processed_traffic_data.csv')
            # Ensure timestamp is datetime and set as index
            if 'timestamp' in self.df.columns:
                self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
                self.df.set_index('timestamp', inplace=True)
            print("Successfully loaded processed_traffic_data.csv")
        except FileNotFoundError:
            print("Error: 'data/processed_traffic_data.csv' not found. Please run data_analysis.py first.")
            raise

        # Load scaler
        try:
            self.scaler = joblib.load('models/scaler.pkl')
            print("Successfully loaded scaler.pkl")
        except FileNotFoundError:
            print("Error: 'models/scaler.pkl' not found. Please run data_analysis.py first.")
            raise

        # Load feature columns
        try:
            with open('config/feature_columns.json', 'r') as f:
                self.feature_columns = json.load(f)
            print("Successfully loaded feature_columns.json")
        except FileNotFoundError:
            print("Error: 'config/feature_columns.json' not found. Please run data_analysis.py first.")
            raise

        # Load server thresholds
        try:
            with open('config/server_thresholds.json', 'r') as f:
                self.server_thresholds = json.load(f)
            print("Successfully loaded server_thresholds.json")
        except FileNotFoundError:
            print("Error: 'config/server_thresholds.json' not found. Please run data_analysis.py first.")
            raise


        print(f"Data loaded: {self.df.shape}")
        print(f"Features: {len(self.feature_columns)}")

        return self.df

    def prepare_training_data(self):
        """Prepare training and testing data"""
        print("Preparing training data...")

        # Get target column (traffic column)
        if 'Page Views' in self.df.columns:
            self.target_column = 'Page Views'
        else:
            traffic_cols = [col for col in self.df.columns if 'traffic' in col.lower() or 'views' in col.lower()]
            if traffic_cols:
                self.target_column = traffic_cols[0]
            else:
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                exclude_features = ['year', 'month', 'day', 'hour', 'day_of_week', 'is_weekend', 'is_business_hours',
                                    'servers_needed', 'hourly_cost', 'servers_needed_dynamic', 'hourly_cost_dynamic']
                target_candidates = [col for col in numeric_cols if
                                     col not in exclude_features and 'lag' not in col.lower() and 'rolling' not in col.lower() and 'sin' not in col.lower() and 'cos' not in col.lower() and 'trend' not in col.lower() and 'volatility' not in col.lower() and 'source' not in col.lower()]
                self.target_column = target_candidates[0] if target_candidates else None
                if not self.target_column:
                    raise ValueError("Could not automatically identify the target traffic column. Please specify it.")


        print(f"Target column: {self.target_column}")

        # Prepare features and target
        # Ensure feature_columns only contains columns present in df
        self.feature_columns = [col for col in self.feature_columns if col in self.df.columns]
        if not self.feature_columns:
            raise ValueError("No valid feature columns found in the DataFrame after filtering.")

        X = self.df[self.feature_columns]
        y = self.df[self.target_column]

        initial_rows = len(X)
        combined_df = pd.concat([X, y], axis=1).dropna()
        X = combined_df[X.columns]
        y = combined_df[y.name]
        if len(X) < initial_rows:
            print(f"Removed {initial_rows - len(X)} rows with NaNs during training data preparation.")

        # Time series split (80% train, 20% test)
        split_index = int(len(X) * 0.8)

        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Scale features
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test

    def train_models(self, X_train, y_train):
        """Train multiple ML models"""
        print("Training multiple models...")

        # Define models
        models_config = {
            'LinearRegression': {
                'model': LinearRegression(),
                'params': {}
            },
            'RandomForest': {
                'model': RandomForestRegressor(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5]
                }
            }
        }

        # Time series cross-validation
        n_samples = len(X_train)
        if n_samples < 2:
            print("Warning: Training data has less than 2 samples. TimeSeriesSplit and GridSearchCV may not work as expected.")
            n_splits_tscv = 1
        elif n_samples < 10:
            n_splits_tscv = max(2, n_samples // 2)
        elif n_samples < 100:
            n_splits_tscv = 3
        else:
            n_splits_tscv = 5

        tscv = TimeSeriesSplit(n_splits=n_splits_tscv)
        print(f"Using TimeSeriesSplit with {n_splits_tscv} splits.")


        for model_name, config in models_config.items():
            print(f"\nTraining {model_name}...")

            if config['params']:
                grid_search = GridSearchCV(
                    config['model'],
                    config['params'],
                    cv=tscv,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=0
                )
                grid_search.fit(X_train, y_train)
                self.models[model_name] = grid_search.best_estimator_
                print(f"Best parameters for {model_name}: {grid_search.best_params_}")
            else:
                config['model'].fit(X_train, y_train)
                self.models[model_name] = config['model']

        print("Model training completed!")
        return self.models

    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        print("Evaluating models...")

        evaluation_results = {}

        for model_name, model in self.models.items():
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mse)

            mape = np.mean(np.abs((y_test - y_pred) / y_test.replace(0, np.nan))) * 100
            mape = 0 if np.isnan(mape) else mape

            evaluation_results[model_name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'MAPE': mape
            }

            print(f"\n{model_name} Performance:")
            print(f"  RMSE: {rmse:.2f}")
            print(f"  MAE: {mae:.2f}")
            print(f"  R²: {r2:.4f}")
            print(f"  MAPE: {mape:.2f}%")

        best_model_name = min(evaluation_results.keys(),
                              key=lambda x: evaluation_results[x]['RMSE'])
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]

        print(f"\nBest Model: {best_model_name}")

        with open('models/evaluation_results.json', 'w') as f:
            json.dump(evaluation_results, f, indent=2)

        return evaluation_results

    def visualize_predictions(self, X_test, y_test):
        """Visualize model predictions"""
        print("Creating prediction visualizations...")

        num_models = len(self.models)
        rows = int(np.ceil(num_models / 2)) if num_models > 0 else 1
        cols = 2 if num_models > 0 else 1
        if num_models == 1:
            rows, cols = 1, 1
        elif num_models == 0:
            print("No models to visualize.")
            return

        # Scatter plot: Actual vs. Predicted Traffic for each model
        fig1, axes1 = plt.subplots(rows, cols, figsize=(8 * cols, 7 * rows))
        fig1.suptitle('Actual vs. Predicted Traffic: Model Comparison', fontsize=20, fontweight='bold', y=1.02)

        colors = sns.color_palette("tab10", n_colors=num_models)

        axes_flat1 = axes1.flatten() if num_models > 1 else [axes1]

        for i, (model_name, model) in enumerate(self.models.items()):
            ax = axes_flat1[i]
            y_pred = model.predict(X_test)

            ax.scatter(y_test, y_pred, alpha=0.6, color=colors[i], s=30, label='Predictions', edgecolor='none')
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val],
                    'r--', lw=2, label='Perfect Prediction (y=x)')

            ax.set_xlabel('Actual Traffic', fontsize=13)
            ax.set_ylabel('Predicted Traffic', fontsize=13)
            ax.set_title(f'{model_name} Predictions', fontsize=15, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.tick_params(axis='both', which='major', labelsize=10)


        for i in range(num_models, len(axes_flat1)):
            fig1.delaxes(axes_flat1[i])

        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        plt.savefig('output_graphs/model_predictions_scatter.png', dpi=300, bbox_inches='tight')
        # plt.close(fig1)

        # Time series plot: Actual vs. Predicted Traffic over time
        fig2 = plt.figure(figsize=(18, 9))
        ax2 = fig2.add_subplot(111)

        plot_points = min(500, len(y_test))
        # Get the actual timestamps for the test set
        # Ensure that self.df.index is used correctly after setting it in load_preprocessed_data
        test_timestamps = self.df.index[len(self.df) - len(y_test):]
        test_timestamps_plot = test_timestamps[-plot_points:]
        y_test_plot = y_test.iloc[-plot_points:]


        ax2.plot(test_timestamps_plot, y_test_plot,
                 label='Actual Traffic', color='black', linewidth=2.5, alpha=0.8, zorder=5)

        for i, (model_name, model) in enumerate(self.models.items()):
            y_pred = model.predict(X_test)
            y_pred_plot = y_pred[-plot_points:]

            ax2.plot(test_timestamps_plot, y_pred_plot,
                     label=f'{model_name} Prediction',
                     color=colors[i], linewidth=1.8, alpha=0.7, linestyle='--')

        ax2.set_title(f'Traffic Prediction Comparison Over Time (Last {plot_points} Data Points)',
                  fontsize=20, fontweight='bold')
        ax2.set_xlabel('Date and Time', fontsize=14)
        ax2.set_ylabel('Traffic Volume', fontsize=14)
        ax2.legend(fontsize=12, loc='upper left')
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.tick_params(axis='both', which='major', labelsize=10)
        fig2.autofmt_xdate(rotation=45)

        plt.tight_layout()
        plt.savefig('output_graphs/time_series_predictions.png', dpi=300, bbox_inches='tight')
        # plt.close(fig2)

    def create_feature_importance_plot(self):
        """Create feature importance plot for tree-based models with enhanced visuals"""
        print("Creating feature importance visualization...")

        if 'RandomForest' in self.models and hasattr(self.models['RandomForest'], 'feature_importances_'):
            rf_model = self.models['RandomForest']
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=False)

            top_n = min(20, len(feature_importance))
            feature_importance_plot = feature_importance.head(top_n)

            plt.figure(figsize=(14, max(8, top_n * 0.4)))
            ax = sns.barplot(data=feature_importance_plot, x='importance', y='feature', palette='viridis')
            plt.title(f'Top {top_n} Feature Importance for Random Forest Model',
                      fontsize=18, fontweight='bold')
            plt.xlabel('Importance Score', fontsize=14)
            plt.ylabel('Feature Name', fontsize=14)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(axis='x', linestyle='--', alpha=0.6)

            for p in ax.patches:
                ax.annotate(f'{p.get_width():.4f}',
                            (p.get_width(), p.get_y() + p.get_height() / 2),
                            xytext=(5, 0), textcoords='offset points',
                            ha='left', va='center', fontsize=9, color='black')

            plt.tight_layout()
            plt.savefig('output_graphs/feature_importance.png', dpi=300, bbox_inches='tight')
            # plt.close()
        else:
            print("RandomForest model not found or does not have feature_importances_ attribute. Skipping feature importance plot.")


    def save_best_model(self):
        """Save the best performing model"""
        print(f"Saving best model: {self.best_model_name}")

        if self.best_model is None:
            print("No best model found to save. Please ensure models are trained.")
            return

        joblib.dump(self.best_model, 'models/best_traffic_model.pkl')

        model_metadata = {
            'best_model_name': self.best_model_name,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open('models/model_metadata.json', 'w') as f:
            json.dump(model_metadata, f, indent=2)

        print("Best model saved successfully!")

    def predict_server_requirements(self, predictions):
        """Convert traffic predictions to server requirements"""
        server_requirements = []

        predictions = np.maximum(0, predictions)

        sorted_thresholds = sorted(self.server_thresholds.items(), key=lambda item: item[1]['threshold'])

        for pred in predictions:
            servers = 1
            for level, config in sorted_thresholds:
                if pred >= config['threshold']:
                    servers = config['servers']
                else:
                    break
            server_requirements.append(servers)

        return server_requirements

    def generate_forecast_report(self, X_test, y_test):
        """Generate a comprehensive forecast report"""
        print("Generating forecast report...")

        if self.best_model is None:
            print("No best model available to generate forecast report.")
            return None, None

        y_pred = self.best_model.predict(X_test)

        server_requirements = self.predict_server_requirements(y_pred)

        # Ensure that the index of y_test is preserved for the forecast_df
        forecast_df = pd.DataFrame({
            'actual_traffic': y_test.values,
            'predicted_traffic': y_pred,
            'servers_needed': server_requirements
        }, index=y_test.index)

        costs = []
        server_cost_map = {config['servers']: config['hourly_cost'] for config in self.server_thresholds.values()}

        for servers in server_requirements:
            cost = server_cost_map.get(servers, 0.0)
            costs.append(cost)

        forecast_df['hourly_cost'] = costs

        forecast_df.to_csv('data/traffic_forecast.csv', index=False)

        mape_value = np.mean(np.abs((y_test - y_pred) / y_test.replace(0, np.nan))) * 100
        mape_value = 0 if np.isnan(mape_value) else mape_value


        report = {
            'forecast_summary': {
                'total_predictions': len(y_pred),
                'avg_predicted_traffic': float(np.mean(y_pred)),
                'max_predicted_traffic': float(np.max(y_pred)),
                'min_predicted_traffic': float(np.min(y_pred)),
                'avg_servers_needed': float(np.mean(server_requirements)),
                'max_servers_needed': int(np.max(server_requirements)),
                'total_predicted_cost': float(np.sum(costs))
            },
            'model_performance': {
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                'mae': float(mean_absolute_error(y_test, y_pred)),
                'r2_score': float(r2_score(y_test, y_pred)),
                'mape': float(mape_value)
            }
        }

        with open('models/forecast_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        print("Forecast report generated!")
        return forecast_df, report

    def run_complete_training_pipeline(self):
        """Run the complete training pipeline"""
        print("=" * 60)
        print("STARTING ML MODEL TRAINING PIPELINE")
        print("=" * 60)

        # Step 1: Load data
        self.load_preprocessed_data()

        # Step 2: Prepare training data
        X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = self.prepare_training_data()

        # Step 3: Train models
        self.train_models(X_train, y_train)

        # Step 4: Evaluate models
        evaluation_results = self.evaluate_models(X_test, y_test)

        # Step 5: Visualize predictions
        self.visualize_predictions(X_test, y_test)

        # Step 6: Feature importance
        self.create_feature_importance_plot()

        # Step 7: Save best model
        self.save_best_model()

        # Step 8: Generate forecast report
        forecast_df, report = self.generate_forecast_report(X_test, y_test)

        print("\n" + "=" * 60)
        print("ML MODEL TRAINING COMPLETED!")
        print("=" * 60)
        if report:
            print(f"Best Model: {self.best_model_name}")
            print(f"Model Performance:")
            print(f"  RMSE: {report['model_performance']['rmse']:.2f}")
            print(f"  R²: {report['model_performance']['r2_score']:.4f}")
            print(f"  MAPE: {report['model_performance']['mape']:.2f}%")
            print(f"\nForecast Summary:")
            print(f"  Average predicted traffic: {report['forecast_summary']['avg_predicted_traffic']:.2f}")
            print(f"  Average servers needed: {report['forecast_summary']['avg_servers_needed']:.1f}")
            print(f"  Total predicted cost: ${report['forecast_summary']['total_predicted_cost']:.2f}")
        else:
            print("Forecast report could not be generated due to an issue.")


        print("\nFiles created:")
        print("  - models/best_traffic_model.pkl")
        print("  - models/model_metadata.json")
        print("  - models/evaluation_results.json")
        print("  - models/forecast_report.json")
        print("  - data/traffic_forecast.csv")
        print("  - output_graphs/model_predictions_scatter.png")
        print("  - output_graphs/time_series_predictions.png")
        print("  - output_graphs/feature_importance.png")


        return self.best_model, evaluation_results, forecast_df, report


if __name__ == "__main__":
    try:
        trainer = TrafficForecastingModel()
        best_model, evaluation_results, forecast_df, report = trainer.run_complete_training_pipeline()

        print("\n" + "=" * 50)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("Next steps:")
        print("1. Review model performance in output_graphs/")
        print("2. Check forecast results in data/traffic_forecast.csv")
        print("3. Run predict.py to make new predictions")
        print("4. Set up cloud integration with aws_controller.py")

        plt.show()
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback

        traceback.print_exc()

