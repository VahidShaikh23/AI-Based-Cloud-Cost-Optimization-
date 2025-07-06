import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import warnings
import json
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# --- RL Specific Imports (for a simple placeholder) ---
import random
from collections import deque

warnings.filterwarnings('ignore')
# Using a better default theme from seaborn
sns.set_theme(style="whitegrid", palette="viridis")


# --- RL Environment Definition ---
class ServerScalingEnv:
    def __init__(self, traffic_data, server_configs):
        self.traffic_data = traffic_data
        self.server_configs = server_configs
        self.current_step = 0
        self.max_steps = len(traffic_data) - 1  # Each step is a time point in traffic data

        # Map server counts to instance types and costs
        self.server_map = {
            1: {'type': 't3.micro', 'cost': 0.0104},
            2: {'type': 't3.small', 'cost': 0.0208},
            3: {'type': 't3.medium', 'cost': 0.0416},
            5: {'type': 't3.large', 'cost': 0.0832},
            8: {'type': 't3.xlarge', 'cost': 0.1664}
        }
        # Extract unique server counts from configs and sort them to define action space
        self.available_server_counts = sorted(list(set(s['servers'] for s in server_configs.values())))
        if not self.available_server_counts:
            # Fallback if server_configs is empty or malformed
            self.available_server_counts = [1, 2, 3, 5, 8]
            print(
                "Warning: server_threshold_config is empty or malformed. Using default server counts for RL environment.")

        # Define state space: (traffic_level_index, current_servers_index)
        # For simplicity, discretize traffic into levels based on thresholds
        self.traffic_levels = sorted([config['threshold'] for config in server_configs.values()])
        self.num_traffic_levels = len(self.traffic_levels) + 1  # Add one for traffic below lowest threshold
        self.num_server_states = len(self.available_server_counts)  # Number of distinct server counts

    def _get_traffic_level_index(self, traffic):
        for i, threshold in enumerate(self.traffic_levels):
            if traffic < threshold:
                return i
        return len(self.traffic_levels)  # Max level if above all thresholds

    def _get_server_count_index(self, servers):
        try:
            return self.available_server_counts.index(servers)
        except ValueError:
            # If server count is not in available, find the closest one or default
            closest_idx = np.argmin(np.abs(np.array(self.available_server_counts) - servers))
            return closest_idx

    def reset(self):
        self.current_step = 0
        # Initial state: Traffic at first data point, starting with 1 server
        initial_traffic = self.traffic_data.iloc[self.current_step]
        initial_servers = self.available_server_counts[0]  # Start with the smallest server count

        state = (self._get_traffic_level_index(initial_traffic), self._get_server_count_index(initial_servers))
        return state

    def step(self, action_index, current_servers):
        # Action is to select a specific number of servers from self.available_server_counts
        next_servers = self.available_server_counts[action_index]

        # Get current traffic for this step
        current_traffic = self.traffic_data.iloc[self.current_step]

        # Calculate cost for current_servers
        # Find the cost for next_servers from server_map, default if not found
        cost_per_hour = self.server_map.get(next_servers, {'cost': 0.0104 * next_servers})['cost']

        # Calculate reward
        # Reward function: Maximize cost efficiency, penalize under-provisioning
        # Example: Reward = -cost - penalty_for_underprovisioning

        # Simple penalty: if traffic is high but servers are low
        penalty = 0

        # Determine the "ideal" servers based on the thresholds for the current traffic
        ideal_servers_for_traffic = self.available_server_counts[0]  # Default to lowest
        for level, config in sorted(self.server_configs.items(), key=lambda item: item[1]['threshold']):
            if current_traffic >= config['threshold']:
                ideal_servers_for_traffic = config['servers']
            else:
                break  # Traffic is below this threshold, use the previous config

        # If chosen servers are significantly less than ideal, apply penalty
        if next_servers < ideal_servers_for_traffic:
            # Penalty scales with how much we are under-provisioning
            penalty = (ideal_servers_for_traffic - next_servers) * 0.1  # A small penalty per server under-provisioned
        elif next_servers > ideal_servers_for_traffic:
            # Small penalty for over-provisioning (less severe than under-provisioning)
            penalty = (next_servers - ideal_servers_for_traffic) * 0.05

        reward = -cost_per_hour - penalty  # Minimize cost, minimize penalty

        self.current_step += 1
        done = self.current_step >= self.max_steps

        next_traffic = self.traffic_data.iloc[min(self.current_step, self.max_steps)]  # Next traffic for next state
        next_state = (self._get_traffic_level_index(next_traffic), self._get_server_count_index(next_servers))

        return next_state, reward, done, {}


# --- Simple Q-Learning Agent Definition ---
class RLAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.99, epsilon_start=1.0,
                 epsilon_end=0.01, epsilon_decay=0.995):
        self.state_size = state_size  # (num_traffic_levels, num_server_states)
        self.action_size = action_size  # len(available_server_counts)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros(
            state_size + (action_size,))  # Q-table dimensions: (traffic_levels, server_states, actions)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randrange(self.action_size)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def learn(self, state, action, reward, next_state):
        current_q_value = self.q_table[state + (action,)]
        max_next_q_value = np.max(self.q_table[next_state])
        new_q_value = current_q_value + self.learning_rate * (
                    reward + self.discount_factor * max_next_q_value - current_q_value)
        self.q_table[state + (action,)] = new_q_value

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def predict(self, current_traffic, current_servers, env):
        # Convert continuous inputs to discretized states for prediction
        traffic_idx = env._get_traffic_level_index(current_traffic)
        servers_idx = env._get_server_count_index(current_servers)
        state = (traffic_idx, servers_idx)

        # Choose action based on learned Q-values (exploitation only)
        action_idx = np.argmax(self.q_table[state])
        recommended_servers = env.available_server_counts[action_idx]
        return recommended_servers


class WebsiteTrafficAnalyzer:
    def __init__(self, data_path, output_timestamp_col='timestamp'):
        self.data_path = data_path
        self.df = None
        self.output_timestamp_col = output_timestamp_col
        self.traffic_col = None
        self.server_threshold_config = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.traffic_prediction_model = None  # To store the trained traffic prediction model
        self.rl_agent = None  # To store the trained RL agent

        # Create necessary directories
        directories = ["output_graphs", "data", "config", "models", "cloud", "utils"]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def load_and_explore_data(self):
        print("Loading website traffic data...")
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Dataset shape: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")
            print("\nFirst few rows:")
            print(self.df.head())
            print("\nData types:")
            print(self.df.dtypes)
            print("\nMissing values:")
            print(self.df.isnull().sum())
            print("\nBasic statistics:")
            print(self.df.describe())
            return self.df
        except FileNotFoundError:
            print(f"Error: File {self.data_path} not found!")
            raise
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def preprocess_data(self):
        print("\nPreprocessing data...")
        # Check for existing datetime columns first
        datetime_cols = [col for col in self.df.columns if 'date' in col.lower() or 'time' in col.lower()]
        found_datetime_col = None

        if datetime_cols:
            for col in datetime_cols:
                try:
                    # Attempt to convert to datetime, coercing errors to NaT
                    temp_series = pd.to_datetime(self.df[col], errors='coerce')
                    # If more than 50% of values are successfully converted, consider it a datetime column
                    if temp_series.notna().sum() / len(temp_series) > 0.5:
                        self.df[col] = temp_series
                        found_datetime_col = col
                        print(f"Converted {col} to datetime.")
                        break  # Found a suitable datetime column, exit loop
                except Exception as e:
                    # print(f"Could not convert {col} to datetime: {e}")
                    continue

        if not found_datetime_col:
            print(f"No suitable datetime column found. Creating synthetic '{self.output_timestamp_col}' column.")
            # Ensure the synthetic timestamp covers the actual data range, which is often hourly.
            # Assuming the data points represent hourly intervals, the frequency 'H' is suitable.
            self.df[self.output_timestamp_col] = pd.date_range(
                start='2023-01-01 00:00:00',
                periods=len(self.df),
                freq='H'
            )
        elif found_datetime_col != self.output_timestamp_col:
            self.df[self.output_timestamp_col] = self.df[found_datetime_col]
            print(f"Standardized datetime column to '{self.output_timestamp_col}'.")

        traffic_keywords = ['visit', 'traffic', 'hit', 'user', 'session', 'page_view', 'views']
        for col in self.df.columns:
            if (self.df[col].dtype in [np.number, 'int64', 'float64'] and
                    any(keyword in col.lower() for keyword in traffic_keywords)):
                self.traffic_col = col
                break

        if not self.traffic_col:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if 'Page Views' in self.df.columns:
                self.traffic_col = 'Page Views'
                print(f"Using 'Page Views' as traffic column.")
            elif len(numeric_cols) > 0:
                self.traffic_col = numeric_cols[0]
                print(f"Using {self.traffic_col} as traffic column (fallback).")
            else:
                raise ValueError("No numeric column found for traffic data!")

        print(f"Using traffic column: {self.traffic_col}")

        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        # Ensure timestamp and traffic columns are not null after processing
        self.df = self.df.dropna(subset=[self.output_timestamp_col, self.traffic_col])
        print(f"Removed {initial_rows - len(self.df)} rows during cleaning")
        print(f"Final dataset shape: {self.df.shape}")

        # Removed: self.df.to_csv('data/processed_traffic_data.csv', index=False)
        # This will now be saved AFTER feature creation in run_analysis

        return self.df

    def create_time_series_features(self):
        print("Creating time series features...")
        self.df = self.df.sort_values(self.output_timestamp_col)
        self.df.set_index(self.output_timestamp_col, inplace=True)

        # Time-based features
        self.df['year'] = self.df.index.year
        self.df['month'] = self.df.index.month
        self.df['day'] = self.df.index.day
        self.df['hour'] = self.df.index.hour
        self.df['day_of_week'] = self.df.index.dayofweek  # Monday=0, Sunday=6
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6]).astype(int)
        self.df['is_business_hours'] = ((self.df['hour'] >= 9) & (self.df['hour'] <= 17)).astype(int)

        # Cyclical features for better ML performance
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df['hour'] / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df['hour'] / 24)
        self.df['day_sin'] = np.sin(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['day_cos'] = np.cos(2 * np.pi * self.df['day_of_week'] / 7)
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)

        day_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
        self.df['weekday'] = self.df['day_of_week'].map(day_map)

        # Lag features
        # Ensure traffic_col is numeric before shifting
        if not pd.api.types.is_numeric_dtype(self.df[self.traffic_col]):
            print(f"Warning: Traffic column '{self.traffic_col}' is not numeric. Attempting conversion.")
            self.df[self.traffic_col] = pd.to_numeric(self.df[self.traffic_col], errors='coerce')

        self.df[f'{self.traffic_col}_lag1'] = self.df[self.traffic_col].shift(1)
        self.df[f'{self.traffic_col}_lag24'] = self.df[self.traffic_col].shift(24)
        self.df[f'{self.traffic_col}_lag168'] = self.df[self.traffic_col].shift(168)

        # Rolling statistics
        # Minimum periods for rolling calculations to avoid NaNs at the start
        min_periods_24h = min(24, len(self.df))
        min_periods_7d = min(168, len(self.df))

        self.df[f'{self.traffic_col}_rolling_mean_24h'] = self.df[self.traffic_col].rolling(24,
                                                                                            min_periods=min_periods_24h).mean()
        self.df[f'{self.traffic_col}_rolling_mean_7d'] = self.df[self.traffic_col].rolling(168,
                                                                                           min_periods=min_periods_7d).mean()
        self.df[f'{self.traffic_col}_rolling_std_24h'] = self.df[self.traffic_col].rolling(24,
                                                                                           min_periods=min_periods_24h).std()
        self.df[f'{self.traffic_col}_rolling_max_24h'] = self.df[self.traffic_col].rolling(24,
                                                                                           min_periods=min_periods_24h).max()
        self.df[f'{self.traffic_col}_rolling_min_24h'] = self.df[self.traffic_col].rolling(24,
                                                                                           min_periods=min_periods_24h).min()

        # Trend features - handle potential division by zero
        self.df[f'{self.traffic_col}_trend_24h'] = self.df[self.traffic_col] / self.df[
            f'{self.traffic_col}_rolling_mean_24h'].replace(0, np.nan)
        self.df[f'{self.traffic_col}_volatility'] = self.df[f'{self.traffic_col}_rolling_std_24h'] / self.df[
            f'{self.traffic_col}_rolling_mean_24h'].replace(0, np.nan)

        self.df = self.df.dropna()  # Drop rows with NaN values resulting from feature creation
        self.df.reset_index(inplace=True) # Reset index before saving to CSV
        print(f"Created time series features. Final shape: {self.df.shape}")

        # Define feature columns for saving
        self.feature_columns = [
            'year', 'month', 'day', 'hour', 'day_of_week', 'is_weekend', 'is_business_hours',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            f'{self.traffic_col}_lag1', f'{self.traffic_col}_lag24', f'{self.traffic_col}_lag168',
            f'{self.traffic_col}_rolling_mean_24h', f'{self.traffic_col}_rolling_mean_7d',
            f'{self.traffic_col}_rolling_std_24h', f'{self.traffic_col}_rolling_max_24h',
            f'{self.traffic_col}_rolling_min_24h', f'{self.traffic_col}_trend_24h',
            f'{self.traffic_col}_volatility'
        ]
        # Save feature columns
        with open('config/feature_columns.json', 'w') as f:
            json.dump(self.feature_columns, f, indent=2)
        print("Feature columns saved to config/feature_columns.json")

        return self.df

    def calculate_server_thresholds(self):
        print("Calculating server scaling thresholds...")

        # Use quantiles for more robust percentile calculation
        traffic_25th_percentile = self.df[self.traffic_col].quantile(0.25)
        traffic_50th_percentile = self.df[self.traffic_col].quantile(0.50)
        traffic_75th_percentile = self.df[self.traffic_col].quantile(0.75)
        traffic_90th_percentile = self.df[self.traffic_col].quantile(0.90)
        traffic_95th_percentile = self.df[self.traffic_col].quantile(0.95)

        self.server_threshold_config = {
            'low_traffic': {
                'threshold': float(traffic_25th_percentile),
                'servers': 1,
                'instance_type': 't3.micro',
                'description': 'Minimal traffic - single server'
            },
            'medium_traffic': {
                'threshold': float(traffic_50th_percentile),
                'servers': 2,
                'instance_type': 't3.small',
                'description': 'Moderate traffic - dual servers'
            },
            'high_traffic': {
                'threshold': float(traffic_75th_percentile),
                'servers': 3,
                'instance_type': 't3.medium',
                'description': 'High traffic - three servers'
            },
            'peak_traffic': {
                'threshold': float(traffic_90th_percentile),
                'servers': 5,
                'instance_type': 't3.large',
                'description': 'Peak traffic - maximum scaling'
            },
            'extreme_traffic': {
                'threshold': float(traffic_95th_percentile),
                'servers': 8,
                'instance_type': 't3.xlarge',
                'description': 'Extreme traffic - emergency scaling'
            }
        }

        # Add cost estimates (example hourly rates)
        cost_per_hour = {
            't3.micro': 0.0104,  # $0.0104/hr
            't3.small': 0.0208,  # $0.0208/hr
            't3.medium': 0.0416,  # $0.0416/hr
            't3.large': 0.0832,  # $0.0832/hr
            't3.xlarge': 0.1664  # $0.1664/hr
        }

        # Ensure all instance types used in self.server_threshold_config have a defined cost_per_hour
        for level, config in self.server_threshold_config.items():
            if config['instance_type'] not in cost_per_hour:
                # Assign a default cost if instance type is not found
                cost_per_hour[config['instance_type']] = 0.0104  # Default to t3.micro cost if not specified
                print(
                    f"Warning: Instance type '{config['instance_type']}' not found in cost_per_hour map. Using default cost of $0.0104/hr.")

            hourly_cost = cost_per_hour[config['instance_type']] * config['servers']
            config['hourly_cost'] = hourly_cost
            config['daily_cost'] = hourly_cost * 24

        with open('config/server_thresholds.json', 'w') as f:
            json.dump(self.server_threshold_config, f, indent=2)

        print("Server thresholds calculated and saved to config/server_thresholds.json")
        return self.server_threshold_config

    def visualize_traffic_patterns(self):
        print("Creating visualizations...")

        # Time series plot
        plt.figure(figsize=(16, 6))
        sns.lineplot(x=self.df[self.output_timestamp_col], y=self.df[self.traffic_col],
                     color='darkblue', linewidth=1.5, alpha=0.8)
        plt.title('Website Traffic Volume Over Time', fontsize=18, fontweight='bold')
        plt.xlabel('Date and Time', fontsize=14)
        plt.ylabel(self.traffic_col.replace('_', ' ').title(), fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('output_graphs/traffic_over_time.png', dpi=300, bbox_inches='tight')
        # plt.close() # Keep open for plt.show() later

        # Hourly patterns
        plt.figure(figsize=(14, 7))
        hourly_avg = self.df.groupby('hour')[self.traffic_col].mean()
        ax = sns.barplot(x=hourly_avg.index, y=hourly_avg.values, palette='Blues_d')
        plt.title('Average Website Traffic by Hour of Day', fontsize=18, fontweight='bold')
        plt.xlabel('Hour of Day (0-23)', fontsize=14)
        plt.ylabel(f'Average {self.traffic_col.replace("_", " ").title()}', fontsize=14)
        plt.xticks(range(24), fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        # Add value labels on top of bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)
        plt.tight_layout()
        plt.savefig('output_graphs/hourly_patterns.png', dpi=300, bbox_inches='tight')
        # plt.close()

        # Daily patterns
        plt.figure(figsize=(11, 7))
        daily_avg = self.df.groupby('weekday')[self.traffic_col].mean()
        day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        daily_avg = daily_avg.reindex(day_order)
        ax = sns.barplot(x=daily_avg.index, y=daily_avg.values, palette='Greens_d')
        plt.title('Average Website Traffic by Day of Week', fontsize=18, fontweight='bold')
        plt.xlabel('Day of Week', fontsize=14)
        plt.ylabel(f'Average {self.traffic_col.replace("_", " ").title()}', fontsize=14)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        # Add value labels on top of bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)
        plt.tight_layout()
        plt.savefig('output_graphs/daily_patterns.png', dpi=300, bbox_inches='tight')
        # plt.close()

        # Heatmap: Traffic Volume by Hour and Day of Week - ENHANCED
        plt.figure(figsize=(16, 9))
        pivot = self.df.pivot_table(index='day_of_week', columns='hour', values=self.traffic_col, aggfunc='mean')
        # Map day_of_week to actual weekday names for clarity
        weekday_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday',
                         6: 'Sunday'}
        pivot.index = pivot.index.map(weekday_names)

        sns.heatmap(pivot, cmap='YlGnBu', annot=True, fmt=".1f",
                    linewidths=.5, linecolor='lightgray', annot_kws={"fontsize": 8},
                    cbar_kws={'label': f'Average {self.traffic_col.replace("_", " ").title()} Volume'})
        plt.title('Average Traffic Heatmap: Hour of Day vs. Day of Week', fontsize=18, fontweight='bold')
        plt.xlabel('Hour of Day', fontsize=14)
        plt.ylabel('Day of Week', fontsize=14)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10, rotation=0)
        plt.tight_layout()
        plt.savefig('output_graphs/traffic_heatmap.png', dpi=300, bbox_inches='tight')
        # plt.close()

        # Distribution with thresholds
        plt.figure(figsize=(12, 8))
        sns.histplot(self.df[self.traffic_col], bins=50, kde=True, color='purple', edgecolor='black', alpha=0.7)

        threshold_colors = sns.color_palette("rocket", n_colors=len(self.server_threshold_config))
        sorted_thresholds_items = sorted(self.server_threshold_config.items(), key=lambda item: item[1]['threshold'])

        for i, (level, config) in enumerate(sorted_thresholds_items):
            plt.axvline(x=config['threshold'], color=threshold_colors[i], linestyle='--',
                        linewidth=2,
                        label=f"{level.replace('_', ' ').title()} ({config['servers']} servers) - Threshold: {config['threshold']:.2f}")

        plt.title(f'{self.traffic_col.replace("_", " ").title()} Distribution with Server Scaling Thresholds',
                  fontsize=18, fontweight='bold')
        plt.xlabel(self.traffic_col.replace('_', ' ').title(), fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.legend(title="Scaling Levels", loc='upper right', bbox_to_anchor=(1.3, 1))
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout(rect=[0, 0, 0.95, 1])
        plt.savefig('output_graphs/traffic_distribution_with_thresholds.png', dpi=300, bbox_inches='tight')
        # plt.close()

        # Cost analysis visualization
        self.visualize_cost_analysis()

    def visualize_cost_analysis(self):
        """Visualize potential cost savings from RL-driven auto-scaling"""
        print("Creating cost analysis visualization for RL-driven auto-scaling...")

        # Calculate server requirements based on traffic (as an output of RL-based decision, or for comparison if RL not fully integrated here)
        server_requirements = []
        costs = []

        for _, row in self.df.iterrows():
            traffic = row[self.traffic_col]
            current_servers = 0
            current_cost = 0

            # Sort thresholds by value to ensure correct assignment (from lowest to highest)
            sorted_thresholds = sorted(self.server_threshold_config.items(), key=lambda item: item[1]['threshold'])

            # Determine servers and cost based on thresholds
            # This part simulates the *desired* dynamic scaling based on traffic,
            # which your RL agent aims to achieve.
            assigned = False
            for level, config in sorted_thresholds:
                if traffic >= config['threshold']:
                    current_servers = config['servers']
                    current_cost = config['hourly_cost']
                    assigned = True
                else:
                    break

            # If no threshold was met (i.e., traffic is below even the lowest threshold), default to the lowest server configuration
            if not assigned and sorted_thresholds:
                current_servers = sorted_thresholds[0][1]['servers']
                current_cost = sorted_thresholds[0][1]['hourly_cost']
            elif not assigned and not sorted_thresholds:
                current_servers = 1
                current_cost = 0.0104

            server_requirements.append(current_servers)
            costs.append(current_cost)

        self.df['servers_needed_dynamic'] = server_requirements
        self.df['hourly_cost_dynamic'] = costs

        # Calculate total dynamic cost (from RL-driven scaling)
        total_dynamic_cost = self.df['hourly_cost_dynamic'].sum()

        # For comparison, let's calculate a "fixed maximum" cost.
        # This represents running the maximum number of servers needed over the period, constantly.
        max_servers_observed = self.df['servers_needed_dynamic'].max() if not self.df['servers_needed_dynamic'].empty else 1

        fixed_hourly_cost_at_max = 0.0
        found_max_config = False
        for level, config in self.server_threshold_config.items():
            if config['servers'] == max_servers_observed:
                fixed_hourly_cost_at_max = config['hourly_cost']
                found_max_config = True
                break

        if not found_max_config:
            fixed_hourly_cost_at_max = max_servers_observed * self.server_map.get(1, {'cost': 0.0104})['cost']


        total_fixed_cost_max_provisioning = fixed_hourly_cost_at_max * len(self.df)

        savings = total_fixed_cost_max_provisioning - total_dynamic_cost
        savings_percentage = (savings / total_fixed_cost_max_provisioning * 100) if total_fixed_cost_max_provisioning > 0 else 0

        # --- Plotting Enhanced Cost Analysis ---
        plt.figure(figsize=(18, 18))

        # Plot 1: Dynamic Server Adjustments
        plt.subplot(3, 1, 1)
        sns.lineplot(x=self.df[self.output_timestamp_col], y=self.df['servers_needed_dynamic'],
                     color='darkblue', linewidth=2)
        plt.title('RL-Driven Dynamic Server Adjustments Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Date and Time', fontsize=12)
        plt.ylabel('Number of Servers', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()

        # Plot 2: Hourly Cost Comparison (Dynamic vs. Fixed Maximum)
        plt.subplot(3, 1, 2)
        sns.lineplot(x=self.df[self.output_timestamp_col], y=self.df['hourly_cost_dynamic'],
                     color='green', linewidth=2, label='RL Dynamic Hourly Cost')
        plt.axhline(y=fixed_hourly_cost_at_max, color='red', linestyle='--',
                    label=f'Fixed Max Hourly Cost (${fixed_hourly_cost_at_max:.2f}/hr)')
        plt.title('Hourly Cloud Cost: Dynamic (RL) vs. Fixed Max Provisioning', fontsize=16, fontweight='bold')
        plt.xlabel('Date and Time', fontsize=12)
        plt.ylabel('Hourly Cost ($)', fontsize=12)
        plt.legend(loc='upper left', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()

        # Plot 3: Total Cost Comparison and Savings
        plt.subplot(3, 1, 3)
        costs_data = pd.DataFrame({
            'Category': ['Total RL Dynamic Cost', 'Total Fixed Max Cost'],
            'Cost': [total_dynamic_cost, total_fixed_cost_max_provisioning]
        })
        ax = sns.barplot(x='Category', y='Cost', data=costs_data, palette=['#1f77b4', '#ff7f0e'])
        plt.title('Total Cloud Cost Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Total Cost ($)', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=10)

        # Add text for savings
        for p in ax.patches:
            ax.annotate(f'${p.get_height():,.2f}',
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=10)

        plt.text(0.5, 0.95, f'Estimated Savings with RL: ${savings:,.2f} ({savings_percentage:.2f}%)',
                 horizontalalignment='center', verticalalignment='center', transform=ax.transAxes,
                 fontsize=14, color='darkgreen', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.6))

        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('output_graphs/rl_cost_analysis.png', dpi=300, bbox_inches='tight')
        # plt.close()

    def train_traffic_prediction_model(self):
        print("\nTraining traffic prediction model...")
        self.feature_columns = [
            'year', 'month', 'day', 'hour', 'day_of_week', 'is_weekend', 'is_business_hours',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            f'{self.traffic_col}_lag1', f'{self.traffic_col}_lag24', f'{self.traffic_col}_lag168',
            f'{self.traffic_col}_rolling_mean_24h', f'{self.traffic_col}_rolling_mean_7d',
            f'{self.traffic_col}_rolling_std_24h', f'{self.traffic_col}_rolling_max_24h',
            f'{self.traffic_col}_rolling_min_24h', f'{self.traffic_col}_trend_24h',
            f'{self.traffic_col}_volatility'
        ]

        features_df = self.df[self.feature_columns].dropna()
        target_df = self.df.loc[features_df.index, self.traffic_col]

        if features_df.empty:
            print("Error: No data available for training after feature engineering and NaN removal.")
            return

        X = self.scaler.fit_transform(features_df)
        y = target_df.values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.traffic_prediction_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.traffic_prediction_model.fit(X_train, y_train)

        y_pred = self.traffic_prediction_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"Traffic Prediction Model RMSE: {rmse:.2f}")
        print(f"Traffic Prediction Model R² Score: {r2:.2f}")

        joblib.dump(self.traffic_prediction_model, 'models/traffic_prediction_model.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        print("Traffic prediction model and scaler saved to 'models/' directory.")

        # Plotting predicted vs actual traffic
        plt.figure(figsize=(14, 7))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        plt.xlabel('Actual Traffic', fontsize=12)
        plt.ylabel('Predicted Traffic', fontsize=12)
        plt.title('Actual vs. Predicted Website Traffic (Random Forest)', fontsize=16, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('output_graphs/traffic_prediction_actual_vs_predicted.png', dpi=300)
        # plt.close()

        # Plotting feature importance
        if hasattr(self.traffic_prediction_model, 'feature_importances_'):
            importances = self.traffic_prediction_model.feature_importances_
            feature_names = features_df.columns
            feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=feature_importance_df.head(15), palette='magma')
            plt.title('Top 15 Feature Importances for Traffic Prediction', fontsize=16, fontweight='bold')
            plt.xlabel('Importance', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.grid(axis='x', linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig('output_graphs/traffic_feature_importance.png', dpi=300)
            # plt.close()

    def train_rl_agent(self, num_episodes=500):
        print(f"\nTraining RL Agent for {num_episodes} episodes...")

        rl_traffic_data = self.df[self.traffic_col]

        env = ServerScalingEnv(traffic_data=rl_traffic_data, server_configs=self.server_threshold_config)

        state_size = (env.num_traffic_levels, env.num_server_states)
        action_size = env.num_server_states

        self.rl_agent = RLAgent(state_size, action_size)

        rewards_per_episode = []
        episode_servers_chosen = []

        for episode in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            current_servers_rl = env.available_server_counts[
                state[1]]

            episode_servers = []

            while not done:
                action_index = self.rl_agent.choose_action(state)
                next_state, reward, done, _ = env.step(action_index,
                                                       current_servers_rl)
                self.rl_agent.learn(state, action_index, reward, next_state)

                episode_reward += reward
                state = next_state
                current_servers_rl = env.available_server_counts[state[1]]
                episode_servers.append(current_servers_rl)

            self.rl_agent.decay_epsilon()
            rewards_per_episode.append(episode_reward)
            episode_servers_chosen.append(episode_servers)

            if (episode + 1) % 50 == 0:
                print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {episode_reward:.2f}, Epsilon: {self.rl_agent.epsilon:.2f}")

        joblib.dump(self.rl_agent, 'models/rl_agent.pkl')
        print("RL agent (Q-table) saved to 'models/rl_agent.pkl'.")

        # Visualize training progress
        plt.figure(figsize=(12, 6))
        plt.plot(rewards_per_episode, color='orange')
        plt.title('RL Agent Training Progress: Rewards per Episode', fontsize=16, fontweight='bold')
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Total Reward', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('output_graphs/rl_training_rewards.png', dpi=300)
        # plt.close()

        print("RL Agent training complete.")

    def run_analysis(self):
        self.load_and_explore_data()
        self.preprocess_data()
        self.create_time_series_features()
        # --- IMPORTANT CHANGE: Save processed data AFTER feature creation ---
        self.df.to_csv('data/processed_traffic_data.csv', index=False)
        print("Processed data saved to data/processed_traffic_data.csv")
        # --- END IMPORTANT CHANGE ---
        self.calculate_server_thresholds()
        self.visualize_traffic_patterns()
        self.train_traffic_prediction_model()
        self.train_rl_agent()
        print("\nAnalysis complete. Check 'output_graphs' and 'models' directories for results.")


if __name__ == '__main__':
    # Ensure you have a 'traffic_data.csv' file in your 'data' directory
    # For demonstration, creating a dummy file if not exists
    if not os.path.exists('data/traffic_data.csv'):
        os.makedirs('data', exist_ok=True)
        # Generate dummy data for a few days, hourly
        num_hours = 24 * 7 * 4 # 4 weeks of hourly data
        timestamps = [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(num_hours)]

        # Simulate hourly traffic with daily and weekly patterns
        traffic_data = []
        for i in range(num_hours):
            hour = i % 24
            day_of_week = (i // 24) % 7 # 0 = Monday, 6 = Sunday

            # Base traffic
            base = 1000

            # Hourly peak (e.g., afternoon)
            hourly_factor = 1 + np.sin(2 * np.pi * (hour - 8) / 24) * 0.5 # Peak around 2 PM (hour 14)

            # Weekly peak (e.g., weekdays higher than weekends)
            weekly_factor = 1.0
            if day_of_week >= 5: # Saturday or Sunday
                weekly_factor = 0.7 # Lower traffic on weekends
            elif day_of_week in [0, 4]: # Monday/Friday might have slight variations
                weekly_factor = 1.1

            # Add some randomness
            noise = np.random.normal(0, 50)

            traffic = base * hourly_factor * weekly_factor + noise
            traffic = max(50, int(traffic)) # Ensure traffic is at least 50
            traffic_data.append(traffic)

        dummy_df = pd.DataFrame({
            'timestamp': timestamps,
            'Page Views': traffic_data
        })
        dummy_df.to_csv('data/traffic_data.csv', index=False)
        print("Created dummy 'data/traffic_data.csv' for demonstration.")

    analyzer = WebsiteTrafficAnalyzer('data/traffic_data.csv')
    analyzer.run_analysis()
    plt.show() # Display all plots after all are generated and saved
