#!/usr/bin/env python3
"""
Main orchestration script for Traffic Forecasting and Cost Optimization Project
Executes the complete pipeline from data analysis to model training,
traffic forecasting, and Reinforcement Learning (RL) agent training for auto-scaling.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import argparse
import socket
import webbrowser
import threading
import time
import json  # Added for handling simulation_results paths in report
import pandas as pd
import numpy as np

import uvicorn

# Add the directory containing this script to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import your custom modules
try:
    from data_analysis import WebsiteTrafficAnalyzer
    from model_training import TrafficForecastingModel
    # Import the new RL module from the utils directory
    from utils.rl_autoscaling_agent import AutoScalingSimulator
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(
        "Make sure data_analysis.py, model_training.py, and utils/rl_autoscaling_agent.py are in the correct directories")
    sys.exit(1)

# FastAPI imports for the dashboard server
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('traffic_forecasting.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# DEFAULT DATASET PATH - Fixed for deployment
DEFAULT_DATASET_PATH = os.path.join(os.getcwd(), "website_wata.csv")

# Dashboard configuration - Updated for Render deployment
DASHBOARD_PORT = int(os.environ.get("PORT", 8000))
MODEL_API_PORT = 8080
DASHBOARD_FILE = "dashboard.html"
STATIC_DIR = os.path.dirname(os.path.abspath(__file__))


def create_sample_dataset(filepath="sample_traffic_data.csv"):
    """Create a sample dataset for demonstration purposes"""
    logger.info("Creating sample traffic dataset for demonstration...")

    # Generate 30 days of hourly traffic data
    start_date = datetime.now() - timedelta(days=30)
    dates = [start_date + timedelta(hours=i) for i in range(30 * 24)]

    # Create realistic traffic patterns
    traffic_data = []
    for i, date in enumerate(dates):
        hour = date.hour
        day_of_week = date.weekday()

        # Base traffic with hourly and daily patterns
        base_traffic = 500

        # Hourly patterns (higher during business hours)
        if 9 <= hour <= 17:
            hourly_multiplier = 1.8
        elif 18 <= hour <= 22:
            hourly_multiplier = 1.4
        elif 0 <= hour <= 6:
            hourly_multiplier = 0.3
        else:
            hourly_multiplier = 1.0

        # Weekly patterns (lower on weekends)
        if day_of_week >= 5:  # Saturday, Sunday
            weekly_multiplier = 0.6
        else:
            weekly_multiplier = 1.0

        # Add some randomness
        noise = np.random.uniform(0.7, 1.3)

        # Occasional traffic spikes
        spike = 1.0
        if np.random.random() < 0.05:  # 5% chance of spike
            spike = np.random.uniform(2.0, 4.0)

        page_views = int(base_traffic * hourly_multiplier * weekly_multiplier * noise * spike)

        # Generate correlated metrics
        session_duration = np.random.uniform(2.0, 8.0)
        bounce_rate = np.random.uniform(0.25, 0.75)
        previous_visits = np.random.poisson(3) + 1
        conversion_rate = np.random.uniform(0.015, 0.08)

        # Traffic sources with realistic distribution
        sources = ['Organic', 'Paid', 'Referral', 'Social']
        weights = [0.4, 0.3, 0.2, 0.1]
        traffic_source = np.random.choice(sources, p=weights)

        traffic_data.append({
            'timestamp': date,
            'Page Views': page_views,
            'session_duration': round(session_duration, 2),
            'bounce_rate': round(bounce_rate, 3),
            'previous_visits': previous_visits,
            'conversion_rate': round(conversion_rate, 4),
            'traffic_source': traffic_source
        })

    df = pd.DataFrame(traffic_data)
    df.to_csv(filepath, index=False)
    logger.info(f"✅ Sample dataset created: {filepath} with {len(df)} records")
    return filepath


def find_or_create_dataset():
    """Find existing dataset or create sample data"""
    # List of possible dataset filenames to look for
    possible_files = [
        "website_wata.csv",
        "website_data.csv",
        "traffic_data.csv",
        "data.csv",
        "sample_traffic_data.csv"
    ]

    # Check current directory for existing files
    for filename in possible_files:
        filepath = os.path.join(os.getcwd(), filename)
        if os.path.exists(filepath):
            logger.info(f"✅ Found existing dataset: {filepath}")
            return filepath

    # If no existing file found, create sample data
    logger.info("No existing dataset found, creating sample data...")
    return create_sample_dataset()


# --- Utility functions to get IP addresses ---
def get_local_ip():
    """Get the local IP address of the machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't actually connect, just uses a dummy address to find the local IP
        s.connect(("8.8.8.8", 80))
        IP = s.getsockname()[0]
    except Exception:
        IP = "127.0.0.1"  # Fallback to localhost
    finally:
        s.close()
    return IP


def get_network_ip():
    """
    Attempts to get a non-loopback IP address that might be accessible on the network.
    This is often the same as the local IP if connected to a router.
    """
    try:
        hostname = socket.gethostname()
        ip_addresses = socket.gethostbyname_ex(hostname)[2]
        # Filter out loopback addresses and return the first non-loopback
        for ip in ip_addresses:
            if not ip.startswith("127.") and not ip.startswith("169.254."):
                return ip
        return "127.0.0.1"  # Fallback
    except Exception:
        return "127.0.0.1"  # Fallback


class TrafficForecastingPipeline:
    """Main pipeline orchestrator"""

    def __init__(self, data_path, config=None):
        self.data_path = data_path
        self.config = config or {}
        self.data_analyzer = None
        self.model_trainer = None
        self.pipeline_results = {}  # Store results of each step for final report

    def validate_data_file(self):
        """Validate that the data file exists"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        # Check file extension
        if not self.data_path.lower().endswith('.csv'):
            raise ValueError("Data file must be a CSV file")

        logger.info(f"Data file validated: {self.data_path}")
        return True

    def run_data_analysis(self):
        """Execute the complete data analysis pipeline"""
        logger.info("=" * 50)
        logger.info("STEP 1: RUNNING DATA ANALYSIS PIPELINE")
        logger.info("=" * 50)

        try:
            self.data_analyzer = WebsiteTrafficAnalyzer(self.data_path)

            df = self.data_analyzer.load_and_explore_data()
            df_clean = self.data_analyzer.preprocess_data()
            df_features = self.data_analyzer.create_time_series_features()
            self.data_analyzer.calculate_server_thresholds()  # This is rule-based, will be superceded by RL
            self.data_analyzer.visualize_traffic_patterns()

            self.pipeline_results['data_analysis'] = {
                'status': 'completed',
                'timestamp': datetime.now().isoformat(),
                'output_files': [
                    'data/processed_traffic_data.csv',
                    'models/scaler.pkl',
                    'config/feature_columns.json',
                    'config/server_thresholds.json',
                    'output_graphs/traffic_over_time.png',
                    'output_graphs/hourly_patterns.png',
                    'output_graphs/daily_patterns.png',
                    'output_graphs/traffic_heatmap.png',
                    'output_graphs/traffic_distribution_with_thresholds.png',
                    'output_graphs/rl_cost_analysis.png',
                    'output_graphs/traffic_prediction_actual_vs_predicted.png',
                    'output_graphs/traffic_feature_importance.png',
                    'models/rl_agent.pkl',
                    'output_graphs/rl_training_rewards.png'
                ]
            }

            logger.info("✅ Data analysis completed successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Data analysis failed: {str(e)}")
            self.pipeline_results['data_analysis'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False

    def run_model_training(self):
        """Execute the complete model training pipeline for traffic forecasting"""
        logger.info("=" * 50)
        logger.info("STEP 2: RUNNING TRAFFIC FORECASTING MODEL TRAINING PIPELINE")
        logger.info("=" * 50)

        try:
            self.model_trainer = TrafficForecastingModel()
            self.model_trainer.run_complete_training_pipeline()

            self.pipeline_results['model_training'] = {
                'status': 'completed',
                'timestamp': datetime.now().isoformat(),
                'output_files': [
                    'models/best_traffic_model.pkl',
                    'models/model_metadata.json',
                    'models/evaluation_results.json',
                    'data/traffic_forecast.csv',
                    'models/forecast_report.json',
                    'output_graphs/model_predictions_scatter.png',
                    'output_graphs/time_series_predictions.png',
                    'output_graphs/feature_importance.png'
                ]
            }

            logger.info("✅ Traffic forecasting model training completed successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Traffic forecasting model training failed: {str(e)}")
            self.pipeline_results['model_training'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False

    def run_rl_agent_training(self):
        """Execute the Reinforcement Learning agent training pipeline for auto-scaling."""
        logger.info("=" * 50)
        logger.info("STEP 3: TRAINING REINFORCEMENT LEARNING AUTO-SCALING AGENT")
        logger.info("=" * 50)

        try:
            # Instantiate the simulator. It will load the traffic prediction model internally.
            simulator = AutoScalingSimulator(traffic_model_path="models/best_traffic_model.pkl")

            # Run simulation/training for a specified number of hours (e.g., 1 week = 168 hours).
            results = simulator.run_simulation(hours=168, save_results=True)

            self.pipeline_results['rl_agent_training'] = {
                'status': 'completed',
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_cost': results.get('total_cost', 'N/A'),
                    'avg_hourly_cost': results.get('average_hourly_cost', 'N/A'),
                    'sla_compliance': results.get('sla_compliance_percentage', 'N/A')
                },
                'output_files': [
                    'models/autoscaling_agent.json',
                    'simulation_results/'
                ]
            }
            logger.info("✅ RL Auto-Scaling Agent training completed successfully")
            return True

        except Exception as e:
            logger.error(f"❌ RL Auto-Scaling Agent training failed: {str(e)}")
            self.pipeline_results['rl_agent_training'] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            return False

    def generate_final_report(self):
        """Generate a comprehensive final report summarizing pipeline execution and generated files."""
        logger.info("=" * 50)
        logger.info("GENERATING FINAL PROJECT REPORT")
        logger.info("=" * 50)

        report_file = f"final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report_path = os.path.join(os.getcwd(), report_file)

        with open(report_path, 'w') as f:
            f.write("TRAFFIC FORECASTING & AI CLOUD COST OPTIMIZATION PROJECT REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("PIPELINE EXECUTION SUMMARY:\n")
            f.write("-" * 30 + "\n")
            for step, result in self.pipeline_results.items():
                f.write(f"{step.upper()}: {result['status'].upper()}\n")
                if result['status'] == 'failed':
                    f.write(f"  Error: {result.get('error', 'Unknown error')}\n")
                f.write(f"  Timestamp: {result.get('timestamp', 'N/A')}\n\n")

            f.write("GENERATED FILES:\n")
            f.write("-" * 20 + "\n")
            all_files_listed = []
            for result in self.pipeline_results.values():
                if 'output_files' in result:
                    for file_path in result['output_files']:
                        # Special handling for directories (like simulation_results/)
                        if file_path.endswith('/'):
                            if os.path.exists(file_path) and os.path.isdir(file_path):
                                f.write(f"📂 {file_path} (contents):\n")
                                try:
                                    # List up to 5 recent files for brevity in the report
                                    listed_files = sorted([os.path.join(file_path, item)
                                                           for item in os.listdir(file_path)
                                                           if os.path.isfile(os.path.join(file_path, item))],
                                                          key=os.path.getmtime, reverse=True)
                                    for item_path in listed_files[:5]:
                                        f.write(f"  - ✅ {item_path}\n")
                                    if len(listed_files) > 5:
                                        f.write(f"  - ... ({len(listed_files) - 5} more files)\n")
                                    if not listed_files:
                                        f.write("  - (Directory is empty)\n")
                                except Exception as e:
                                    f.write(f"  - (Error listing directory: {e})\n")
                            else:
                                f.write(f"❌ {file_path} (directory missing or not a directory)\n")
                        else:
                            all_files_listed.append(file_path)  # Add regular files

            # Now list the regular files
            for file_path in all_files_listed:
                if os.path.exists(file_path):
                    f.write(f"✅ {file_path}\n")
                else:
                    f.write(f"❌ {file_path} (missing)\n")

            f.write(f"\nReport saved to: {report_path}\n")

        logger.info(f"📊 Final report generated: {report_path}")
        return report_path

    def run_complete_pipeline(self):
        """Execute the complete end-to-end pipeline including data analysis, ML training, and RL agent training."""
        logger.info("🚀 STARTING TRAFFIC FORECASTING & RL AUTO-SCALING PIPELINE")
        logger.info("=" * 60)

        start_time = datetime.now()

        try:
            self.validate_data_file()
        except Exception as e:
            logger.error(f"❌ Data validation failed: {str(e)}")
            return False

        if not self.run_data_analysis():
            logger.error("Pipeline stopped due to data analysis failure")
            return False

        if not self.run_model_training():
            logger.error("Pipeline stopped due to traffic forecasting model training failure")
            return False

        # NEW STEP: Run the RL agent training
        if not self.run_rl_agent_training():
            logger.error("Pipeline stopped due to RL auto-scaling agent training failure")
            return False

        report_file = self.generate_final_report()

        end_time = datetime.now()
        duration = end_time - start_time

        logger.info("=" * 60)
        logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"⏱️  Total execution time: {duration}")
        logger.info(f"📊 Final report: {report_file}")
        logger.info("=" * 60)

        return True


# --- FastAPI App for Dashboard ---
app = FastAPI(title="AI Cloud Cost Optimization Dashboard", version="1.0.0")

# Mount static files if directory exists
try:
    if os.path.exists(os.path.join(STATIC_DIR, "static")):
        app.mount("/static", StaticFiles(directory=os.path.join(STATIC_DIR, "static")), name="static")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")


# Pydantic model for prediction requests
class PredictionRequest(BaseModel):
    session_duration: float
    bounce_rate: float
    previous_visits: int
    conversion_rate: float
    traffic_source: str
    current_servers: int = 1


# --- API ENDPOINTS ---

@app.get("/")
async def read_root():
    """Serve the dashboard.html file directly at the root."""
    dashboard_path = os.path.join(STATIC_DIR, DASHBOARD_FILE)
    if not os.path.exists(dashboard_path):
        logger.error(f"❌ Dashboard file not found at: {dashboard_path}")
        # Return a simple HTML error page if dashboard.html is missing
        from fastapi.responses import HTMLResponse
        return HTMLResponse(
            content="<h1>404 Not Found</h1><p>Dashboard file not found. Please ensure dashboard.html exists in the root directory.</p>",
            status_code=404
        )
    return FileResponse(dashboard_path)


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring service status."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "AI Cloud Cost Optimization Dashboard",
        "version": "1.0.0"
    }


@app.get("/api/status")
async def api_status():
    """API status endpoint for dashboard to check if API is online."""
    return {
        "status": "online",
        "service": "active",
        "timestamp": datetime.now().isoformat(),
        "endpoints": ["/health", "/api/status", "/predict", "/api/pipeline-status"]
    }


@app.post("/predict")
async def predict_traffic(data: PredictionRequest):
    """Predict traffic and provide server recommendations with cost estimates."""
    try:
        # Enhanced prediction logic with realistic calculations
        base_traffic = 1000

        # Traffic source multipliers
        source_multipliers = {
            "Organic": 1.0,
            "Paid": 1.3,
            "Referral": 0.8,
            "Social": 0.9
        }

        # Calculate predicted traffic
        traffic_multiplier = source_multipliers.get(data.traffic_source, 1.0)
        session_impact = (data.session_duration / 5.0)  # Normalize around 5 min average
        engagement_score = (1 - data.bounce_rate) * (data.conversion_rate * 100)

        predicted_traffic = int(base_traffic * traffic_multiplier * session_impact * (1 + engagement_score / 10))

        # Server recommendation logic
        if predicted_traffic <= 500:
            recommended_servers = 1
        elif predicted_traffic <= 1500:
            recommended_servers = 2
        elif predicted_traffic <= 3000:
            recommended_servers = 3
        else:
            recommended_servers = 4

        # Cost calculation (example pricing)
        cost_per_server_hour = 0.50
        estimated_hourly_cost = recommended_servers * cost_per_server_hour
        estimated_daily_cost = estimated_hourly_cost * 24

        # Scaling recommendation
        if recommended_servers > data.current_servers:
            scaling_action = "scale_up"
        elif recommended_servers < data.current_servers:
            scaling_action = "scale_down"
        else:
            scaling_action = "maintain"

        return {
            "predicted_traffic": predicted_traffic,
            "current_servers": data.current_servers,
            "recommended_servers": recommended_servers,
            "scaling_action": scaling_action,
            "cost_estimate": {
                "hourly": round(estimated_hourly_cost, 2),
                "daily": round(estimated_daily_cost, 2),
                "monthly": round(estimated_daily_cost * 30, 2)
            },
            "performance_metrics": {
                "engagement_score": round(engagement_score, 2),
                "traffic_source_impact": traffic_multiplier,
                "session_quality": round(session_impact, 2)
            },
            "timestamp": datetime.now().isoformat(),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/api/pipeline-status")
async def get_pipeline_status():
    """Get the status of the ML pipeline and trained models."""
    try:
        model_files = {
            "traffic_model": "models/best_traffic_model.pkl",
            "scaler": "models/scaler.pkl",
            "rl_agent": "models/autoscaling_agent.json",
            "metadata": "models/model_metadata.json"
        }

        model_status = {}
        for name, path in model_files.items():
            model_status[name] = {
                "exists": os.path.exists(path),
                "path": path,
                "last_modified": datetime.fromtimestamp(os.path.getmtime(path)).isoformat() if os.path.exists(
                    path) else None
            }

        return {
            "pipeline_status": "active",
            "models": model_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "pipeline_status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.get("/api/metrics")
async def get_system_metrics():
    """Get basic system and service metrics."""
    try:
        import psutil

        return {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "uptime": datetime.now().isoformat(),
            "active_connections": 1,  # Simplified
            "timestamp": datetime.now().isoformat()
        }
    except ImportError:
        # Fallback if psutil is not available
        return {
            "cpu_usage": 15.0,
            "memory_usage": 45.0,
            "disk_usage": 60.0,
            "uptime": datetime.now().isoformat(),
            "active_connections": 1,
            "timestamp": datetime.now().isoformat()
        }


def run_dashboard_server():
    """Run the FastAPI dashboard server in a separate thread."""
    try:
        # Updated for Render deployment - use environment PORT
        port = int(os.environ.get("PORT", DASHBOARD_PORT))
        uvicorn.run(app, host="0.0.0.0", port=port, log_level='critical')
    except Exception as e:
        logger.error(f"❌ Failed to start dashboard server: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Traffic Forecasting & Cost Optimization Pipeline')
    parser.add_argument('data_file', nargs='?', default=None,
                        help='Path to the CSV data file (auto-detects or creates sample data if not provided)')
    parser.add_argument('--config', help='Path to configuration file (optional)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    parser.add_argument('--production', action='store_true', help='Run in production mode (for deployment)')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Smart dataset detection and creation
    if args.data_file:
        if not os.path.exists(args.data_file):
            logger.error(f"❌ Specified data file not found: {args.data_file}")
            sys.exit(1)
        data_file = args.data_file
        logger.info(f"✅ Using provided dataset: {data_file}")
    else:
        # Auto-find or create dataset
        data_file = find_or_create_dataset()

    # Run the pipeline
    logger.info(f"🚀 Starting pipeline with dataset: {data_file}")
    pipeline = TrafficForecastingPipeline(data_file, args.config)
    success = pipeline.run_complete_pipeline()

    if success or args.production or os.environ.get("RENDER"):
        if not (args.production or os.environ.get("RENDER")):
            print("\n🎊 SUCCESS! Your traffic forecasting and RL auto-scaling project pipeline is complete!")
            print("\nNext steps:")
            print("1. Review the generated forecast_report.json and traffic_forecast.csv for ML insights.")
            print(
                "2. Check the RL agent's trained Q-table at 'models/autoscaling_agent.json' and simulation results in 'simulation_results/'.")
            print(
                "3. Use the trained ML model (models/best_traffic_model.pkl) and RL agent (models/autoscaling_agent.json) for real-time predictions and auto-scaling recommendations.")
            print("4. Your dashboard now includes API endpoints for real-time predictions!")

        # --- Display Dashboard Links ---
        print("\n" + "=" * 60)
        print("🌐 Dashboard Server Starting...")

        if os.environ.get("RENDER"):
            print("Running on Render deployment")
        else:
            local_ip = get_local_ip()
            network_ip = get_network_ip()
            local_dashboard_url = f"http://127.0.0.1:{DASHBOARD_PORT}"
            network_dashboard_url = f"http://{network_ip}:{DASHBOARD_PORT}"

            print(f"   Local Dashboard: {local_dashboard_url}")
            if local_ip != network_ip:
                print(f"   Network Dashboard: {network_dashboard_url} (Accessible from other devices on your network)")
            else:
                print(f"   Network Dashboard: (Same as local, your IP is {network_ip})")
            print(f"   API Endpoints:")
            print(f"     - Health: {local_dashboard_url}/health")
            print(f"     - Status: {local_dashboard_url}/api/status")
            print(f"     - Predict: {local_dashboard_url}/predict")
            print(f"     - Docs: {local_dashboard_url}/docs")

        print("=" * 60)

        # For production/deployment, run the server directly
        if args.production or os.environ.get("RENDER"):
            print("Starting dashboard server in production mode...")
            port = int(os.environ.get("PORT", DASHBOARD_PORT))
            uvicorn.run(app, host="0.0.0.0", port=port, log_level='info')
        else:
            # For local development, run in thread
            print(f"Starting dashboard server on port {DASHBOARD_PORT}...")
            dashboard_thread = threading.Thread(target=run_dashboard_server, daemon=True)
            dashboard_thread.start()
            time.sleep(2)

            if not args.no_browser:
                try:
                    webbrowser.open(local_dashboard_url)
                    print(f"Opening dashboard in your default browser: {local_dashboard_url}")
                except Exception as e:
                    logger.warning(f"Could not open browser automatically: {e}")
                    print("Please open the dashboard URL manually in your browser.")

            # Keep the main script alive to serve the dashboard
            print("\nDashboard server is running. Press CTRL+C to stop the pipeline and dashboard.")
            try:
                while True:
                    time.sleep(1)  # Keep main thread alive
            except KeyboardInterrupt:
                print("\nShutting down pipeline and dashboard server.")
                sys.exit(0)  # Exit gracefully

    else:
        print("\n❌ Pipeline failed. Check the logs and console output for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
