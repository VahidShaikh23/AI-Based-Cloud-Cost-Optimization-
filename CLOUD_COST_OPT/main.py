#!/usr/bin/env python3
"""
Main orchestration script for Traffic Forecasting and Cost Optimization Project
Executes the complete pipeline from data analysis to model training,
traffic forecasting, and Reinforcement Learning (RL) agent training for auto-scaling.
"""

import os
import sys
import logging
from datetime import datetime
import argparse
import socket
import webbrowser
import threading
import time
import json # Added for handling simulation_results paths in report

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
    print("Make sure data_analysis.py, model_training.py, and utils/rl_autoscaling_agent.py are in the correct directories")
    sys.exit(1)

# FastAPI imports for the dashboard server
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

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

# DEFAULT DATASET PATH
# IMPORTANT: Ensure this path is correct for your system or pass it via command line
DEFAULT_DATASET_PATH = "/Users/vahid/Desktop/website_wata.csv"

# Dashboard configuration
DASHBOARD_PORT = 8001
MODEL_API_PORT = 8000
DASHBOARD_FILE = "dashboard.html"
STATIC_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Utility functions to get IP addresses ---
def get_local_ip():
    """Get the local IP address of the machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't actually connect, just uses a dummy address to find the local IP
        s.connect(("8.8.8.8", 80))
        IP = s.getsockname()[0]
    except Exception:
        IP = "127.0.0.1" # Fallback to localhost
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
        return "127.0.0.1" # Fallback
    except Exception:
        return "127.0.0.1" # Fallback


class TrafficForecastingPipeline:
    """Main pipeline orchestrator"""

    def __init__(self, data_path, config=None):
        self.data_path = data_path
        self.config = config or {}
        self.data_analyzer = None
        self.model_trainer = None
        self.pipeline_results = {} # Store results of each step for final report

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
            self.data_analyzer.calculate_server_thresholds() # This is rule-based, will be superceded by RL
            self.data_analyzer.visualize_traffic_patterns()
            # Removed: X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = self.data_analyzer.prepare_for_ml()
            # Removed: self.data_analyzer.generate_summary_report()

            self.pipeline_results['data_analysis'] = {
                'status': 'completed',
                'timestamp': datetime.now().isoformat(),
                'output_files': [
                    'data/processed_traffic_data.csv', # This file is now correctly saved in data_analysis.py after feature creation
                    'models/scaler.pkl', # This is saved by data_analysis.py's train_traffic_prediction_model
                    'config/feature_columns.json', # This is saved by data_analysis.py
                    'config/server_thresholds.json', # This is saved by data_analysis.py
                    'output_graphs/traffic_over_time.png',
                    'output_graphs/hourly_patterns.png',
                    'output_graphs/daily_patterns.png',
                    'output_graphs/traffic_heatmap.png',
                    'output_graphs/traffic_distribution_with_thresholds.png',
                    'output_graphs/rl_cost_analysis.png', # Updated plot name
                    'output_graphs/traffic_prediction_actual_vs_predicted.png', # Saved by data_analysis.py
                    'output_graphs/traffic_feature_importance.png', # Saved by data_analysis.py
                    'models/rl_agent.pkl', # Saved by data_analysis.py
                    'output_graphs/rl_training_rewards.png' # Saved by data_analysis.py
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
            # Ensure 'models/best_traffic_model.pkl' is generated by run_model_training().
            simulator = AutoScalingSimulator(traffic_model_path="models/best_traffic_model.pkl")

            # Run simulation/training for a specified number of hours (e.g., 1 week = 168 hours).
            # The 'save_results=True' will save the trained Q-table and detailed simulation data.
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
                    'models/autoscaling_agent.json', # The trained Q-table for the RL agent
                    'simulation_results/' # Directory where simulation logs/results are saved
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
                            all_files_listed.append(file_path) # Add regular files

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
app = FastAPI()

# Mount the directory containing dashboard.html and other static assets (CSS, JS)
# Assuming dashboard.html, and a 'static' folder (containing style.css, script.js)
# are directly in the project root.
# The `html=True` serves index.html if found at the root of the mounted directory.
# Since dashboard.html is not index.html, we explicitly serve it at '/' route.
app.mount("/static", StaticFiles(directory=os.path.join(STATIC_DIR, "static")), name="static")

@app.get("/")
async def read_root():
    """Serve the dashboard.html file directly at the root."""
    dashboard_path = os.path.join(STATIC_DIR, DASHBOARD_FILE)
    if not os.path.exists(dashboard_path):
        logger.error(f"❌ Dashboard file not found at: {dashboard_path}")
        # Return a simple HTML error page if dashboard.html is missing
        return FileResponse(
            status_code=404,
            content="<h1>404 Not Found</h1><p>Dashboard file not found. Please ensure dashboard.html exists in the root directory.</p>",
            media_type="text/html"
        )
    return FileResponse(dashboard_path)

def run_dashboard_server():
    """Run the FastAPI dashboard server in a separate thread."""
    try:
        # Use log_level='critical' to suppress uvicorn's default access logs
        # This keeps the main pipeline output cleaner.
        uvicorn.run(app, host="0.0.0.0", port=DASHBOARD_PORT, log_level='critical')
    except Exception as e:
        logger.error(f"❌ Failed to start dashboard server: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Traffic Forecasting & Cost Optimization Pipeline')
    parser.add_argument('data_file', nargs='?', default=DEFAULT_DATASET_PATH,
                        help=f'Path to the CSV data file (default: {DEFAULT_DATASET_PATH})')
    parser.add_argument('--config', help='Path to configuration file (optional)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Use the provided data file or default
    data_file = args.data_file

    # Check if the default file exists
    if data_file == DEFAULT_DATASET_PATH:
        if not os.path.exists(DEFAULT_DATASET_PATH):
            print(f"❌ Default dataset not found at: {DEFAULT_DATASET_PATH}")
            print("\nPlease either:")
            print("1. Make sure your dataset is at the default location")
            print("2. Run with: python main.py /path/to/your/dataset.csv")
            sys.exit(1)
        else:
            print(f"✅ Using default dataset: {DEFAULT_DATASET_PATH}")

    pipeline = TrafficForecastingPipeline(data_file, args.config)
    success = pipeline.run_complete_pipeline()

    if success:
        print("\n🎊 SUCCESS! Your traffic forecasting and RL auto-scaling project pipeline is complete!")
        print("\nNext steps:")
        print("1. Review the generated forecast_report.json and traffic_forecast.csv for ML insights.")
        print("2. Check the RL agent's trained Q-table at 'models/autoscaling_agent.json' and simulation results in 'simulation_results/'.")
        print("3. Use the trained ML model (models/best_traffic_model.pkl) and RL agent (models/autoscaling_agent.json) for real-time predictions and auto-scaling recommendations.")
        print("4. Deploy the 'model_api.py' service for dynamic server scaling using the trained RL agent.")

        # --- Display Dashboard Links ---
        print("\n" + "=" * 60)
        print("🌐 Dashboard Links:")
        local_ip = get_local_ip()
        network_ip = get_network_ip()

        local_dashboard_url = f"http://127.0.0.1:{DASHBOARD_PORT}"
        network_dashboard_url = f"http://{network_ip}:{DASHBOARD_PORT}"
        model_api_url = f"http://127.0.0.1:{MODEL_API_PORT}" # For internal communication

        print(f"   Local Dashboard: {local_dashboard_url}")
        if local_ip != network_ip:
            print(f"   Network Dashboard: {network_dashboard_url} (Accessible from other devices on your network)")
        else:
            print(f"   Network Dashboard: (Same as local, your IP is {network_ip})")
        print(f"   Model API (for dashboard): {model_api_url}")
        print("   (Ensure 'model_api.py' is running on port 8000 in a separate terminal or deployment)")
        print("=" * 60)

        # Start dashboard server in a new thread
        print(f"\nStarting dashboard server on port {DASHBOARD_PORT}...")
        dashboard_thread = threading.Thread(target=run_dashboard_server, daemon=True)
        dashboard_thread.start()
        # Give the server a moment to start
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
                time.sleep(1) # Keep main thread alive
        except KeyboardInterrupt:
            print("\nShutting down pipeline and dashboard server.")
            sys.exit(0) # Exit gracefully

    else:
        print("\n❌ Pipeline failed. Check the logs and console output for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()

