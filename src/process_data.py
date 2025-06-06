import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import pickle
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import gc  # Garbage collection for memory management

warnings.filterwarnings('ignore')


class MemoryEfficientTrafficPreprocessor:
    """
    Memory-efficient preprocessing pipeline for Milan internet traffic data
    Handles large datasets with chunked processing
    """

    def __init__(self, data_directory, sequence_length=144, prediction_horizon=12):
        self.data_directory = data_directory
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler = None
        self.feature_columns = None
        self.processed_data = None

    def convert_timestamps_chunked(self, df, chunk_size=1000000):
        """
        Convert timestamps in chunks to avoid memory issues

        Args:
            df (DataFrame): Input dataframe
            chunk_size (int): Number of rows to process at once
        """
        print(f"🔄 Converting timestamps in chunks of {chunk_size:,}...")

        def timestamp_to_datetime(timestamp_ms):
            try:
                return datetime.fromtimestamp(timestamp_ms / 1000)
            except:
                return pd.NaT

        total_rows = len(df)
        processed_chunks = []

        # Process in chunks
        for i in range(0, total_rows, chunk_size):
            end_idx = min(i + chunk_size, total_rows)
            chunk = df.iloc[i:end_idx].copy()

            print(f"  📊 Processing chunk {i // chunk_size + 1}/{(total_rows - 1) // chunk_size + 1} "
                  f"(rows {i:,} to {end_idx - 1:,})")

            # Convert timestamps for this chunk
            chunk['datetime'] = chunk['time_interval'].apply(timestamp_to_datetime)
            chunk = chunk.dropna(subset=['datetime'])

            # Add time-based features
            chunk['year'] = chunk['datetime'].dt.year
            chunk['month'] = chunk['datetime'].dt.month
            chunk['day'] = chunk['datetime'].dt.day
            chunk['hour'] = chunk['datetime'].dt.hour
            chunk['minute'] = chunk['datetime'].dt.minute
            chunk['day_of_week'] = chunk['datetime'].dt.dayofweek
            chunk['day_of_year'] = chunk['datetime'].dt.dayofyear
            chunk['is_weekend'] = chunk['day_of_week'].isin([5, 6]).astype(int)

            # Create time-based cyclical features
            chunk['hour_sin'] = np.sin(2 * np.pi * chunk['hour'] / 24)
            chunk['hour_cos'] = np.cos(2 * np.pi * chunk['hour'] / 24)
            chunk['day_sin'] = np.sin(2 * np.pi * chunk['day_of_week'] / 7)
            chunk['day_cos'] = np.cos(2 * np.pi * chunk['day_of_week'] / 7)
            chunk['month_sin'] = np.sin(2 * np.pi * chunk['month'] / 12)
            chunk['month_cos'] = np.cos(2 * np.pi * chunk['month'] / 12)

            processed_chunks.append(chunk)

            # Force garbage collection to free memory
            gc.collect()

        # Combine all chunks
        print("  🔗 Combining processed chunks...")
        result_df = pd.concat(processed_chunks, ignore_index=True)

        # Clean up
        del processed_chunks
        gc.collect()

        print(f"✅ Converted {len(result_df):,} timestamps successfully")
        return result_df

    def load_and_aggregate_by_day(self, start_date="2013-11-01", end_date="2014-01-01",
                                  top_n_squares=50):
        """
        Load data day by day and aggregate to save memory

        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            top_n_squares (int): Number of top squares to track
        """
        print(f"🔄 Loading and aggregating data from {start_date} to {end_date}...")

        # First pass: identify top squares across entire dataset
        print("  📊 First pass: Identifying top squares...")
        square_totals = {}

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        current_date = start

        # Sample a few days to identify top squares (memory efficient)
        sample_dates = 0
        while current_date <= end and sample_dates < 5:  # Sample first 5 days
            date_str = current_date.strftime("%Y-%m-%d")
            filename = f"sms-call-internet-mi-{date_str}.txt"
            file_path = os.path.join(self.data_directory, filename)

            if os.path.exists(file_path):
                print(f"    📁 Sampling {filename}...")
                df = self._load_single_file(file_path)
                if df is not None:
                    day_totals = df.groupby('square_id')['internet'].sum()
                    for square_id, total in day_totals.items():
                        square_totals[square_id] = square_totals.get(square_id, 0) + total
                sample_dates += 1

            current_date += timedelta(days=1)

        # Get top squares
        top_squares = sorted(square_totals.keys(),
                             key=lambda x: square_totals[x], reverse=True)[:top_n_squares]
        print(f"  🎯 Selected top {len(top_squares)} squares: {top_squares[:10]}... (showing first 10)")

        # Second pass: process full dataset for top squares only
        print("  📊 Second pass: Processing full dataset for top squares...")

        # Create time index for full period
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        # Create complete time index (10-minute intervals)
        time_index = pd.date_range(start=start, end=end, freq='10T')
        print(f"  ⏰ Time range: {start} to {end}")
        print(f"  📏 Time steps: {len(time_index):,} (10-minute intervals)")

        # Initialize result matrix
        traffic_matrix = pd.DataFrame(0.0, index=time_index, columns=top_squares)

        # Process day by day
        current_date = start
        day_count = 0

        while current_date <= end:
            date_str = current_date.strftime("%Y-%m-%d")
            filename = f"sms-call-internet-mi-{date_str}.txt"
            file_path = os.path.join(self.data_directory, filename)

            if os.path.exists(file_path):
                print(f"    📁 Processing {filename}... (Day {day_count + 1})")

                # Load day's data
                df = self._load_single_file(file_path)
                if df is not None:
                    # Filter to top squares only
                    df = df[df['square_id'].isin(top_squares)]

                    if len(df) > 0:
                        # Convert timestamps (smaller chunk)
                        df = self.convert_timestamps_chunked(df, chunk_size=500000)

                        # Round to 10-minute intervals
                        df['datetime_rounded'] = df['datetime'].dt.round('10T')

                        # Aggregate by time and square
                        day_pivot = df.groupby(['datetime_rounded', 'square_id'])['internet'].sum().unstack(
                            fill_value=0)

                        # Add to main matrix
                        for time_idx in day_pivot.index:
                            if time_idx in traffic_matrix.index:
                                for square in day_pivot.columns:
                                    if square in traffic_matrix.columns:
                                        traffic_matrix.loc[time_idx, square] += day_pivot.loc[time_idx, square]

                day_count += 1
                # Clean up memory
                gc.collect()

            current_date += timedelta(days=1)

        print(f"✅ Processed {day_count} days successfully")
        print(f"📊 Final matrix shape: {traffic_matrix.shape}")

        # Create temporal features
        temporal_features = pd.DataFrame(index=traffic_matrix.index)
        temporal_features['hour'] = temporal_features.index.hour
        temporal_features['day_of_week'] = temporal_features.index.dayofweek
        temporal_features['month'] = temporal_features.index.month
        temporal_features['is_weekend'] = temporal_features.index.dayofweek.isin([5, 6]).astype(int)

        # Cyclical features
        temporal_features['hour_sin'] = np.sin(2 * np.pi * temporal_features['hour'] / 24)
        temporal_features['hour_cos'] = np.cos(2 * np.pi * temporal_features['hour'] / 24)
        temporal_features['day_sin'] = np.sin(2 * np.pi * temporal_features['day_of_week'] / 7)
        temporal_features['day_cos'] = np.cos(2 * np.pi * temporal_features['day_of_week'] / 7)
        temporal_features['month_sin'] = np.sin(2 * np.pi * temporal_features['month'] / 12)
        temporal_features['month_cos'] = np.cos(2 * np.pi * temporal_features['month'] / 12)

        return traffic_matrix, temporal_features, top_squares

    def _load_single_file(self, file_path):
        """Load a single data file"""
        try:
            df = pd.read_csv(file_path, sep='\t', header=None)

            if df.shape[1] == 1:
                df = pd.read_csv(file_path, sep=' ', header=None, skipinitialspace=True)

            if df.shape[1] >= 7:
                # Extract: square_id, time_interval, internet
                df = df.iloc[:, [0, 1, 6]]
                df.columns = ['square_id', 'time_interval', 'internet']
                return df
            else:
                return None
        except Exception as e:
            print(f"    ❌ Error loading {file_path}: {e}")
            return None

    def handle_outliers(self, data, method='iqr', threshold=3):
        """Handle outliers in the data"""
        print(f"🔄 Handling outliers using {method} method...")

        data_clean = data.copy()

        if method == 'iqr':
            for col in data_clean.columns:
                Q1 = data_clean[col].quantile(0.25)
                Q3 = data_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                data_clean[col] = data_clean[col].clip(lower=lower_bound, upper=upper_bound)

        print(f"✅ Outlier handling complete")
        return data_clean

    def normalize_data(self, data, method='minmax'):
        """Normalize the data for neural network training"""
        print(f"🔄 Normalizing data using {method} scaler...")

        if method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()

        normalized_data = self.scaler.fit_transform(data)
        normalized_df = pd.DataFrame(normalized_data, index=data.index, columns=data.columns)

        print(f"✅ Data normalized - range: [{normalized_df.min().min():.3f}, {normalized_df.max().max():.3f}]")
        return normalized_df

    def create_sequences(self, traffic_data, temporal_features):
        """Create sequences for deep learning models"""
        print(f"🔄 Creating sequences...")
        print(f"  📏 Sequence length: {self.sequence_length} steps ({self.sequence_length * 10} minutes)")
        print(f"  🎯 Prediction horizon: {self.prediction_horizon} steps ({self.prediction_horizon * 10} minutes)")

        traffic_array = traffic_data.values
        temporal_array = temporal_features.values

        X_traffic = []
        X_temporal = []
        y = []

        for i in range(self.sequence_length, len(traffic_array) - self.prediction_horizon + 1):
            X_traffic.append(traffic_array[i - self.sequence_length:i])
            X_temporal.append(temporal_array[i - self.sequence_length:i])
            y.append(traffic_array[i:i + self.prediction_horizon])

        X_traffic = np.array(X_traffic)
        X_temporal = np.array(X_temporal)
        y = np.array(y)

        print(f"✅ Created sequences:")
        print(f"  📊 X_traffic shape: {X_traffic.shape} (samples, time_steps, squares)")
        print(f"  🕐 X_temporal shape: {X_temporal.shape} (samples, time_steps, temporal_features)")
        print(f"  🎯 y shape: {y.shape} (samples, prediction_horizon, squares)")

        return X_traffic, X_temporal, y

    def split_data(self, X_traffic, X_temporal, y, train_size=0.8, val_size=0.1):
        """Split data into train, validation, and test sets"""
        print(f"🔄 Splitting data...")

        n_samples = len(X_traffic)
        train_end = int(n_samples * train_size)
        val_end = int(n_samples * (train_size + val_size))

        X_traffic_train = X_traffic[:train_end]
        X_temporal_train = X_temporal[:train_end]
        y_train = y[:train_end]

        X_traffic_val = X_traffic[train_end:val_end]
        X_temporal_val = X_temporal[train_end:val_end]
        y_val = y[train_end:val_end]

        X_traffic_test = X_traffic[val_end:]
        X_temporal_test = X_temporal[val_end:]
        y_test = y[val_end:]

        print(f"✅ Data split:")
        print(f"  🏋️ Train: {len(X_traffic_train):,} samples ({len(X_traffic_train) / n_samples:.1%})")
        print(f"  ✅ Validation: {len(X_traffic_val):,} samples ({len(X_traffic_val) / n_samples:.1%})")
        print(f"  🧪 Test: {len(X_traffic_test):,} samples ({len(X_traffic_test) / n_samples:.1%})")

        return (X_traffic_train, X_temporal_train, y_train,
                X_traffic_val, X_temporal_val, y_val,
                X_traffic_test, X_temporal_test, y_test)

    def save_processed_data(self, data_dict, filepath):
        """Save processed data and metadata"""
        print(f"💾 Saving processed data to {filepath}...")

        data_dict['metadata'] = {
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'scaler_type': type(self.scaler).__name__ if self.scaler else None,
            'n_squares': data_dict['X_traffic_train'].shape[2],
            'n_temporal_features': data_dict['X_temporal_train'].shape[2],
            'processing_date': datetime.now().isoformat()
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data_dict, f)

        print(f"✅ Data saved successfully!")

    def process_full_pipeline_memory_efficient(self, start_date="2013-11-01", end_date="2014-01-01",
                                               top_n_squares=30, save_path="processed_traffic_data.pkl"):
        """
        Memory-efficient complete preprocessing pipeline
        """
        print("🚀 Starting memory-efficient preprocessing pipeline...")
        print("=" * 80)

        # Step 1: Load and aggregate data efficiently
        traffic_matrix, temporal_features, top_squares = self.load_and_aggregate_by_day(
            start_date, end_date, top_n_squares)

        # Step 2: Handle outliers
        traffic_clean = self.handle_outliers(traffic_matrix, method='iqr')

        # Step 3: Normalize data
        traffic_normalized = self.normalize_data(traffic_clean, method='minmax')

        # Step 4: Create sequences
        X_traffic, X_temporal, y = self.create_sequences(traffic_normalized, temporal_features)

        # Step 5: Split data
        train_val_test = self.split_data(X_traffic, X_temporal, y)
        (X_traffic_train, X_temporal_train, y_train,
         X_traffic_val, X_temporal_val, y_val,
         X_traffic_test, X_temporal_test, y_test) = train_val_test

        # Step 6: Prepare data dictionary
        processed_data = {
            'X_traffic_train': X_traffic_train,
            'X_temporal_train': X_temporal_train,
            'y_train': y_train,
            'X_traffic_val': X_traffic_val,
            'X_temporal_val': X_temporal_val,
            'y_val': y_val,
            'X_traffic_test': X_traffic_test,
            'X_temporal_test': X_temporal_test,
            'y_test': y_test,
            'scaler': self.scaler,
            'top_squares': top_squares,
            'temporal_feature_names': temporal_features.columns.tolist()
        }

        # Step 7: Save processed data
        self.save_processed_data(processed_data, save_path)

        print("=" * 80)
        print("🎉 Memory-efficient preprocessing completed successfully!")
        print(f"💾 Processed data saved to: {save_path}")

        return processed_data


# Example usage
def main():
    """Example usage of the memory-efficient preprocessor"""

    # Initialize preprocessor
    data_dir = r"C:\Users\Fan\PycharmProjects\dataset2\dataverse_files"

    # Configuration for deep learning
    preprocessor = MemoryEfficientTrafficPreprocessor(
        data_directory=data_dir,
        sequence_length=144,  # 24 hours of history
        prediction_horizon=12  # Predict next 2 hours
    )

    # Run memory-efficient preprocessing pipeline
    processed_data = preprocessor.process_full_pipeline_memory_efficient(
        start_date="2013-11-01",
        end_date="2014-01-01",  # Full 2+ month dataset
        top_n_squares=30,
        save_path="../processed_data/milan_traffic_data_processed_full_memory_efficient.pkl"
    )

    print("\n📋 SUMMARY FOR DEEP LEARNING:")
    print(f"🎯 Ready for models like LSTM, GRU, Transformer, CNN-LSTM")
    print(f"📊 Input sequences: {processed_data['X_traffic_train'].shape}")
    print(f"🕐 Temporal features: {processed_data['X_temporal_train'].shape}")
    print(f"🎯 Prediction targets: {processed_data['y_train'].shape}")
    print(f"🏢 Squares included: {len(processed_data['top_squares'])}")
    print(f"⚡ Normalized and ready for training!")


if __name__ == "__main__":
    main()