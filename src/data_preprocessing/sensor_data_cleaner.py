"""
Sensor Data Cleaner
Cleans and preprocesses the sensor data from CSV files
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings


class SensorDataCleaner:
    """Cleaner for pipeline sensor data"""
    
    def __init__(self, data_path: str, timestamp_col: str = 'timestamp'):
        """
        Initialize data cleaner
        
        Args:
            data_path: Path to the CSV sensor data file
            timestamp_col: Name of the timestamp column
        """
        self.data_path = data_path
        self.timestamp_col = timestamp_col
        self.raw_data = None
        self.clean_data = None
        self.scalers = {}
        self.sensor_columns = []
        
    def load_data(self) -> pd.DataFrame:
        """
        Load sensor data from CSV file
        
        Returns:
            Raw sensor data DataFrame
        """
        try:
            self.raw_data = pd.read_csv(self.data_path)
            print(f"Loaded {len(self.raw_data)} records with {len(self.raw_data.columns)} columns")
            return self.raw_data
        except Exception as e:
            raise RuntimeError(f"Failed to load data from {self.data_path}: {str(e)}")
    
    def identify_sensor_columns(self) -> List[str]:
        """
        Identify sensor data columns (non-timestamp columns)
        
        Returns:
            List of sensor column names
        """
        if self.raw_data is None:
            self.load_data()
            
        # All columns except timestamp are sensor columns
        self.sensor_columns = [col for col in self.raw_data.columns 
                              if col != self.timestamp_col]
        
        print(f"Identified {len(self.sensor_columns)} sensor columns")
        return self.sensor_columns
    
    def parse_timestamps(self) -> pd.DataFrame:
        """
        Parse and validate timestamps
        
        Returns:
            DataFrame with parsed timestamps
        """
        if self.raw_data is None:
            self.load_data()
            
        data = self.raw_data.copy()
        
        # Parse timestamp column
        try:
            data[self.timestamp_col] = pd.to_datetime(data[self.timestamp_col])
        except Exception as e:
            warnings.warn(f"Failed to parse timestamps: {str(e)}")
            # Try alternative formats or keep as is
            pass
        
        # Sort by timestamp
        data = data.sort_values(self.timestamp_col).reset_index(drop=True)
        
        # Check for time gaps
        if pd.api.types.is_datetime64_any_dtype(data[self.timestamp_col]):
            time_diffs = data[self.timestamp_col].diff().dt.total_seconds()
            median_interval = time_diffs.median()
            print(f"Median time interval: {median_interval} seconds")
            
            # Identify large gaps
            large_gaps = time_diffs > median_interval * 2
            if large_gaps.sum() > 0:
                print(f"Found {large_gaps.sum()} large time gaps")
        
        return data
    
    def handle_missing_values(self, data: pd.DataFrame, 
                            strategy: str = 'interpolate') -> pd.DataFrame:
        """
        Handle missing values in sensor data
        
        Args:
            data: Input DataFrame
            strategy: Strategy for handling missing values
                     ('interpolate', 'forward_fill', 'backward_fill', 'mean', 'median')
        
        Returns:
            DataFrame with missing values handled
        """
        if not self.sensor_columns:
            self.identify_sensor_columns()
            
        data_clean = data.copy()
        
        # Convert sensor columns to numeric, coercing errors to NaN
        for col in self.sensor_columns:
            data_clean[col] = pd.to_numeric(data_clean[col], errors='coerce')
        
        # Count missing values
        missing_counts = data_clean[self.sensor_columns].isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing > 0:
            print(f"Found {total_missing} missing values across {len(self.sensor_columns)} sensors")
            
            # Apply missing value strategy
            if strategy == 'interpolate':
                # Linear interpolation for time series data
                data_clean[self.sensor_columns] = data_clean[self.sensor_columns].interpolate(method='linear')
                
            elif strategy == 'forward_fill':
                data_clean[self.sensor_columns] = data_clean[self.sensor_columns].fillna(method='ffill')
                
            elif strategy == 'backward_fill':
                data_clean[self.sensor_columns] = data_clean[self.sensor_columns].fillna(method='bfill')
                
            elif strategy in ['mean', 'median']:
                # Use sklearn imputer
                imputer = SimpleImputer(strategy=strategy)
                data_clean[self.sensor_columns] = imputer.fit_transform(data_clean[self.sensor_columns])
                
            # Final check and forward fill any remaining NaN
            remaining_na = data_clean[self.sensor_columns].isnull().sum().sum()
            if remaining_na > 0:
                print(f"Forward filling {remaining_na} remaining NaN values")
                data_clean[self.sensor_columns] = data_clean[self.sensor_columns].fillna(method='ffill')
                
                # If still NaN at the beginning, backward fill
                remaining_na = data_clean[self.sensor_columns].isnull().sum().sum()
                if remaining_na > 0:
                    data_clean[self.sensor_columns] = data_clean[self.sensor_columns].fillna(method='bfill')
        
        return data_clean
    
    def remove_outliers(self, data: pd.DataFrame, 
                       method: str = 'iqr', threshold: float = 3.0) -> pd.DataFrame:
        """
        Remove outliers from sensor data
        
        Args:
            data: Input DataFrame
            method: Outlier detection method ('iqr', 'zscore')
            threshold: Threshold for outlier detection
        
        Returns:
            DataFrame with outliers removed
        """
        if not self.sensor_columns:
            self.identify_sensor_columns()
            
        data_clean = data.copy()
        outlier_count = 0
        
        for col in self.sensor_columns:
            if data_clean[col].dtype in ['float64', 'int64']:
                if method == 'iqr':
                    # Interquartile range method
                    Q1 = data_clean[col].quantile(0.25)
                    Q3 = data_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    
                    outliers = (data_clean[col] < lower_bound) | (data_clean[col] > upper_bound)
                    
                elif method == 'zscore':
                    # Z-score method
                    z_scores = np.abs((data_clean[col] - data_clean[col].mean()) / data_clean[col].std())
                    outliers = z_scores > threshold
                
                # Replace outliers with NaN and then interpolate
                outlier_mask = outliers & data_clean[col].notna()
                outlier_count += outlier_mask.sum()
                data_clean.loc[outlier_mask, col] = np.nan
        
        if outlier_count > 0:
            print(f"Removed {outlier_count} outliers")
            # Interpolate the removed outliers
            data_clean[self.sensor_columns] = data_clean[self.sensor_columns].interpolate(method='linear')
        
        return data_clean
    
    def normalize_data(self, data: pd.DataFrame, 
                      method: str = 'standard') -> pd.DataFrame:
        """
        Normalize sensor data
        
        Args:
            data: Input DataFrame
            method: Normalization method ('standard', 'minmax', 'robust')
        
        Returns:
            Normalized DataFrame
        """
        if not self.sensor_columns:
            self.identify_sensor_columns()
            
        data_norm = data.copy()
        
        for col in self.sensor_columns:
            if data_norm[col].dtype in ['float64', 'int64']:
                if method == 'standard':
                    scaler = StandardScaler()
                elif method == 'minmax':
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler()
                elif method == 'robust':
                    from sklearn.preprocessing import RobustScaler
                    scaler = RobustScaler()
                else:
                    continue
                
                # Fit and transform
                values = data_norm[col].values.reshape(-1, 1)
                data_norm[col] = scaler.fit_transform(values).flatten()
                
                # Store scaler for inverse transform
                self.scalers[col] = scaler
        
        print(f"Normalized {len(self.scalers)} sensor columns using {method} scaling")
        return data_norm
    
    def create_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional time-based features
        
        Args:
            data: Input DataFrame with timestamp
            
        Returns:
            DataFrame with additional time features
        """
        data_enhanced = data.copy()
        
        if pd.api.types.is_datetime64_any_dtype(data[self.timestamp_col]):
            # Extract time components
            data_enhanced['hour'] = data[self.timestamp_col].dt.hour
            data_enhanced['day_of_week'] = data[self.timestamp_col].dt.dayofweek
            data_enhanced['day_of_year'] = data[self.timestamp_col].dt.dayofyear
            data_enhanced['month'] = data[self.timestamp_col].dt.month
            
            # Cyclical encoding for time features
            data_enhanced['hour_sin'] = np.sin(2 * np.pi * data_enhanced['hour'] / 24)
            data_enhanced['hour_cos'] = np.cos(2 * np.pi * data_enhanced['hour'] / 24)
            data_enhanced['day_sin'] = np.sin(2 * np.pi * data_enhanced['day_of_week'] / 7)
            data_enhanced['day_cos'] = np.cos(2 * np.pi * data_enhanced['day_of_week'] / 7)
            
            print("Added time-based features")
        
        return data_enhanced
    
    def clean_sensor_data(self, missing_strategy: str = 'interpolate',
                         outlier_method: str = 'iqr',
                         outlier_threshold: float = 3.0,
                         normalize_method: str = 'standard',
                         add_time_features: bool = True) -> pd.DataFrame:
        """
        Complete sensor data cleaning pipeline
        
        Args:
            missing_strategy: Strategy for missing values
            outlier_method: Method for outlier detection
            outlier_threshold: Threshold for outlier removal
            normalize_method: Normalization method
            add_time_features: Whether to add time-based features
            
        Returns:
            Cleaned sensor data DataFrame
        """
        print("Starting sensor data cleaning pipeline...")
        
        # Step 1: Load and parse timestamps
        data = self.parse_timestamps()
        
        # Step 2: Identify sensor columns
        self.identify_sensor_columns()
        
        # Step 3: Handle missing values
        data = self.handle_missing_values(data, missing_strategy)
        
        # Step 4: Remove outliers
        data = self.remove_outliers(data, outlier_method, outlier_threshold)
        
        # Step 5: Normalize data
        data = self.normalize_data(data, normalize_method)
        
        # Step 6: Add time features
        if add_time_features:
            data = self.create_time_features(data)
        
        self.clean_data = data
        print("Sensor data cleaning completed!")
        
        return data
    
    def get_sensor_statistics(self) -> pd.DataFrame:
        """
        Get statistics for sensor columns
        
        Returns:
            DataFrame with sensor statistics
        """
        if self.clean_data is None:
            raise ValueError("No cleaned data available. Run clean_sensor_data() first.")
        
        if not self.sensor_columns:
            self.identify_sensor_columns()
        
        stats = self.clean_data[self.sensor_columns].describe()
        return stats
    
    def inverse_transform(self, normalized_data: np.ndarray, 
                         column_names: List[str]) -> np.ndarray:
        """
        Inverse transform normalized data back to original scale
        
        Args:
            normalized_data: Normalized data array
            column_names: Names of the columns
            
        Returns:
            Data in original scale
        """
        if not self.scalers:
            raise ValueError("No scalers available. Run normalize_data() first.")
        
        original_data = normalized_data.copy()
        
        for i, col in enumerate(column_names):
            if col in self.scalers:
                values = original_data[:, i].reshape(-1, 1)
                original_data[:, i] = self.scalers[col].inverse_transform(values).flatten()
        
        return original_data


if __name__ == "__main__":
    # Example usage
    cleaner = SensorDataCleaner("../../data/0708YTS4.csv")
    clean_data = cleaner.clean_sensor_data()
    
    print(f"\nCleaned data shape: {clean_data.shape}")
    print(f"Sensor columns: {len(cleaner.sensor_columns)}")
    print(f"\nFirst few rows:")
    print(clean_data.head())