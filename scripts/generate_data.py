# analysis.py
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Connect to database
conn = sqlite3.connect('C:\Users\Poorvi\Desktop\vendoe-analysis\data\sample_data.db')

# Load data
vendors_df = pd.read_sql_query("SELECT * FROM vendors", conn)
orders_df = pd.read_sql_query("SELECT * FROM purchase_orders", conn)

# Data cleaning
orders_df.rename(columns={'order_id': 'po_id'}, inplace=True)  # Rename to match your original code

# Merge data
merged_df = orders_df.merge(vendors_df, on='vendor_id')

# Calculate performance metrics - REMOVED 'category' from groupby
vendor_performance = merged_df.groupby(['vendor_id', 'vendor_name']).agg({
    'po_id': 'count',
    'cost': ['sum', 'mean'],  # Changed from 'total_amount' to 'cost'
    'delivery_days': 'mean'   # Added delivery days metric
}).reset_index()

# Flatten column names
vendor_performance.columns = ['vendor_id', 'vendor_name', 'order_count', 'total_spend', 'avg_order_value', 'avg_delivery_days']

# Calculate delivery rate (using reliability_rating as proxy)
vendor_performance['delivery_rate'] = vendor_performance['avg_delivery_days'].apply(
    lambda x: max(60, 100 - (x * 5))  # Convert delivery days to percentage
)

# Use existing quality ratings instead of simulating
quality_map = merged_df.groupby('vendor_id')['quality_rating'].mean()
vendor_performance['quality_score'] = vendor_performance['vendor_id'].map(quality_map) * 20  # Convert 1-5 scale to 20-100

# Use existing reliability ratings for response score
reliability_map = merged_df.groupby('vendor_id')['reliability_rating'].mean()
vendor_performance['response_score'] = vendor_performance['vendor_id'].map(reliability_map) / 5  # Convert to 0.2-1.0 scale

# Normalize scores for comparison
scaler = MinMaxScaler()
performance_metrics = vendor_performance[['delivery_rate', 'quality_score', 'response_score', 'avg_order_value']]
performance_metrics.loc[:, ['delivery_rate_norm', 'quality_score_norm', 'response_score_norm', 'value_score_norm']] = scaler.fit_transform(performance_metrics)

# Calculate overall performance score using actual data
vendor_performance['overall_score'] = (
    vendor_performance['delivery_rate']/100 * 0.4 + 
    vendor_performance['quality_score']/100 * 0.3 + 
    vendor_performance['response_score'] * 0.2 +
    performance_metrics['value_score_norm'] * 0.1
)

# Vendor segmentation using K-means
cluster_data = vendor_performance[['delivery_rate', 'quality_score', 'response_score']]
kmeans = KMeans(n_clusters=3, random_state=42)
vendor_performance['segment'] = kmeans.fit_predict(cluster_data)
vendor_performance['segment'] = vendor_performance['segment'].map({0: 'Strategic', 1: 'Standard', 2: 'Underperforming'})

# Add a placeholder category for compatibility with your original code
vendor_performance['category'] = 'Vendor'

# Save processed data for Power BI
vendor_performance.to_csv('../outputs/vendor_performance_processed.csv', index=False)

# Create visualizations
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=vendor_performance, 
    x='delivery_rate', 
    y='quality_score', 
    hue='segment',
    size='total_spend',
    sizes=(20, 200)
)
plt.title('Vendor Performance Segmentation')
plt.xlabel('Delivery Rate (%)')
plt.ylabel('Quality Score')
plt.savefig('../outputs/vendor_segmentation.png', bbox_inches='tight')
plt.show()

# Top vendors by overall score
plt.figure(figsize=(10, 6))
top_vendors = vendor_performance.nlargest(10, 'overall_score')
sns.barplot(data=top_vendors, x='overall_score', y='vendor_name', hue='segment', dodge=False)
plt.title('Top Vendors by Overall Score')
plt.xlabel('Overall Performance Score')
plt.ylabel('Vendor Name')
plt.savefig('../outputs/top_vendors.png', bbox_inches='tight')
plt.show()

# Performance summary by vendor
plt.figure(figsize=(12, 6))
metrics_df = vendor_performance[['vendor_name', 'delivery_rate', 'quality_score', 'response_score']]
metrics_df = metrics_df.set_index('vendor_name')
metrics_df.plot(kind='bar', figsize=(12, 6))
plt.title('Vendor Performance Metrics')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.savefig('../outputs/vendor_metrics.png', bbox_inches='tight')
plt.show()

print("Analysis completed successfully!")
print(f"Processed data saved to C:\Users\Poorvi\Desktop\vendoe-analysis\scripts\generate_data.py")

# Print results
print("\nVendor Performance Summary:")
print(vendor_performance[['vendor_name', 'order_count', 'total_spend', 'overall_score', 'segment']].to_string(index=False))

conn.close()