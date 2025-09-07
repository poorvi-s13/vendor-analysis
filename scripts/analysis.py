# analysis.py - FIXED VERSION
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
conn = sqlite3.connect(r'C:\Users\Poorvi\Desktop\vendoe-analysis\data\sample_data.db')

# Load data
vendors_df = pd.read_sql_query("SELECT * FROM vendors", conn)
orders_df = pd.read_sql_query("SELECT * FROM purchase_orders", conn)

# Debug: Check what columns we have
print("Columns in vendors_df:", vendors_df.columns.tolist())
print("Columns in orders_df:", orders_df.columns.tolist())

# Data cleaning
orders_df.rename(columns={'order_id': 'po_id'}, inplace=True)

# Merge data
merged_df = orders_df.merge(vendors_df, on='vendor_id')

# Debug: Check merged columns
print("Columns in merged_df:", merged_df.columns.tolist())

# FIXED: Calculate performance metrics - NO CATEGORY COLUMN
vendor_performance = merged_df.groupby(['vendor_id', 'vendor_name']).agg({
    'po_id': 'count',
    'cost': ['sum', 'mean'],
    'delivery_days': 'mean'
}).reset_index()

# Flatten column names
vendor_performance.columns = ['vendor_id', 'vendor_name', 'order_count', 'total_spend', 'avg_order_value', 'avg_delivery_days']

# Calculate delivery rate
vendor_performance['delivery_rate'] = vendor_performance['avg_delivery_days'].apply(
    lambda x: max(60, 100 - (x * 5))
)

# Use existing quality ratings
quality_map = merged_df.groupby('vendor_id')['quality_rating'].mean()
vendor_performance['quality_score'] = vendor_performance['vendor_id'].map(quality_map) * 20

# Use reliability ratings for response score
reliability_map = merged_df.groupby('vendor_id')['reliability_rating'].mean()
vendor_performance['response_score'] = vendor_performance['vendor_id'].map(reliability_map) / 5

# Normalize scores
scaler = MinMaxScaler()
performance_metrics = vendor_performance[['delivery_rate', 'quality_score', 'response_score', 'avg_order_value']]
performance_metrics_norm = scaler.fit_transform(performance_metrics)
vendor_performance[['delivery_rate_norm', 'quality_score_norm', 'response_score_norm', 'value_score_norm']] = performance_metrics_norm

# Calculate overall performance score
vendor_performance['overall_score'] = (
    vendor_performance['delivery_rate']/100 * 0.4 + 
    vendor_performance['quality_score']/100 * 0.3 + 
    vendor_performance['response_score'] * 0.2 +
    vendor_performance['value_score_norm'] * 0.1
)

# Vendor segmentation using K-means
cluster_data = vendor_performance[['delivery_rate', 'quality_score', 'response_score']]
kmeans = KMeans(n_clusters=3, random_state=42)
vendor_performance['segment'] = kmeans.fit_predict(cluster_data)
vendor_performance['segment'] = vendor_performance['segment'].map({0: 'Strategic', 1: 'Standard', 2: 'Underperforming'})

# Add placeholder category
vendor_performance['category'] = 'Vendor'

# Save processed data
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

print("Analysis completed successfully!")
print(r"Processed data saved to C:\Users\Poorvi\Desktop\vendoe-analysis\outputs\vendor_performance_processed.csv")

conn.close()