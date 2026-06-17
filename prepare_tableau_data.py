"""
Datathon 2026 — Data Preparation for Tableau Public
====================================================
Script tạo toàn bộ bảng dữ liệu enriched/aggregated để load vào Tableau.
Output: thư mục tableau_data/ chứa 13 file CSV.
"""

import pandas as pd
import numpy as np
import os
import sys
import io
import warnings
warnings.filterwarnings('ignore')

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# ─── Config ───────────────────────────────────────────────────────────
DATA_DIR = r'd:\AI\Soan bai\Project competiion\Data'
OUT_DIR  = r'd:\AI\Soan bai\Project competiion\tableau_data'
os.makedirs(OUT_DIR, exist_ok=True)

def load(name, **kw):
    """Load CSV with progress print."""
    path = os.path.join(DATA_DIR, name)
    df = pd.read_csv(path, **kw)
    print(f"  ✓ Loaded {name}: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df

def save(df, name):
    """Save to tableau_data/ with summary."""
    path = os.path.join(OUT_DIR, name)
    df.to_csv(path, index=False)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  → Saved {name}: {df.shape[0]:,} rows × {df.shape[1]} cols ({size_mb:.1f} MB)")

# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Load raw data
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 1: Loading raw data")
print("="*60)

orders      = load('orders.csv',      parse_dates=['order_date'])
order_items = load('order_items.csv')
payments    = load('payments.csv')
products    = load('products.csv')
customers   = load('customers.csv',   parse_dates=['signup_date'])
geography   = load('geography.csv')
promotions  = load('promotions.csv',  parse_dates=['start_date', 'end_date'])
shipments   = load('shipments.csv',   parse_dates=['ship_date', 'delivery_date'])
returns     = load('returns.csv',     parse_dates=['return_date'])
reviews     = load('reviews.csv',     parse_dates=['review_date'])
inventory   = load('inventory.csv',   parse_dates=['snapshot_date'])
web_traffic = load('web_traffic.csv', parse_dates=['date'])
sales       = load('sales.csv',      parse_dates=['Date'])

# ═══════════════════════════════════════════════════════════════════════
# STEP 2: dim_products.csv — Products with computed margin
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 2: dim_products.csv")
print("="*60)

dim_products = products.copy()
dim_products['gross_margin_pct'] = ((dim_products['price'] - dim_products['cogs']) / dim_products['price'] * 100).round(2)
dim_products['profit_per_unit']  = (dim_products['price'] - dim_products['cogs']).round(2)

# Avg rating per product
avg_rating = reviews.groupby('product_id')['rating'].agg(['mean', 'count']).reset_index()
avg_rating.columns = ['product_id', 'avg_rating', 'review_count']
avg_rating['avg_rating'] = avg_rating['avg_rating'].round(2)

# Return stats per product
ret_stats = returns.groupby('product_id').agg(
    total_returns=('return_id', 'count'),
    total_return_qty=('return_quantity', 'sum'),
    total_refund=('refund_amount', 'sum')
).reset_index()

dim_products = dim_products.merge(avg_rating, on='product_id', how='left')
dim_products = dim_products.merge(ret_stats, on='product_id', how='left')
dim_products[['review_count', 'total_returns', 'total_return_qty']] = \
    dim_products[['review_count', 'total_returns', 'total_return_qty']].fillna(0).astype(int)
dim_products['total_refund'] = dim_products['total_refund'].fillna(0)

save(dim_products, 'dim_products.csv')

# ═══════════════════════════════════════════════════════════════════════
# STEP 3: dim_geography.csv — Geography (as-is)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 3: dim_geography.csv")
print("="*60)

save(geography, 'dim_geography.csv')

# ═══════════════════════════════════════════════════════════════════════
# STEP 4: dim_promotions.csv — Promotions (as-is)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 4: dim_promotions.csv")
print("="*60)

save(promotions, 'dim_promotions.csv')

# ═══════════════════════════════════════════════════════════════════════
# STEP 5: fact_orders_enriched.csv — Denormalized fact table
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 5: fact_orders_enriched.csv (main fact table)")
print("="*60)

# Start with order_items as grain (each line item)
fact = order_items.copy()

# ── Join orders ──
fact = fact.merge(
    orders[['order_id', 'order_date', 'customer_id', 'zip', 
            'order_status', 'payment_method', 'device_type', 'order_source']],
    on='order_id', how='left'
)

# ── Join products (key fields only, to avoid bloat) ──
fact = fact.merge(
    products[['product_id', 'product_name', 'category', 'segment', 'size', 'color', 'price', 'cogs']],
    on='product_id', how='left'
)

# ── Join payments ──
fact = fact.merge(
    payments[['order_id', 'payment_value', 'installments']],
    on='order_id', how='left'
)

# ── Join geography (region, city) ──
fact = fact.merge(
    geography[['zip', 'city', 'region']].drop_duplicates(subset='zip'),
    on='zip', how='left'
)

# ── Computed fields ──
fact['line_revenue']       = (fact['quantity'] * fact['unit_price']).round(2)
fact['line_cost']          = (fact['quantity'] * fact['cogs']).round(2)
fact['line_gross_profit']  = (fact['line_revenue'] - fact['line_cost']).round(2)
fact['line_margin_pct']    = np.where(
    fact['line_revenue'] > 0,
    (fact['line_gross_profit'] / fact['line_revenue'] * 100).round(2),
    0
)
fact['has_promo']          = (~fact['promo_id'].isna() & (fact['promo_id'] != '')).astype(int)
fact['has_double_promo']   = (~fact['promo_id_2'].isna() & (fact['promo_id_2'] != '')).astype(int)
fact['discount_pct']       = np.where(
    fact['line_revenue'] + fact['discount_amount'] > 0,
    (fact['discount_amount'] / (fact['line_revenue'] + fact['discount_amount']) * 100).round(2),
    0
)

# ── Date dimensions ──
fact['order_year']    = fact['order_date'].dt.year
fact['order_month']   = fact['order_date'].dt.month
fact['order_quarter'] = fact['order_date'].dt.quarter
fact['order_dow']     = fact['order_date'].dt.dayofweek  # 0=Mon
fact['order_dow_name']= fact['order_date'].dt.day_name()
fact['order_ym']      = fact['order_date'].dt.to_period('M').astype(str)

save(fact, 'fact_orders_enriched.csv')

# ═══════════════════════════════════════════════════════════════════════
# STEP 6: fact_returns_enriched.csv
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 6: fact_returns_enriched.csv")
print("="*60)

fact_ret = returns.copy()
fact_ret = fact_ret.merge(
    products[['product_id', 'product_name', 'category', 'segment', 'size', 'color', 'price', 'cogs']],
    on='product_id', how='left'
)
fact_ret = fact_ret.merge(
    orders[['order_id', 'order_date', 'customer_id', 'order_source', 'device_type']],
    on='order_id', how='left'
)
# Time to return
fact_ret['days_to_return'] = (fact_ret['return_date'] - fact_ret['order_date']).dt.days
fact_ret['return_year']    = fact_ret['return_date'].dt.year
fact_ret['return_month']   = fact_ret['return_date'].dt.month
fact_ret['return_ym']      = fact_ret['return_date'].dt.to_period('M').astype(str)

save(fact_ret, 'fact_returns_enriched.csv')

# ═══════════════════════════════════════════════════════════════════════
# STEP 7: dim_customers_rfm.csv — RFM Segmentation
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 7: dim_customers_rfm.csv (RFM Segmentation)")
print("="*60)

# Reference date = last order date in dataset + 1 day
ref_date = orders['order_date'].max() + pd.Timedelta(days=1)
print(f"  Reference date for Recency: {ref_date.date()}")

# Compute RFM per customer
rfm = orders.merge(payments[['order_id', 'payment_value']], on='order_id', how='left')
rfm = rfm[rfm['order_status'] != 'cancelled']  # Exclude cancelled

rfm_agg = rfm.groupby('customer_id').agg(
    last_order_date   = ('order_date', 'max'),
    first_order_date  = ('order_date', 'min'),
    frequency         = ('order_id', 'nunique'),
    monetary          = ('payment_value', 'sum'),
    avg_order_value   = ('payment_value', 'mean'),
).reset_index()

rfm_agg['recency_days'] = (ref_date - rfm_agg['last_order_date']).dt.days
rfm_agg['tenure_days']  = (ref_date - rfm_agg['first_order_date']).dt.days
rfm_agg['monetary']     = rfm_agg['monetary'].round(2)
rfm_agg['avg_order_value'] = rfm_agg['avg_order_value'].round(2)

# ── RFM Scores (quintiles 1-5) ──
rfm_agg['R_score'] = pd.qcut(rfm_agg['recency_days'], q=5, labels=[5,4,3,2,1]).astype(int)
rfm_agg['F_score'] = pd.qcut(rfm_agg['frequency'].rank(method='first'), q=5, labels=[1,2,3,4,5]).astype(int)
rfm_agg['M_score'] = pd.qcut(rfm_agg['monetary'].rank(method='first'), q=5, labels=[1,2,3,4,5]).astype(int)
rfm_agg['RFM_score'] = rfm_agg['R_score'] * 100 + rfm_agg['F_score'] * 10 + rfm_agg['M_score']

# ── RFM Segment Labels ──
def rfm_segment(row):
    r, f, m = row['R_score'], row['F_score'], row['M_score']
    if r >= 4 and f >= 4:
        return 'Champions'
    elif r >= 3 and f >= 3:
        return 'Loyal Customers'
    elif r >= 4 and f <= 2:
        return 'New Customers'
    elif r >= 3 and f >= 1 and m >= 3:
        return 'Potential Loyalists'
    elif r <= 2 and f >= 3:
        return 'At Risk'
    elif r <= 2 and f >= 4 and m >= 4:
        return 'Cant Lose Them'
    elif r <= 2 and f <= 2:
        return 'Lost'
    elif r == 3 and f <= 2:
        return 'About To Sleep'
    else:
        return 'Need Attention'

rfm_agg['rfm_segment'] = rfm_agg.apply(rfm_segment, axis=1)

# ── Merge with customer demographics ──
dim_cust = customers.merge(rfm_agg, on='customer_id', how='left')
dim_cust = dim_cust.merge(
    geography[['zip', 'region']].drop_duplicates(subset='zip'),
    on='zip', how='left'
)

# Signup year/month
dim_cust['signup_year']  = dim_cust['signup_date'].dt.year
dim_cust['signup_month'] = dim_cust['signup_date'].dt.month
dim_cust['signup_ym']    = dim_cust['signup_date'].dt.to_period('M').astype(str)

# Fill NAs for customers without orders
dim_cust['frequency']    = dim_cust['frequency'].fillna(0).astype(int)
dim_cust['monetary']     = dim_cust['monetary'].fillna(0)
dim_cust['rfm_segment']  = dim_cust['rfm_segment'].fillna('Never Purchased')

# Segment distribution
print("\n  RFM Segment distribution:")
print(dim_cust['rfm_segment'].value_counts().to_string())

save(dim_cust, 'dim_customers_rfm.csv')

# ═══════════════════════════════════════════════════════════════════════
# STEP 8: fact_shipments_enriched.csv
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 8: fact_shipments_enriched.csv")
print("="*60)

fact_ship = shipments.copy()
fact_ship['delivery_days']     = (fact_ship['delivery_date'] - fact_ship['ship_date']).dt.days
fact_ship['processing_days']   = None  # Will compute after join

# Join order info for processing time
fact_ship = fact_ship.merge(
    orders[['order_id', 'order_date', 'zip', 'order_status']],
    on='order_id', how='left'
)
fact_ship['processing_days'] = (fact_ship['ship_date'] - fact_ship['order_date']).dt.days
fact_ship['total_lead_time'] = (fact_ship['delivery_date'] - fact_ship['order_date']).dt.days

# Join geography for region
fact_ship = fact_ship.merge(
    geography[['zip', 'region', 'city']].drop_duplicates(subset='zip'),
    on='zip', how='left'
)

# Date dims
fact_ship['ship_year']  = fact_ship['ship_date'].dt.year
fact_ship['ship_month'] = fact_ship['ship_date'].dt.month
fact_ship['ship_ym']    = fact_ship['ship_date'].dt.to_period('M').astype(str)

# Delivery performance buckets
fact_ship['delivery_bucket'] = pd.cut(
    fact_ship['delivery_days'],
    bins=[-1, 2, 5, 7, 14, 999],
    labels=['1-2 days', '3-5 days', '6-7 days', '8-14 days', '15+ days']
)

save(fact_ship, 'fact_shipments_enriched.csv')

# ═══════════════════════════════════════════════════════════════════════
# STEP 9: fact_sales_daily.csv — Sales with computed fields
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 9: fact_sales_daily.csv")
print("="*60)

fact_sales = sales.copy()
fact_sales.columns = ['date', 'revenue', 'cogs']
fact_sales['gross_profit']    = (fact_sales['revenue'] - fact_sales['cogs']).round(2)
fact_sales['gross_margin_pct']= (fact_sales['gross_profit'] / fact_sales['revenue'] * 100).round(2)

# Date dimensions
fact_sales['year']         = fact_sales['date'].dt.year
fact_sales['month']        = fact_sales['date'].dt.month
fact_sales['quarter']      = fact_sales['date'].dt.quarter
fact_sales['day_of_week']  = fact_sales['date'].dt.dayofweek
fact_sales['dow_name']     = fact_sales['date'].dt.day_name()
fact_sales['is_weekend']   = (fact_sales['day_of_week'] >= 5).astype(int)
fact_sales['year_month']   = fact_sales['date'].dt.to_period('M').astype(str)

# Rolling averages
fact_sales = fact_sales.sort_values('date')
fact_sales['revenue_ma7']  = fact_sales['revenue'].rolling(7, min_periods=1).mean().round(2)
fact_sales['revenue_ma30'] = fact_sales['revenue'].rolling(30, min_periods=1).mean().round(2)
fact_sales['cogs_ma7']     = fact_sales['cogs'].rolling(7, min_periods=1).mean().round(2)
fact_sales['cogs_ma30']    = fact_sales['cogs'].rolling(30, min_periods=1).mean().round(2)

# YoY growth (compare to same day last year)
fact_sales['revenue_ly'] = fact_sales['revenue'].shift(365)
fact_sales['yoy_growth_pct'] = np.where(
    fact_sales['revenue_ly'] > 0,
    ((fact_sales['revenue'] - fact_sales['revenue_ly']) / fact_sales['revenue_ly'] * 100).round(2),
    np.nan
)
fact_sales.drop(columns=['revenue_ly'], inplace=True)

save(fact_sales, 'fact_sales_daily.csv')

# ═══════════════════════════════════════════════════════════════════════
# STEP 10: fact_inventory.csv — Inventory (clean)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 10: fact_inventory.csv")
print("="*60)

fact_inv = inventory.copy()
# Add inventory value
fact_inv = fact_inv.merge(
    products[['product_id', 'price', 'cogs']].rename(
        columns={'price': 'product_price', 'cogs': 'product_cogs'}
    ),
    on='product_id', how='left'
)
fact_inv['inventory_value_retail'] = (fact_inv['stock_on_hand'] * fact_inv['product_price']).round(2)
fact_inv['inventory_value_cost']   = (fact_inv['stock_on_hand'] * fact_inv['product_cogs']).round(2)

# Inventory health label
def inv_health(row):
    if row['stockout_flag'] == 1:
        return 'Stockout'
    elif row['overstock_flag'] == 1:
        return 'Overstock'
    elif row['reorder_flag'] == 1:
        return 'Reorder Needed'
    else:
        return 'Healthy'

fact_inv['inventory_health'] = fact_inv.apply(inv_health, axis=1)

save(fact_inv, 'fact_inventory.csv')

# ═══════════════════════════════════════════════════════════════════════
# STEP 11: fact_web_traffic.csv — Web traffic (clean)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 11: fact_web_traffic.csv")
print("="*60)

fact_web = web_traffic.copy()
fact_web.rename(columns={'date': 'traffic_date'}, inplace=True)
fact_web['year']  = fact_web['traffic_date'].dt.year
fact_web['month'] = fact_web['traffic_date'].dt.month
fact_web['year_month'] = fact_web['traffic_date'].dt.to_period('M').astype(str)

# Pages per session
fact_web['pages_per_session'] = (fact_web['page_views'] / fact_web['sessions']).round(2)

save(fact_web, 'fact_web_traffic.csv')

# ═══════════════════════════════════════════════════════════════════════
# STEP 12: agg_cohort_retention.csv — Cohort Retention Matrix
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 12: agg_cohort_retention.csv")
print("="*60)

# First purchase month per customer (cohort)
first_order = orders[orders['order_status'] != 'cancelled'].groupby('customer_id')['order_date'].min().reset_index()
first_order.columns = ['customer_id', 'cohort_date']
first_order['cohort_month'] = first_order['cohort_date'].dt.to_period('M')

# All orders with cohort
orders_cohort = orders[orders['order_status'] != 'cancelled'][['order_id', 'customer_id', 'order_date']].copy()
orders_cohort['order_month'] = orders_cohort['order_date'].dt.to_period('M')
orders_cohort = orders_cohort.merge(first_order[['customer_id', 'cohort_month']], on='customer_id')

# Period number (months since cohort)
orders_cohort['period_number'] = (
    (orders_cohort['order_month'].dt.year - orders_cohort['cohort_month'].dt.year) * 12 +
    (orders_cohort['order_month'].dt.month - orders_cohort['cohort_month'].dt.month)
)

# Cohort size
cohort_sizes = orders_cohort.groupby('cohort_month')['customer_id'].nunique().reset_index()
cohort_sizes.columns = ['cohort_month', 'cohort_size']

# Retention counts
retention = orders_cohort.groupby(['cohort_month', 'period_number'])['customer_id'].nunique().reset_index()
retention.columns = ['cohort_month', 'period_number', 'active_customers']

# Merge sizes
retention = retention.merge(cohort_sizes, on='cohort_month')
retention['retention_rate'] = (retention['active_customers'] / retention['cohort_size'] * 100).round(2)

# Convert period to string for Tableau
retention['cohort_month'] = retention['cohort_month'].astype(str)

# Limit to first 24 months for readability
retention = retention[retention['period_number'] <= 24]

save(retention, 'agg_cohort_retention.csv')

# ═══════════════════════════════════════════════════════════════════════
# STEP 13: agg_monthly_summary.csv — Monthly KPI rollup
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 13: agg_monthly_summary.csv")
print("="*60)

# Monthly sales
monthly_sales = fact_sales.groupby('year_month').agg(
    revenue       = ('revenue', 'sum'),
    cogs          = ('cogs', 'sum'),
    gross_profit  = ('gross_profit', 'sum'),
    avg_daily_rev = ('revenue', 'mean'),
    days_in_month = ('revenue', 'count'),
).reset_index()
monthly_sales['gross_margin_pct'] = (monthly_sales['gross_profit'] / monthly_sales['revenue'] * 100).round(2)

# Monthly orders
orders_monthly = orders.copy()
orders_monthly['year_month'] = orders_monthly['order_date'].dt.to_period('M').astype(str)
mo = orders_monthly.groupby('year_month').agg(
    total_orders     = ('order_id', 'nunique'),
    unique_customers = ('customer_id', 'nunique'),
    cancelled_orders = ('order_status', lambda x: (x == 'cancelled').sum()),
).reset_index()
mo['cancel_rate_pct'] = (mo['cancelled_orders'] / mo['total_orders'] * 100).round(2)

# Monthly returns
returns_monthly = returns.copy()
returns_monthly['year_month'] = returns_monthly['return_date'].dt.to_period('M').astype(str)
mr = returns_monthly.groupby('year_month').agg(
    total_returns = ('return_id', 'count'),
    total_refund  = ('refund_amount', 'sum'),
).reset_index()

# Merge all
monthly = monthly_sales.merge(mo, on='year_month', how='left')
monthly = monthly.merge(mr, on='year_month', how='left')
monthly['total_returns'] = monthly['total_returns'].fillna(0).astype(int)
monthly['total_refund']  = monthly['total_refund'].fillna(0)
monthly['return_rate_pct'] = np.where(
    monthly['total_orders'] > 0,
    (monthly['total_returns'] / monthly['total_orders'] * 100).round(2),
    0
)
monthly['aov'] = (monthly['revenue'] / monthly['total_orders']).round(2)

# Revenue growth MoM
monthly = monthly.sort_values('year_month')
monthly['revenue_prev'] = monthly['revenue'].shift(1)
monthly['mom_growth_pct'] = np.where(
    monthly['revenue_prev'] > 0,
    ((monthly['revenue'] - monthly['revenue_prev']) / monthly['revenue_prev'] * 100).round(2),
    np.nan
)
monthly.drop(columns=['revenue_prev'], inplace=True)

save(monthly, 'agg_monthly_summary.csv')

# ═══════════════════════════════════════════════════════════════════════
# STEP 14: agg_reviews_summary.csv — Reviews aggregated
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("STEP 14: agg_reviews_summary.csv")
print("="*60)

reviews_enriched = reviews.merge(
    products[['product_id', 'category', 'segment']],
    on='product_id', how='left'
)
reviews_enriched['review_ym'] = reviews_enriched['review_date'].dt.to_period('M').astype(str)

# Monthly avg rating by category
rev_summary = reviews_enriched.groupby(['review_ym', 'category']).agg(
    avg_rating    = ('rating', 'mean'),
    review_count  = ('review_id', 'count'),
    pct_5_star    = ('rating', lambda x: (x == 5).mean() * 100),
    pct_1_star    = ('rating', lambda x: (x == 1).mean() * 100),
).reset_index()
rev_summary['avg_rating'] = rev_summary['avg_rating'].round(2)
rev_summary['pct_5_star'] = rev_summary['pct_5_star'].round(2)
rev_summary['pct_1_star'] = rev_summary['pct_1_star'].round(2)

save(rev_summary, 'agg_reviews_summary.csv')

# ═══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("✅ ALL FILES CREATED SUCCESSFULLY")
print("="*60)

import glob
total_size = 0
files = sorted(glob.glob(os.path.join(OUT_DIR, '*.csv')))
print(f"\n{'File':<40} {'Rows':>10} {'Size (MB)':>10}")
print("-" * 62)
for f in files:
    df = pd.read_csv(f, nrows=0)
    nrows = sum(1 for _ in open(f, encoding='utf-8')) - 1
    size = os.path.getsize(f) / 1024 / 1024
    total_size += size
    print(f"{os.path.basename(f):<40} {nrows:>10,} {size:>10.1f}")

print("-" * 62)
print(f"{'TOTAL':<40} {'':>10} {total_size:>10.1f}")
print(f"\n📁 Output directory: {OUT_DIR}")
print(f"📊 Ready for Tableau Public (total {total_size:.1f} MB)")

# Tableau Public limit check
if total_size > 1000:
    print("⚠️  WARNING: Total file size exceeds 1 GB. Consider reducing data.")
else:
    print("✓  Data size is within Tableau Public limits.")
