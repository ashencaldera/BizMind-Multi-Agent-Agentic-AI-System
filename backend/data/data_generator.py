"""
BizMind Data Generator
Generates realistic synthetic business data for a retail/e-commerce company.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

PRODUCTS = [
    {"id": "P001", "name": "Wireless Headphones", "category": "Electronics", "base_price": 129.99, "base_cost": 55.0},
    {"id": "P002", "name": "Running Shoes", "category": "Sports", "base_price": 89.99, "base_cost": 35.0},
    {"id": "P003", "name": "Coffee Maker", "category": "Appliances", "base_price": 79.99, "base_cost": 30.0},
    {"id": "P004", "name": "Yoga Mat", "category": "Sports", "base_price": 34.99, "base_cost": 12.0},
    {"id": "P005", "name": "Desk Lamp", "category": "Home", "base_price": 44.99, "base_cost": 18.0},
    {"id": "P006", "name": "Protein Powder", "category": "Health", "base_price": 54.99, "base_cost": 22.0},
    {"id": "P007", "name": "Bluetooth Speaker", "category": "Electronics", "base_price": 69.99, "base_cost": 28.0},
    {"id": "P008", "name": "Water Bottle", "category": "Sports", "base_price": 24.99, "base_cost": 8.0},
]

SEGMENTS = ["Premium", "Regular", "Budget", "New"]
CHANNELS = ["Organic Search", "Paid Ads", "Social Media", "Email", "Referral", "Direct"]
REGIONS = ["North", "South", "East", "West", "Central"]

def generate_sales_data(days: int = 180) -> pd.DataFrame:
    records = []
    base_date = datetime.now() - timedelta(days=days)

    for d in range(days):
        current_date = base_date + timedelta(days=d)
        day_of_week = current_date.weekday()
        month = current_date.month

        # Seasonal multiplier: Q4 holiday boost
        seasonal = 1.0
        if month in [11, 12]:
            seasonal = 1.6
        elif month in [6, 7]:
            seasonal = 1.2
        elif month in [1, 2]:
            seasonal = 0.8

        # Weekend boost
        weekend_boost = 1.3 if day_of_week >= 5 else 1.0

        # Trend: gradual 0.3% daily growth
        trend = 1.0 + (d * 0.003)

        for product in PRODUCTS:
            for region in REGIONS:
                base_units = random.randint(5, 30)
                noise = np.random.normal(1.0, 0.15)
                units = max(1, int(base_units * seasonal * weekend_boost * trend * noise))

                # Price variation ±10%
                price = product["base_price"] * np.random.uniform(0.92, 1.08)
                revenue = units * price
                cost = units * product["base_cost"]
                profit = revenue - cost

                records.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "product_id": product["id"],
                    "product_name": product["name"],
                    "category": product["category"],
                    "region": region,
                    "units_sold": units,
                    "price": round(price, 2),
                    "revenue": round(revenue, 2),
                    "cost": round(cost, 2),
                    "profit": round(profit, 2),
                    "channel": random.choice(CHANNELS),
                })
    return pd.DataFrame(records)


def generate_customer_data(n: int = 500) -> pd.DataFrame:
    customers = []
    base_date = datetime.now()

    for i in range(n):
        segment = random.choices(SEGMENTS, weights=[15, 50, 25, 10])[0]
        join_days_ago = random.randint(1, 720)
        join_date = base_date - timedelta(days=join_days_ago)

        if segment == "Premium":
            avg_order = np.random.normal(250, 50)
            frequency = np.random.normal(8, 2)
            churn_prob = np.random.beta(1, 9)
        elif segment == "Regular":
            avg_order = np.random.normal(90, 20)
            frequency = np.random.normal(4, 1.5)
            churn_prob = np.random.beta(2, 6)
        elif segment == "Budget":
            avg_order = np.random.normal(35, 10)
            frequency = np.random.normal(2, 1)
            churn_prob = np.random.beta(3, 5)
        else:  # New
            avg_order = np.random.normal(65, 25)
            frequency = np.random.normal(1.5, 0.5)
            churn_prob = np.random.beta(4, 4)

        last_purchase_days = random.randint(1, 120)
        churned = churn_prob > 0.6 or last_purchase_days > 90

        customers.append({
            "customer_id": f"C{i+1:04d}",
            "segment": segment,
            "join_date": join_date.strftime("%Y-%m-%d"),
            "last_purchase_days_ago": last_purchase_days,
            "avg_order_value": round(max(10, avg_order), 2),
            "purchase_frequency": round(max(0.5, frequency), 1),
            "total_orders": max(1, int(frequency * join_days_ago / 90)),
            "lifetime_value": round(max(10, avg_order) * max(1, int(frequency * join_days_ago / 90)), 2),
            "churn_risk": round(churn_prob, 3),
            "is_churned": churned,
            "preferred_channel": random.choice(CHANNELS),
            "preferred_category": random.choice([p["category"] for p in PRODUCTS]),
            "region": random.choice(REGIONS),
            "nps_score": random.randint(1, 10),
        })
    return pd.DataFrame(customers)


def generate_inventory_data() -> pd.DataFrame:
    records = []
    for product in PRODUCTS:
        current_stock = random.randint(20, 500)
        reorder_point = random.randint(30, 100)
        max_capacity = random.randint(400, 1000)
        daily_sales_avg = random.randint(5, 40)
        lead_time_days = random.randint(3, 14)
        days_of_stock = current_stock / max(1, daily_sales_avg)

        records.append({
            "product_id": product["id"],
            "product_name": product["name"],
            "category": product["category"],
            "current_stock": current_stock,
            "reorder_point": reorder_point,
            "max_capacity": max_capacity,
            "daily_sales_avg": daily_sales_avg,
            "lead_time_days": lead_time_days,
            "days_of_stock_remaining": round(days_of_stock, 1),
            "stock_value": round(current_stock * product["base_cost"], 2),
            "turnover_rate": round(daily_sales_avg * 30 / max(1, current_stock), 2),
            "status": "Critical" if current_stock <= reorder_point else
                      "Low" if current_stock <= reorder_point * 1.5 else
                      "Overstock" if current_stock >= max_capacity * 0.9 else "Optimal",
        })
    return pd.DataFrame(records)


def generate_marketing_data(days: int = 90) -> pd.DataFrame:
    records = []
    base_date = datetime.now() - timedelta(days=days)

    campaigns = [
        {"name": "Summer Sale", "channel": "Paid Ads", "budget": 5000},
        {"name": "Email Newsletter", "channel": "Email", "budget": 800},
        {"name": "Social Influencer", "channel": "Social Media", "budget": 3000},
        {"name": "Retargeting Q2", "channel": "Paid Ads", "budget": 2500},
        {"name": "Loyalty Program", "channel": "Email", "budget": 1200},
        {"name": "Brand Awareness", "channel": "Social Media", "budget": 4000},
    ]

    for d in range(days):
        current_date = base_date + timedelta(days=d)
        for campaign in campaigns:
            daily_budget = campaign["budget"] / days
            impressions = int(daily_budget * random.uniform(50, 150))
            ctr = np.random.beta(2, 8) * 0.12  # 0-12% CTR
            clicks = int(impressions * ctr)
            conversion_rate = np.random.beta(1, 20) * 0.08
            conversions = int(clicks * conversion_rate)
            revenue = conversions * random.uniform(60, 200)
            spend = daily_budget * random.uniform(0.85, 1.1)

            records.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "campaign": campaign["name"],
                "channel": campaign["channel"],
                "impressions": impressions,
                "clicks": clicks,
                "conversions": conversions,
                "spend": round(spend, 2),
                "revenue": round(revenue, 2),
                "roi": round((revenue - spend) / max(1, spend) * 100, 1),
                "ctr": round(ctr * 100, 2),
                "conversion_rate": round(conversion_rate * 100, 2),
                "cpa": round(spend / max(1, conversions), 2),
            })
    return pd.DataFrame(records)


# Pre-generate datasets (cached)
_cache = {}

def get_sales_data() -> pd.DataFrame:
    if "sales" not in _cache:
        _cache["sales"] = generate_sales_data(180)
    return _cache["sales"]

def get_customer_data() -> pd.DataFrame:
    if "customers" not in _cache:
        _cache["customers"] = generate_customer_data(500)
    return _cache["customers"]

def get_inventory_data() -> pd.DataFrame:
    if "inventory" not in _cache:
        _cache["inventory"] = generate_inventory_data()
    return _cache["inventory"]

def get_marketing_data() -> pd.DataFrame:
    if "marketing" not in _cache:
        _cache["marketing"] = generate_marketing_data(90)
    return _cache["marketing"]
