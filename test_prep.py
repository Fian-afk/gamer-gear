import pandas as pd
import numpy as np
import re

df = pd.read_csv('raw-dataset/final_cleaned_dataset.csv')
print("Initial shape:", df.shape)

# Step 1: Standardisasi Satuan
KURS_INR_TO_IDR = 190.0 # 1 INR = Rp 190.00
df['price_idr'] = df['price'] * KURS_INR_TO_IDR
df['ram_gb'] = df['ram(GB)'].astype(int)

def parse_storage(val):
    if pd.isna(val):
        return np.nan
    val_str = str(val).strip()
    match = re.search(r'(\d+(?:\.\d+)?)\s*(GB|TB)', val_str, re.IGNORECASE)
    if match:
        num, unit = float(match.group(1)), match.group(2).upper()
        if unit == 'TB':
            return int(num * 1024)
        elif unit == 'GB':
            return int(num)
    return np.nan

df['storage_gb'] = df['storage_size'].apply(parse_storage)

# Step 2: Ekstraksi Fitur Teks
def extract_gpu_series(text):
    if pd.isna(text):
        return 'Unknown'
    text = str(text)
    rtx_match = re.search(r'(RTX\s*\d{4}(?:\s*(?:Ti|TI|Super|SUPER|A))?)', text, re.IGNORECASE)
    if rtx_match:
        val = re.sub(r'\s+', ' ', rtx_match.group(1).upper())
        return val.replace('TI', 'Ti')
    gtx_match = re.search(r'(GTX\s*\d{4}(?:\s*(?:Ti|TI))?)', text, re.IGNORECASE)
    if gtx_match:
        return re.sub(r'\s+', ' ', gtx_match.group(1).upper()).replace('TI', 'Ti')
    rx_match = re.search(r'(RX\s*\d{4}[A-Z]?)', text, re.IGNORECASE)
    if rx_match:
        return re.sub(r'\s+', ' ', rx_match.group(1).upper())
    if 'Iris Xe' in text:
        return 'Intel Iris Xe'
    if 'Arc' in text:
        return 'Intel Arc'
    if 'Radeon' in text:
        return 'AMD Radeon'
    return 'Other/Integrated'

df['gpu_series'] = df['graphics_card'].apply(extract_gpu_series)

def extract_cpu_features(text):
    if pd.isna(text):
        return pd.Series({'cpu_brand': 'Unknown', 'cpu_class': 'Unknown', 'cpu_gen': np.nan})
    text = str(text)
    
    brand = 'Unknown'
    if re.search(r'Intel|Core', text, re.I):
        brand = 'Intel'
    elif re.search(r'AMD|Ryzen', text, re.I):
        brand = 'AMD'
    elif re.search(r'Snapdragon', text, re.I):
        brand = 'Qualcomm'
        
    cpu_class = 'Other'
    if re.search(r'Ultra\s*9', text, re.I):
        cpu_class = 'Core Ultra 9'
    elif re.search(r'Ultra\s*7', text, re.I):
        cpu_class = 'Core Ultra 7'
    elif re.search(r'i9|Core\s*9', text, re.I):
        cpu_class = 'Core i9'
    elif re.search(r'i7|Core\s*7', text, re.I):
        cpu_class = 'Core i7'
    elif re.search(r'i5|Core\s*5', text, re.I):
        cpu_class = 'Core i5'
    elif re.search(r'i3|Core\s*3', text, re.I):
        cpu_class = 'Core i3'
    elif re.search(r'Ryzen\s*9|Ryzen\s*AI\s*9', text, re.I):
        cpu_class = 'Ryzen 9'
    elif re.search(r'Ryzen\s*7|Ryzen\s*AI\s*7', text, re.I):
        cpu_class = 'Ryzen 7'
    elif re.search(r'Ryzen\s*5', text, re.I):
        cpu_class = 'Ryzen 5'
    elif re.search(r'Ryzen\s*3', text, re.I):
        cpu_class = 'Ryzen 3'
        
    gen_match = re.search(r'(\d+)(?:th|nd|rd|st)\s*Gen', text, re.I)
    if gen_match:
        gen = int(gen_match.group(1))
    else:
        model_intel = re.search(r'(\d{2})\d{3}[A-Z]*', text)
        if model_intel and brand == 'Intel':
            gen = int(model_intel.group(1))
        else:
            model_amd = re.search(r'[Rr]yzen\s*[3579]\s*(\d)\d{3}', text)
            if model_amd and brand == 'AMD':
                gen = int(model_amd.group(1))
            else:
                gen = np.nan

    return pd.Series({'cpu_brand': brand, 'cpu_class': cpu_class, 'cpu_gen': gen})

cpu_feats = df['processor'].apply(extract_cpu_features)
df = pd.concat([df, cpu_feats], axis=1)

df['screen_size_inch'] = df['display_size']
df['refresh_rate_hz'] = df['name'].str.extract(r'(\d{2,3})\s*[Hh][Zz]', expand=False).astype(float)

# Step 3: Pemetaan Tier Performa
gpu_tier_map = {
    'Intel Iris Xe': 1, 'AMD Radeon': 1, 'Other/Integrated': 1,
    'GTX 1050 Ti': 1, 'GTX 1650': 1, 'RX 5500M': 1, 'RX 6500M': 1,
    'RTX 2050': 2, 'RTX 3030': 2, 'RTX 3050': 2, 'RTX 3050 A': 2, 'RTX 3050 Ti': 2, 'GTX 1660Ti': 2,
    'RTX 2060': 3, 'RTX 3060': 3, 'RTX 4050': 3, 'RX 6700S': 3,
    'RTX 4060': 4, 'RTX 5050': 4, 'RTX 3070': 4, 'RTX 3070 Ti': 4,
    'RTX 4070': 5, 'RTX 5060': 5, 'RTX 5070': 5, 'RTX 5070 Ti': 5, 'RTX 3080': 5, 'RTX 3080 Ti': 5,
    'RTX 4080': 6, 'RTX 4090': 6, 'RTX 5080': 6, 'RTX 5090': 6
}
df['gpu_tier'] = df['gpu_series'].map(gpu_tier_map).fillna(2).astype(int)

cpu_tier_map = {
    'Other': 1, 'Core i3': 1, 'Ryzen 3': 1,
    'Core i5': 2, 'Ryzen 5': 2,
    'Core i7': 3, 'Ryzen 7': 3, 'Core Ultra 7': 4,
    'Core i9': 5, 'Ryzen 9': 5, 'Core Ultra 9': 5
}
df['cpu_tier'] = df['cpu_class'].map(cpu_tier_map).fillna(2).astype(int)

# Step 4: Penanganan Missing Values & Anomali/Outlier
# Storage anomaly (8 GB, 1 GB) -> koreksi menjadi 1024 GB (1 TB) karena keduanya adalah laptop ROG Strix Scar
df.loc[df['storage_gb'] <= 8, 'storage_gb'] = 1024

# Missing refresh rate -> imputasi median laptop gaming (144 Hz)
df['refresh_rate_hz'] = df['refresh_rate_hz'].fillna(144.0)

# Step 5: Transformasi Distribusi Harga
skew_before = df['price_idr'].skew()
df['log_price_idr'] = np.log1p(df['price_idr'])
skew_after = df['log_price_idr'].skew()

print("Price skew before:", skew_before)
print("Price skew after:", skew_after)
print("Sample cleaned dataframe:")
print(df[['name', 'price_idr', 'log_price_idr', 'ram_gb', 'storage_gb', 'gpu_series', 'gpu_tier', 'cpu_class', 'cpu_tier', 'refresh_rate_hz']].head(5))
