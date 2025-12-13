from ml_system.scripts.utils.saving_utils import save_
from supabase import create_client
from dotenv import load_dotenv
from tqdm import tqdm
import polars as pl
import os
load_dotenv()

def format_dataframe(df: pl.DataFrame, target_col: str):
    col_map = {col.lower(): col for col in df.columns}
    df = df.rename(col_map)
    df = df.filter(pl.col(target_col).is_not_null())
    return df

def from_supabase(configs: dict) -> pl.DataFrame:
    load_dotenv()
    conn = create_client(supabase_url=os.getenv("supabase_url"), supabase_key=os.getenv("supabase_api_key"))
    json_data = []
    batch_size, offset = configs.data.batch_size, configs.data.offset
    raw_data_path = configs.data.paths.raw_data
    total_rows = conn.table(configs.data.raw_data_table_name).select("count", count="exact").execute().count
    # Create progress bar
    progress_bar = tqdm(total=total_rows,desc="Loading data from Supabase",unit=" rows")
    while True:
        response = conn.table(configs.data.raw_data_table_name).select("*").limit(batch_size).offset(offset).execute()
        batch = response.data
        if not batch:
            break
        json_data.extend(batch)
        offset+=batch_size
        progress_bar.update(len(batch))
    progress_bar.close()
    raw_data = pl.DataFrame(json_data)
    df = format_dataframe(df=raw_data, target_col=configs.data.columns.target)
    save_(to_store=df, path=raw_data_path, format='parquet')
    print('Loading Completed')
