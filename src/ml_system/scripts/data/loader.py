from ml_system.scripts.utils.saving_utils import save_
from ml_system.scripts.utils.loading_utils import load_
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
    raw_data_path = configs.data.paths.raw_data
    if configs.pipeline.type == 'online':
        conn = create_client(supabase_url=os.getenv("SUPABASE_URL"), supabase_key=os.getenv("SUPABASE_API_KEY"))
        json_data = []
        batch_size, offset = configs.data.batch_size, configs.data.offset
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
    else:
        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(f'{raw_data_path} not exists')

    print(f'Loading Completed: {configs.pipeline.type}')
