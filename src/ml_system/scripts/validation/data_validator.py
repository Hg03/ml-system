from ml_system.scripts.utils.loading_utils import load_
from ml_system.scripts.utils.saving_utils import save_
from omegaconf import DictConfig
import pandera.polars as pa
from pandera.api.polars.components import Column
import polars as pl
from pandera import Check

def raw_validation(configs: DictConfig):
    raw_data_path = configs.data.paths.raw_data
    raw_data = load_(path=configs.data.paths.raw_data, format='parquet')

    feature_schema = pa.DataFrameSchema({
        # Identifiers
        "id": Column(str, nullable=False, unique=True),
        "created_at": Column(pl.Datetime, nullable=False),
        
        # Demographics
        "person_name": Column(str, nullable=False),
        "age": Column(int, checks=[
            Check.greater_than_or_equal_to(10),
            Check.less_than_or_equal_to(100)
        ]),
        "gender": Column(str, checks=Check.isin(["Male", "Female", "Other"])),
        
        # Time tracking
        "date": Column(pl.Datetime, nullable=False),
        "daily_screen_time_min": Column(float, checks=[
            Check.greater_than_or_equal_to(0),
            Check.less_than_or_equal_to(1440)  # max minutes in a day
        ]),
        "social_media_time_min": Column(float, checks=[
            Check.greater_than_or_equal_to(0),
            Check.less_than_or_equal_to(1440)
        ]),
        
        # Platform
        "platform": Column(str, checks=Check.isin([
            "Instagram", "WhatsApp", "Facebook", "Twitter", "TikTok", "YouTube", "Snapchat"
        ])),
        
        # Interactions
        "negative_interactions_count": Column(int, checks=Check.greater_than_or_equal_to(0)),
        "positive_interactions_count": Column(int, checks=Check.greater_than_or_equal_to(0)),
        
        # Health metrics
        "sleep_hours": Column(float, checks=[
            Check.greater_than_or_equal_to(0),
            Check.less_than_or_equal_to(24)
        ]),
        "physical_activity_min": Column(float, checks=[
            Check.greater_than_or_equal_to(0),
            Check.less_than_or_equal_to(1440)
        ]),
        
        # Mental health
        "anxiety_level": Column(float, checks=[
            Check.greater_than_or_equal_to(1),
            Check.less_than_or_equal_to(5)  # assuming 0-10 scale
        ]),
        "mental_state": Column(str, checks=Check.isin([
            "At_Risk", "Stressed", "Healthy"
        ])),
    }, 
    coerce=True,  # Auto-convert types where possible
    strict=True   # No extra columns allowed
    )
    validated_data = feature_schema.validate(raw_data)
    save_(to_store=validated_data, path=raw_data_path, format='parquet')
    print('Validation Completed')
