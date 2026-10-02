from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    window_size: int = 100
    min_samples_for_drift: int = 30
    psi_buckets: int = 10
    psi_threshold: float = 0.2
    ks_pvalue_threshold: float = 0.05
    model_path: str = "models/model.pkl"
    reference_data_path: str = "data/reference.csv"


settings = Settings()
