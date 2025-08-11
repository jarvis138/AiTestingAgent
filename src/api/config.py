import os
from typing import Optional


class Settings:
    def __init__(self) -> None:
        self.api_key: Optional[str] = os.getenv('API_KEY')
        self.database_url: str = os.getenv('DATABASE_URL', 'sqlite:///data/agent.db')
        self.playwright_dir: Optional[str] = os.getenv('PLAYWRIGHT_DIR')


settings = Settings()
