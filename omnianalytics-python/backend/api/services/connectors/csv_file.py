import pandas as pd
from typing import Dict, Any, List, Tuple
import time
from .base import BaseConnector
import os

class CSVFileConnector(BaseConnector):
	"""CSV/Excel/JSON file connector."""
	def __init__(self, config: Dict[str, Any]):
		super().__init__(config)
		self.file_path = config.get('file_path')
		self.file_type = config.get('file_type', 'csv')

	async def connect(self) -> None:
		if not self.file_path or not os.path.exists(self.file_path):
			raise FileNotFoundError(f"File not found: {self.file_path}")

	async def disconnect(self) -> None:
		pass

	async def test_connection(self) -> Tuple[bool, float, str]:
		start_time = time.time()
		try:
			await self.connect()
			latency = (time.time() - start_time) * 1000
			return True, latency, "File accessible"
		except Exception as e:
			return False, (time.time() - start_time) * 1000, f"File access failed: {str(e)}"

	async def execute_query(self, query: str = None, **kwargs) -> pd.DataFrame:
		await self.connect()
		if self.file_type == 'csv':
			df = pd.read_csv(self.file_path)
		elif self.file_type in ['xlsx', 'xls']:
			df = pd.read_excel(self.file_path)
		elif self.file_type == 'json':
			df = pd.read_json(self.file_path)
		elif self.file_type == 'parquet':
			df = pd.read_parquet(self.file_path)
		else:
			raise ValueError(f"Unsupported file type: {self.file_type}")
		return df

	async def get_schema(self) -> List[Dict[str, Any]]:
		df = await self.execute_query()
		columns = [{'name': col, 'type': str(df[col].dtype)} for col in df.columns]
		return [{'columns': columns}]

	async def get_tables(self) -> List[str]:
		return [os.path.basename(self.file_path)]
