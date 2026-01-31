import aiomysql
import pandas as pd
from typing import Dict, Any, List, Tuple
import time
from .base import BaseConnector

class MySQLConnector(BaseConnector):
	"""MySQL database connector."""
	def __init__(self, config: Dict[str, Any]):
		super().__init__(config)
		self.pool = None

	async def connect(self) -> None:
		try:
			self.pool = await aiomysql.create_pool(
				host=self.config.get('host', 'localhost'),
				port=self.config.get('port', 3306),
				db=self.config.get('database'),
				user=self.config.get('username'),
				password=self.config.get('password'),
				minsize=1,
				maxsize=10,
			)
			self.connection = await self.pool.acquire()
		except Exception as e:
			raise ConnectionError(f"Failed to connect to MySQL: {str(e)}")

	async def disconnect(self) -> None:
		if self.connection:
			await self.pool.release(self.connection)
			self.connection = None
		if self.pool:
			self.pool.close()
			await self.pool.wait_closed()
			self.pool = None

	async def test_connection(self) -> Tuple[bool, float, str]:
		start_time = time.time()
		try:
			if not self.connection:
				await self.connect()
			async with self.connection.cursor() as cur:
				await cur.execute('SELECT 1')
				result = await cur.fetchone()
				success = result[0] == 1
				latency = (time.time() - start_time) * 1000
				message = "Connection successful" if success else "Connection failed"
				return success, latency, message
		except Exception as e:
			return False, (time.time() - start_time) * 1000, f"Connection failed: {str(e)}"

	async def execute_query(self, query: str, **kwargs) -> pd.DataFrame:
		if not self.connection:
			await self.connect()
		try:
			async with self.connection.cursor(aiomysql.DictCursor) as cur:
				await cur.execute(query)
				result = await cur.fetchall()
				df = pd.DataFrame(result)
				return df
		except Exception as e:
			raise Exception(f"Query execution failed: {str(e)}")

	async def get_schema(self) -> List[Dict[str, Any]]:
		query = """
		SELECT TABLE_SCHEMA, TABLE_NAME, COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT
		FROM INFORMATION_SCHEMA.COLUMNS
		WHERE TABLE_SCHEMA = DATABASE()
		ORDER BY TABLE_NAME, ORDINAL_POSITION
		"""
		df = await self.execute_query(query)
		schema = []
		for table_name, group in df.groupby(['TABLE_NAME']):
			columns = []
			for _, row in group.iterrows():
				columns.append({
					'name': row['COLUMN_NAME'],
					'type': row['DATA_TYPE'],
					'nullable': row['IS_NULLABLE'] == 'YES',
					'default': row['COLUMN_DEFAULT']
				})
			schema.append({
				'table': table_name,
				'columns': columns
			})
		return schema

	async def get_tables(self) -> List[str]:
		query = "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = DATABASE()"
		df = await self.execute_query(query)
		return df['TABLE_NAME'].tolist() if not df.empty else []
