import motor.motor_asyncio
import pandas as pd
from typing import Dict, Any, List, Tuple
import time
from .base import BaseConnector

class MongoDBConnector(BaseConnector):
	"""MongoDB database connector."""
	def __init__(self, config: Dict[str, Any]):
		super().__init__(config)
		self.client = None
		self.db = None

	async def connect(self) -> None:
		uri = self.config.get('uri') or f"mongodb://{self.config.get('username', '')}:{self.config.get('password', '')}@{self.config.get('host', 'localhost')}:{self.config.get('port', 27017)}/"
		self.client = motor.motor_asyncio.AsyncIOMotorClient(uri)
		self.db = self.client[self.config.get('database')]

	async def disconnect(self) -> None:
		if self.client:
			self.client.close()
			self.client = None
			self.db = None

	async def test_connection(self) -> Tuple[bool, float, str]:
		start_time = time.time()
		try:
			if not self.client:
				await self.connect()
			await self.db.command("ping")
			latency = (time.time() - start_time) * 1000
			return True, latency, "Connection successful"
		except Exception as e:
			return False, (time.time() - start_time) * 1000, f"Connection failed: {str(e)}"

	async def execute_query(self, query: str, **kwargs) -> pd.DataFrame:
		# For MongoDB, query is a collection name, kwargs can include filter, projection, limit, etc.
		if not self.db:
			await self.connect()
		collection = self.db[query]
		cursor = collection.find(kwargs.get('filter', {}), kwargs.get('projection'))
		if 'limit' in kwargs:
			cursor = cursor.limit(kwargs['limit'])
		docs = await cursor.to_list(length=kwargs.get('limit', 100))
		df = pd.DataFrame(docs)
		return df

	async def get_schema(self) -> List[Dict[str, Any]]:
		# MongoDB is schemaless, so we sample documents
		tables = await self.get_tables()
		schema = []
		for table in tables:
			collection = self.db[table]
			sample = await collection.find_one()
			columns = []
			if sample:
				for k, v in sample.items():
					columns.append({'name': k, 'type': str(type(v))})
			schema.append({'collection': table, 'columns': columns})
		return schema

	async def get_tables(self) -> List[str]:
		if not self.db:
			await self.connect()
		return await self.db.list_collection_names()
