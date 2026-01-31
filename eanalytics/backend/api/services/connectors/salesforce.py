from simple_salesforce import Salesforce
import pandas as pd
from typing import Dict, Any, List, Tuple
import time
from .base import BaseConnector

class SalesforceConnector(BaseConnector):
	"""Salesforce connector."""
	def __init__(self, config: Dict[str, Any]):
		super().__init__(config)
		self.sf = None

	async def connect(self) -> None:
		self.sf = Salesforce(
			username=self.config.get('username'),
			password=self.config.get('password'),
			security_token=self.config.get('security_token'),
			domain=self.config.get('domain', 'login')
		)

	async def disconnect(self) -> None:
		self.sf = None

	async def test_connection(self) -> Tuple[bool, float, str]:
		start_time = time.time()
		try:
			if not self.sf:
				await self.connect()
			# Try a simple query
			result = self.sf.query("SELECT Id FROM Account LIMIT 1")
			latency = (time.time() - start_time) * 1000
			return True, latency, "Connection successful"
		except Exception as e:
			return False, (time.time() - start_time) * 1000, f"Connection failed: {str(e)}"

	async def execute_query(self, query: str, **kwargs) -> pd.DataFrame:
		if not self.sf:
			await self.connect()
		result = self.sf.query_all(query)
		records = result['records']
		df = pd.DataFrame(records)
		return df

	async def get_schema(self) -> List[Dict[str, Any]]:
		if not self.sf:
			await self.connect()
		# List all objects
		objects = self.sf.describe()['sobjects']
		schema = []
		for obj in objects:
			fields = self.sf.__getattr__(obj['name']).describe()['fields']
			columns = [{'name': f['name'], 'type': f['type']} for f in fields]
			schema.append({'object': obj['name'], 'columns': columns})
		return schema

	async def get_tables(self) -> List[str]:
		if not self.sf:
			await self.connect()
		objects = self.sf.describe()['sobjects']
		return [obj['name'] for obj in objects]
