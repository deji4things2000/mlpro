# QueryEngine service for executing queries on data sources
from ..database import DataSource
from .connectors import get_connector

class QueryEngine:
	async def execute_single_source(self, data_source: DataSource, query: str, params=None):
		connector = get_connector(data_source.type, data_source.connection_config)
		await connector.connect()
		try:
			df = await connector.execute_query(query, **(params or {}))
			return df.to_dict(orient="records")
		finally:
			await connector.disconnect()

	async def execute_cross_source(self, query_request, user_id, db):
		# Placeholder for cross-source query logic
		# For now, just return a message
		return {"message": "Cross-source query not yet implemented"}
