from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import pandas as pd

class BaseConnector(ABC):
    """Base class for all data source connectors."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection = None
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the data source."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the data source."""
        pass
    
    @abstractmethod
    async def test_connection(self) -> Tuple[bool, float, str]:
        """
        Test connection to the data source.
        
        Returns:
            Tuple[bool, float, str]: (success, latency_ms, message)
        """
        pass
    
    @abstractmethod
    async def execute_query(self, query: str, **kwargs) -> pd.DataFrame:
        """
        Execute a query on the data source.
        
        Args:
            query: Query string
            **kwargs: Additional parameters
            
        Returns:
            pandas.DataFrame: Query results
        """
        pass
    
    @abstractmethod
    async def get_schema(self) -> List[Dict[str, Any]]:
        """
        Get database/collection schema.
        
        Returns:
            List of tables/collections with their schemas
        """
        pass
    
    @abstractmethod
    async def get_tables(self) -> List[str]:
        """
        Get list of tables/collections.
        
        Returns:
            List of table/collection names
        """
        pass
    
    async def get_sample_data(self, table: str, limit: int = 100) -> pd.DataFrame:
        """
        Get sample data from a table.
        
        Args:
            table: Table/collection name
            limit: Number of rows to return
            
        Returns:
            pandas.DataFrame: Sample data
        """
        query = f"SELECT * FROM {table} LIMIT {limit}"
        return await self.execute_query(query)