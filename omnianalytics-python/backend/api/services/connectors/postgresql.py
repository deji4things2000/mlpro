import asyncpg
import pandas as pd
from typing import Dict, Any, List, Tuple
import time
from .base import BaseConnector

class PostgreSQLConnector(BaseConnector):
    """PostgreSQL database connector."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.pool = None
    
    async def connect(self) -> None:
        """Connect to PostgreSQL database."""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.config.get('host', 'localhost'),
                port=self.config.get('port', 5432),
                database=self.config.get('database'),
                user=self.config.get('username'),
                password=self.config.get('password'),
                min_size=1,
                max_size=10,
            )
            self.connection = await self.pool.acquire()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to PostgreSQL: {str(e)}")
    
    async def disconnect(self) -> None:
        """Disconnect from PostgreSQL database."""
        if self.connection:
            await self.pool.release(self.connection)
            self.connection = None
        
        if self.pool:
            await self.pool.close()
            self.pool = None
    
    async def test_connection(self) -> Tuple[bool, float, str]:
        """Test PostgreSQL connection."""
        start_time = time.time()
        
        try:
            if not self.connection:
                await self.connect()
            
            async with self.connection.transaction():
                result = await self.connection.fetchrow('SELECT 1 as test')
                success = result['test'] == 1
                latency = (time.time() - start_time) * 1000  # Convert to ms
                message = "Connection successful" if success else "Connection failed"
                
                return success, latency, message
                
        except Exception as e:
            return False, (time.time() - start_time) * 1000, f"Connection failed: {str(e)}"
    
    async def execute_query(self, query: str, **kwargs) -> pd.DataFrame:
        """Execute a SQL query."""
        if not self.connection:
            await self.connect()
        
        try:
            async with self.connection.transaction():
                result = await self.connection.fetch(query)
                
                # Convert to DataFrame
                if result:
                    columns = list(result[0].keys())
                    data = [list(row.values()) for row in result]
                    df = pd.DataFrame(data, columns=columns)
                else:
                    df = pd.DataFrame()
                
                return df
                
        except Exception as e:
            raise Exception(f"Query execution failed: {str(e)}")
    
    async def get_schema(self) -> List[Dict[str, Any]]:
        """Get database schema."""
        query = """
        SELECT 
            table_schema,
            table_name,
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns
        WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY table_schema, table_name, ordinal_position
        """
        
        df = await self.execute_query(query)
        
        # Group by table
        schema = []
        for (schema_name, table_name), group in df.groupby(['table_schema', 'table_name']):
            columns = []
            for _, row in group.iterrows():
                columns.append({
                    'name': row['column_name'],
                    'type': row['data_type'],
                    'nullable': row['is_nullable'] == 'YES',
                    'default': row['column_default']
                })
            
            schema.append({
                'schema': schema_name,
                'table': table_name,
                'columns': columns
            })
        
        return schema
    
    async def get_tables(self) -> List[str]:
        """Get list of tables."""
        query = """
        SELECT schemaname, tablename 
        FROM pg_catalog.pg_tables 
        WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
        ORDER BY schemaname, tablename
        """
        
        df = await self.execute_query(query)
        tables = [f"{row['schemaname']}.{row['tablename']}" for _, row in df.iterrows()]
        
        return tables