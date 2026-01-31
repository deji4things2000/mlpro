from typing import Dict, Any
from .base import BaseConnector
from .postgresql import PostgreSQLConnector
from .mysql import MySQLConnector
from .mongodb import MongoDBConnector
from .salesforce import SalesforceConnector
from .csv_file import CSVFileConnector

CONNECTOR_MAP = {
    "postgresql": PostgreSQLConnector,
    "mysql": MySQLConnector,
    "mongodb": MongoDBConnector,
    "salesforce": SalesforceConnector,
    "csv": CSVFileConnector,
    "excel": CSVFileConnector,  # Same as CSV for now
    "json": CSVFileConnector,   # Same as CSV for now
}

def get_connector(source_type: str, config: Dict[str, Any]) -> BaseConnector:
    """Get connector instance for the given source type."""
    connector_class = CONNECTOR_MAP.get(source_type.lower())
    
    if not connector_class:
        raise ValueError(f"Unsupported data source type: {source_type}")
    
    return connector_class(config)