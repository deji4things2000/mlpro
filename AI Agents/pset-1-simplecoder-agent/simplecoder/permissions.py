"""
Permissions management for SimpleCoder agent.

This module manages task- and session-level permissions for reading, writing files,
and executing potentially sensitive operations. It provides:
1. Session-based permission tracking
2. Task-level permission scoping
3. User confirmation for sensitive operations
4. Permission inheritance and escalation
"""

from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from pathlib import Path
import logging

# Setup logging
logger = logging.getLogger(__name__)

class PermissionLevel(Enum):
    """Permission levels for operations."""
    NONE = 0
    READ = 1
    WRITE = 2
    EXECUTE = 3
    ADMIN = 4  # For system-level operations

class OperationType(Enum):
    """Types of operations that require permissions."""
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    EXECUTE_CODE = "execute_code"
    DELETE_FILE = "delete_file"
    LIST_DIRECTORY = "list_directory"
    SEARCH_FILES = "search_files"
    EDIT_FILE = "edit_file"
    CREATE_FILE = "create_file"
    INSTALL_PACKAGE = "install_package"
    NETWORK_REQUEST = "network_request"

@dataclass
class PermissionRequest:
    """Represents a permission request."""
    operation: OperationType
    target: str  # e.g., file path, directory, URL
    context: Dict[str, Any] = field(default_factory=dict)
    required_level: PermissionLevel = PermissionLevel.READ
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'operation': self.operation.value,
            'target': self.target,
            'context': self.context,
            'required_level': self.required_level.value
        }

@dataclass
class PermissionGrant:
    """Represents a granted permission."""
    request: PermissionRequest
    granted_level: PermissionLevel
    granted_by: str  # 'user', 'session_policy', 'inherited'
    timestamp: float
    expires_at: Optional[float] = None  # None means doesn't expire
    
    def is_valid(self) -> bool:
        """Check if the permission grant is still valid."""
        import time
        if self.expires_at and time.time() > self.expires_at:
            return False
        return self.granted_level.value >= self.request.required_level.value

class PermissionScope(Enum):
    """Scope of permissions."""
    SESSION = "session"  # Applies to entire session
    TASK = "task"        # Applies to specific task
    DIRECTORY = "directory"  # Applies to directory tree
    FILE = "file"        # Applies to specific file

@dataclass
class PermissionPolicy:
    """Defines a permission policy rule."""
    scope: PermissionScope
    target_pattern: str  # Can be glob pattern or regex
    allowed_operations: Set[OperationType]
    max_level: PermissionLevel
    conditions: Dict[str, Any] = field(default_factory=dict)
    inheritable: bool = True  # Whether child paths inherit this policy
    
    def matches(self, operation: OperationType, target: str) -> bool:
        """Check if this policy applies to the given operation and target."""
        if operation not in self.allowed_operations:
            return False
            
        # Simple glob pattern matching (can be enhanced to regex)
        from fnmatch import fnmatch
        return fnmatch(target, self.target_pattern)

class PermissionManager:
    """Manages permissions for the SimpleCoder agent."""
    
    def __init__(self, session_id: str, config_path: Optional[str] = None):
        self.session_id = session_id
        self.permission_grants: List[PermissionGrant] = []
        self.policies: List[PermissionPolicy] = []
        self.user_confirmed: Set[str] = set()  # Cache of user-confirmed operations
        self.permission_cache: Dict[str, PermissionGrant] = {}
        
        # Default working directory (can be configured)
        self.working_directory = os.getcwd()
        self.safe_directories = {
            self.working_directory,
            os.path.join(self.working_directory, "src"),
            os.path.join(self.working_directory, "tests"),
        }
        
        # Load default policies
        self._load_default_policies()
        
        # Load custom policies if config file exists
        if config_path and os.path.exists(config_path):
            self.load_policies_from_file(config_path)
    
    def _load_default_policies(self):
        """Load default security policies."""
        
        # Allow reading/writing in working directory and subdirectories
        self.policies.append(PermissionPolicy(
            scope=PermissionScope.DIRECTORY,
            target_pattern=os.path.join(self.working_directory, "**"),
            allowed_operations={
                OperationType.READ_FILE,
                OperationType.WRITE_FILE,
                OperationType.LIST_DIRECTORY,
                OperationType.SEARCH_FILES,
                OperationType.EDIT_FILE,
                OperationType.CREATE_FILE,
            },
            max_level=PermissionLevel.WRITE,
            inheritable=True
        ))
        
        # Allow reading in safe directories
        for safe_dir in self.safe_directories:
            self.policies.append(PermissionPolicy(
                scope=PermissionScope.DIRECTORY,
                target_pattern=os.path.join(safe_dir, "**"),
                allowed_operations={
                    OperationType.READ_FILE,
                    OperationType.LIST_DIRECTORY,
                    OperationType.SEARCH_FILES,
                },
                max_level=PermissionLevel.READ,
                inheritable=True
            ))
        
        # Restrict operations outside working directory
        self.policies.append(PermissionPolicy(
            scope=PermissionScope.DIRECTORY,
            target_pattern="/**",  # Any path
            allowed_operations=set(),  # No operations allowed by default
            max_level=PermissionLevel.NONE,
            inheritable=False
        ))
        
        # Always require user confirmation for these operations
        self.policies.append(PermissionPolicy(
            scope=PermissionScope.SESSION,
            target_pattern="**",
            allowed_operations={
                OperationType.DELETE_FILE,
                OperationType.EXECUTE_CODE,
                OperationType.INSTALL_PACKAGE,
                OperationType.NETWORK_REQUEST,
            },
            max_level=PermissionLevel.ADMIN,  # Requires explicit user confirmation
            conditions={"require_user_confirmation": True},
            inheritable=False
        ))
    
    def check_permission(self, request: PermissionRequest) -> bool:
        """
        Check if an operation is permitted.
        Returns True if allowed, False if denied.
        """
        # Check cache first
        cache_key = f"{request.operation.value}:{request.target}"
        if cache_key in self.permission_cache:
            grant = self.permission_cache[cache_key]
            if grant.is_valid():
                return True
        
        # Check applicable policies
        applicable_policies = [
            policy for policy in self.policies
            if policy.matches(request.operation, request.target)
        ]
        
        if not applicable_policies:
            logger.warning(f"No policy found for operation {request.operation} on {request.target}")
            return False
        
        # Get the most permissive policy (highest max_level)
        applicable_policies.sort(key=lambda p: p.max_level.value, reverse=True)
        best_policy = applicable_policies[0]
        
        # Check if operation level is within policy limits
        if request.required_level.value > best_policy.max_level.value:
            logger.warning(
                f"Operation {request.operation} requires level {request.required_level} "
                f"but policy only allows {best_policy.max_level}"
            )
            return False
        
        # Check if user confirmation is required
        if best_policy.conditions.get("require_user_confirmation", False):
            if not self._has_user_confirmation(request):
                logger.info(f"User confirmation required for {request.operation} on {request.target}")
                return False
        
        # Create and cache permission grant
        grant = PermissionGrant(
            request=request,
            granted_level=best_policy.max_level,
            granted_by="policy",
            timestamp=os.path.getatime(__file__)  # Use file access time as timestamp
        )
        
        self.permission_grants.append(grant)
        self.permission_cache[cache_key] = grant
        
        return True
    
    def _has_user_confirmation(self, request: PermissionRequest) -> bool:
        """
        Check if user has confirmed this operation.
        In a real implementation, this would prompt the user.
        For this assignment, we prompt in the CLI and cache the decision for the session.
        """
        confirmation_key = f"{request.operation.value}:{request.target}"
        
        if confirmation_key in self.user_confirmed:
            return True
        
        # Optional escape hatch for demos / autograding.
        if os.getenv("SIMPLECODER_AUTO_APPROVE", "").lower() in {"1", "true", "yes"}:
            self.user_confirmed.add(confirmation_key)
            return True

        # Auto-confirm *reads* inside safe directories to keep UX smooth.
        try:
            target_path = Path(request.target)
            if request.operation in {OperationType.READ_FILE, OperationType.LIST_DIRECTORY, OperationType.SEARCH_FILES}:
                for safe_dir in self.safe_directories:
                    if str(target_path).startswith(str(safe_dir)):
                        self.user_confirmed.add(confirmation_key)
                        return True
        except Exception:
            pass

        return False
    
    def request_permission(self, request: PermissionRequest) -> bool:
        """
        Request permission for an operation, prompting user if needed.
        Returns True if granted, False if denied.
        """
        if self.check_permission(request):
            return True
        
        # If not automatically granted, ask user
        return self._prompt_user_for_permission(request)
    
    def _prompt_user_for_permission(self, request: PermissionRequest) -> bool:
        """Prompt user for permission (simulated for assignment)."""
        print("\n🔒 PERMISSION REQUIRED")
        print(f"Operation: {request.operation.value}")
        print(f"Target: {request.target}")
        print(f"Reason: {request.context.get('reason', 'No reason provided')}")
        print(f"Required permission level: {request.required_level.name}")
        print("Allow? [y]es / [n]o / [a]lways (this session)")

        try:
            response = input("> ").strip().lower()
        except EOFError:
            response = "n"

        if response in ["y", "yes"]:
            # Create a user-granted permission
            import time
            grant = PermissionGrant(
                request=request,
                granted_level=PermissionLevel.ADMIN,  # User grants admin level
                granted_by="user",
                timestamp=time.time(),
                expires_at=time.time() + 3600  # Expire in 1 hour
            )
            
            self.permission_grants.append(grant)
            cache_key = f"{request.operation.value}:{request.target}"
            self.permission_cache[cache_key] = grant
            self.user_confirmed.add(cache_key)
            
            print("Permission granted by user")
            return True
        elif response in ["a", "always"]:
            import time
            grant = PermissionGrant(
                request=request,
                granted_level=PermissionLevel.ADMIN,
                granted_by="user",
                timestamp=time.time(),
                expires_at=None,
            )

            self.permission_grants.append(grant)
            cache_key = f"{request.operation.value}:{request.target}"
            self.permission_cache[cache_key] = grant
            self.user_confirmed.add(cache_key)
            print("Permission granted (always for this session)")
            return True
        else:
            print("Permission denied by user")
            return False
    
    def add_policy(self, policy: PermissionPolicy):
        """Add a new permission policy."""
        self.policies.append(policy)
        logger.info(f"Added policy for scope {policy.scope} on pattern {policy.target_pattern}")
    
    def load_policies_from_file(self, config_path: str):
        """Load policies from a JSON configuration file."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            for policy_config in config.get('policies', []):
                policy = PermissionPolicy(
                    scope=PermissionScope(policy_config['scope']),
                    target_pattern=policy_config['target_pattern'],
                    allowed_operations={
                        OperationType(op) for op in policy_config['allowed_operations']
                    },
                    max_level=PermissionLevel(policy_config['max_level']),
                    conditions=policy_config.get('conditions', {}),
                    inheritable=policy_config.get('inheritable', True)
                )
                self.add_policy(policy)
                
            logger.info(f"Loaded {len(config.get('policies', []))} policies from {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load policies from {config_path}: {e}")
    
    def export_policies(self, output_path: str):
        """Export current policies to a JSON file."""
        policies_data = []
        
        for policy in self.policies:
            policy_data = {
                'scope': policy.scope.value,
                'target_pattern': policy.target_pattern,
                'allowed_operations': [op.value for op in policy.allowed_operations],
                'max_level': policy.max_level.value,
                'conditions': policy.conditions,
                'inheritable': policy.inheritable
            }
            policies_data.append(policy_data)
        
        config = {
            'session_id': self.session_id,
            'working_directory': self.working_directory,
            'policies': policies_data
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(config, f, indent=2)
            logger.info(f"Exported {len(policies_data)} policies to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export policies: {e}")
    
    def get_permission_summary(self) -> Dict:
        """Get a summary of current permissions and policies."""
        return {
            'session_id': self.session_id,
            'total_grants': len(self.permission_grants),
            'total_policies': len(self.policies),
            'working_directory': self.working_directory,
            'safe_directories': list(self.safe_directories),
            'active_grants': sum(1 for grant in self.permission_grants if grant.is_valid())
        }
    
    def clear_expired_grants(self):
        """Clean up expired permission grants."""
        initial_count = len(self.permission_grants)
        self.permission_grants = [grant for grant in self.permission_grants if grant.is_valid()]
        
        # Also clear cache
        self.permission_cache = {
            k: v for k, v in self.permission_cache.items() if v.is_valid()
        }
        
        cleared = initial_count - len(self.permission_grants)
        if cleared > 0:
            logger.info(f"Cleared {cleared} expired permission grants")


# Factory function for easy creation
def create_permission_manager(session_id: str = None, config_path: str = None) -> PermissionManager:
    """Factory function to create a permission manager."""
    if not session_id:
        import uuid
        session_id = str(uuid.uuid4())[:8]
    
    return PermissionManager(session_id, config_path)


# Example usage
if __name__ == "__main__":
    # Create a permission manager
    pm = create_permission_manager("test_session")
    
    # Test permission requests
    test_requests = [
        PermissionRequest(
            operation=OperationType.READ_FILE,
            target="./main.py",
            context={"reason": "Need to understand existing code"},
            required_level=PermissionLevel.READ
        ),
        PermissionRequest(
            operation=OperationType.WRITE_FILE,
            target="./new_file.py",
            context={"reason": "Create new implementation"},
            required_level=PermissionLevel.WRITE
        ),
        PermissionRequest(
            operation=OperationType.DELETE_FILE,
            target="/etc/passwd",  # Sensitive file!
            context={"reason": "Clean up system file"},
            required_level=PermissionLevel.ADMIN
        ),
    ]
    
    for req in test_requests:
        print(f"\nChecking permission for {req.operation.value} on {req.target}")
        if pm.request_permission(req):
            print("✅ Permission granted")
        else:
            print("❌ Permission denied")
    
    # Print summary
    summary = pm.get_permission_summary()
    print(f"\nPermission Summary: {summary}")