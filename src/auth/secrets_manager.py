"""
Secrets management system supporting AWS Secrets Manager and HashiCorp Vault.
"""

import os
import json
import logging
from typing import Dict, Optional, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class SecretsManager(ABC):
    """Abstract base class for secrets management"""
    
    @abstractmethod
    def get_secret(self, secret_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a secret by name"""
        pass
    
    @abstractmethod
    def set_secret(self, secret_name: str, secret_value: Dict[str, Any]) -> bool:
        """Set a secret value"""
        pass
    
    @abstractmethod
    def delete_secret(self, secret_name: str) -> bool:
        """Delete a secret"""
        pass

class AWSSecretsManager(SecretsManager):
    """AWS Secrets Manager implementation"""
    
    def __init__(self, region_name: Optional[str] = None):
        try:
            import boto3
            self.client = boto3.client(
                'secretsmanager',
                region_name=region_name or os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
            )
            logger.info("AWS Secrets Manager initialized successfully")
        except ImportError:
            logger.error("boto3 not installed. Install with: pip install boto3")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize AWS Secrets Manager: {e}")
            raise
    
    def get_secret(self, secret_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a secret from AWS Secrets Manager"""
        try:
            response = self.client.get_secret_value(SecretId=secret_name)
            if 'SecretString' in response:
                return json.loads(response['SecretString'])
            else:
                logger.warning(f"Secret {secret_name} not found or not a string")
                return None
        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_name}: {e}")
            return None
    
    def set_secret(self, secret_name: str, secret_value: Dict[str, Any]) -> bool:
        """Set a secret in AWS Secrets Manager"""
        try:
            secret_string = json.dumps(secret_value)
            self.client.create_secret(
                Name=secret_name,
                SecretString=secret_string
            )
            logger.info(f"Secret {secret_name} created successfully")
            return True
        except self.client.exceptions.ResourceExistsException:
            # Update existing secret
            try:
                self.client.update_secret(
                    SecretId=secret_name,
                    SecretString=json.dumps(secret_value)
                )
                logger.info(f"Secret {secret_name} updated successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to update secret {secret_name}: {e}")
                return False
        except Exception as e:
            logger.error(f"Failed to set secret {secret_name}: {e}")
            return False
    
    def delete_secret(self, secret_name: str) -> bool:
        """Delete a secret from AWS Secrets Manager"""
        try:
            self.client.delete_secret(
                SecretId=secret_name,
                ForceDeleteWithoutRecovery=True
            )
            logger.info(f"Secret {secret_name} deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to delete secret {secret_name}: {e}")
            return False

class HashiCorpVault(SecretsManager):
    """HashiCorp Vault implementation"""
    
    def __init__(self, vault_url: str, token: str):
        try:
            import hvac
            self.client = hvac.Client(url=vault_url, token=token)
            if not self.client.is_authenticated():
                raise Exception("Failed to authenticate with Vault")
            logger.info("HashiCorp Vault initialized successfully")
        except ImportError:
            logger.error("hvac not installed. Install with: pip install hvac")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize HashiCorp Vault: {e}")
            raise
    
    def get_secret(self, secret_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a secret from HashiCorp Vault"""
        try:
            response = self.client.secrets.kv.v2.read_secret_version(
                path=secret_name,
                mount_point='secret'
            )
            return response['data']['data']
        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_name}: {e}")
            return None
    
    def set_secret(self, secret_name: str, secret_value: Dict[str, Any]) -> bool:
        """Set a secret in HashiCorp Vault"""
        try:
            self.client.secrets.kv.v2.create_or_update_secret(
                path=secret_name,
                secret=secret_value,
                mount_point='secret'
            )
            logger.info(f"Secret {secret_name} set successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to set secret {secret_name}: {e}")
            return False
    
    def delete_secret(self, secret_name: str) -> bool:
        """Delete a secret from HashiCorp Vault"""
        try:
            self.client.secrets.kv.v2.delete_metadata_and_all_versions(
                path=secret_name,
                mount_point='secret'
            )
            logger.info(f"Secret {secret_name} deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to delete secret {secret_name}: {e}")
            return False

class LocalSecretsManager(SecretsManager):
    """Local file-based secrets manager for development"""
    
    def __init__(self, secrets_dir: str = "secrets"):
        self.secrets_dir = secrets_dir
        os.makedirs(secrets_dir, exist_ok=True)
        logger.info(f"Local secrets manager initialized at {secrets_dir}")
    
    def get_secret(self, secret_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a secret from local file"""
        try:
            secret_path = os.path.join(self.secrets_dir, f"{secret_name}.json")
            if os.path.exists(secret_path):
                with open(secret_path, 'r') as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve secret {secret_name}: {e}")
            return None
    
    def set_secret(self, secret_name: str, secret_value: Dict[str, Any]) -> bool:
        """Set a secret in local file"""
        try:
            secret_path = os.path.join(self.secrets_dir, f"{secret_name}.json")
            with open(secret_path, 'w') as f:
                json.dump(secret_value, f, indent=2)
            logger.info(f"Secret {secret_name} saved locally")
            return True
        except Exception as e:
            logger.error(f"Failed to set secret {secret_name}: {e}")
            return False
    
    def delete_secret(self, secret_name: str) -> bool:
        """Delete a secret from local file"""
        try:
            secret_path = os.path.join(self.secrets_dir, f"{secret_name}.json")
            if os.path.exists(secret_path):
                os.remove(secret_path)
                logger.info(f"Secret {secret_name} deleted locally")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete secret {secret_name}: {e}")
            return False

def get_secrets_manager() -> SecretsManager:
    """Factory function to get the appropriate secrets manager"""
    # Check environment variables for configuration
    vault_url = os.getenv('VAULT_URL')
    vault_token = os.getenv('VAULT_TOKEN')
    aws_region = os.getenv('AWS_DEFAULT_REGION')
    
    if vault_url and vault_token:
        logger.info("Using HashiCorp Vault")
        return HashiCorpVault(vault_url, vault_token)
    elif aws_region or os.getenv('AWS_ACCESS_KEY_ID'):
        logger.info("Using AWS Secrets Manager")
        return AWSSecretsManager(aws_region)
    else:
        logger.info("Using local secrets manager")
        return LocalSecretsManager()

# Common secret names
class SecretNames:
    OPENAI_API_KEY = "openai/api_key"
    AWS_CREDENTIALS = "aws/credentials"
    DATABASE_URL = "database/url"
    JWT_SECRET = "jwt/secret"
    TEST_CREDENTIALS = "test/credentials"

# Example usage
if __name__ == "__main__":
    # Test the secrets manager
    try:
        secrets_mgr = get_secrets_manager()
        
        # Test setting and getting a secret
        test_secret = {"api_key": "test123", "user": "testuser"}
        secrets_mgr.set_secret("test/example", test_secret)
        
        retrieved = secrets_mgr.get_secret("test/example")
        print(f"Retrieved secret: {retrieved}")
        
        # Cleanup
        secrets_mgr.delete_secret("test/example")
        
    except Exception as e:
        print(f"Secrets manager test failed: {e}") 