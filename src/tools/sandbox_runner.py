"""
Sandbox test runner for secure test execution in isolated containers.
"""

import docker
import tempfile
import os
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class SandboxConfig:
    """Configuration for sandbox execution"""
    memory_limit: str = "512m"
    cpu_limit: float = 1.0
    network_mode: str = "none"  # No network access by default
    timeout_seconds: int = 300
    read_only: bool = True
    remove: bool = True

@dataclass
class TestResult:
    """Result of test execution"""
    success: bool
    output: str
    error: str
    execution_time: float
    exit_code: int
    container_id: str

class SandboxTestRunner:
    """Secure test runner using Docker containers"""
    
    def __init__(self, docker_client: Optional[docker.DockerClient] = None):
        """Initialize the sandbox runner"""
        try:
            self.client = docker_client or docker.from_env()
            self.test_containers = []
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            raise
    
    def create_test_container(self, test_code: str, test_name: str = "test") -> str:
        """Create a temporary test file and return the path"""
        # Create temporary directory for test files
        temp_dir = tempfile.mkdtemp(prefix=f"ai_test_{test_name}_")
        test_file = os.path.join(temp_dir, f"{test_name}.spec.ts")
        
        with open(test_file, 'w') as f:
            f.write(test_code)
        
        return temp_dir
    
    def run_test_in_sandbox(self, test_code: str, test_name: str = "test", 
                           config: Optional[SandboxConfig] = None) -> TestResult:
        """Run a test in an isolated sandbox container"""
        if config is None:
            config = SandboxConfig()
        
        start_time = time.time()
        container_id = None
        
        try:
            # Create temporary test files
            temp_dir = self.create_test_container(test_code, test_name)
            
            # Create container with restricted permissions
            container = self.client.containers.run(
                image="mcr.microsoft.com/playwright:v1.44.0-focal",
                command=["npx", "playwright", "test", f"/tmp/{test_name}.spec.ts", "--reporter=list"],
                volumes={
                    temp_dir: {
                        'bind': '/tmp',
                        'mode': 'ro'  # Read-only mount
                    }
                },
                environment={
                    'PLAYWRIGHT_BROWSERS_PATH': '/ms-playwright',
                    'NODE_ENV': 'test'
                },
                mem_limit=config.memory_limit,
                cpu_quota=int(config.cpu_limit * 100000),
                cpu_period=100000,
                network_mode=config.network_mode,
                read_only=config.read_only,
                remove=config.remove,
                detach=True
            )
            
            container_id = container.id
            self.test_containers.append(container_id)
            
            # Wait for completion with timeout
            try:
                result = container.wait(timeout=config.timeout_seconds)
                logs = container.logs().decode('utf-8')
                exit_code = result['StatusCode']
                
                success = exit_code == 0
                error = "" if success else logs
                
                return TestResult(
                    success=success,
                    output=logs,
                    error=error,
                    execution_time=time.time() - start_time,
                    exit_code=exit_code,
                    container_id=container_id
                )
                
            except Exception as e:
                # Force stop container on timeout or error
                try:
                    container.stop(timeout=5)
                except:
                    pass
                raise e
                
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return TestResult(
                success=False,
                output="",
                error=str(e),
                execution_time=time.time() - start_time,
                exit_code=-1,
                container_id=container_id or "unknown"
            )
        
        finally:
            # Cleanup temporary files
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass
    
    def run_multiple_tests(self, tests: List[Dict], config: Optional[SandboxConfig] = None) -> List[TestResult]:
        """Run multiple tests in parallel sandboxes"""
        if config is None:
            config = SandboxConfig()
        
        results = []
        for test in tests:
            test_code = test.get('code', '')
            test_name = test.get('name', f'test_{len(results)}')
            
            result = self.run_test_in_sandbox(test_code, test_name, config)
            results.append(result)
            
            # Small delay between tests to avoid overwhelming system
            time.sleep(1)
        
        return results
    
    def cleanup(self):
        """Clean up any remaining test containers"""
        for container_id in self.test_containers:
            try:
                container = self.client.containers.get(container_id)
                if container.status == 'running':
                    container.stop(timeout=5)
                container.remove()
            except Exception as e:
                logger.warning(f"Failed to cleanup container {container_id}: {e}")
        
        self.test_containers.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

def create_sandbox_config(allow_network: bool = False, 
                         memory_limit: str = "512m",
                         timeout_seconds: int = 300) -> SandboxConfig:
    """Create a sandbox configuration with specified settings"""
    return SandboxConfig(
        memory_limit=memory_limit,
        network_mode="bridge" if allow_network else "none",
        timeout_seconds=timeout_seconds,
        read_only=True,
        remove=True
    )

# Example usage
if __name__ == "__main__":
    # Test the sandbox runner
    test_code = """
import { test, expect } from '@playwright/test';

test('basic test', async ({ page }) => {
  await page.goto('https://example.com');
  await expect(page).toHaveTitle(/Example Domain/);
});
"""
    
    try:
        with SandboxTestRunner() as runner:
            result = runner.run_test_in_sandbox(test_code, "example_test")
            print(f"Test result: {result}")
    except Exception as e:
        print(f"Failed to run test: {e}") 