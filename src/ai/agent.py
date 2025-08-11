import os
import json
from typing import Dict, Any, List, Optional
import requests
import logging
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment configuration
OPENAI_KEY = os.getenv('OPENAI_API_KEY')
BEDROCK_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
BEDROCK_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
BEDROCK_REGION = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
API_BASE = os.getenv('FASTAPI_URL', 'http://localhost:8000')

# LLM Configuration
LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'openai').lower()
LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-4o-mini')

class PredictRequest(BaseModel):
    file: str = Field(..., description="File path to analyze")
    features: Dict[str, Any] = Field(default_factory=dict, description="Code metrics features")

class GenerateRequest(BaseModel):
    source: Dict[str, Any] = Field(..., description="Source information for test generation")

class RunTestsRequest(BaseModel):
    tests: List[str] = Field(..., description="List of test files to run")

class SelfHealRequest(BaseModel):
    error: str = Field(..., description="Error message from test failure")
    dom: str = Field(default="", description="DOM snapshot for analysis")

def get_llm_client():
    """Initialize LLM client based on configuration"""
    try:
        if LLM_PROVIDER == 'openai' and OPENAI_KEY:
            logger.info("Using OpenAI LLM")
            return OpenAI(api_key=OPENAI_KEY, temperature=0, model=LLM_MODEL)
        elif LLM_PROVIDER == 'bedrock' and BEDROCK_ACCESS_KEY:
            logger.info("Using AWS Bedrock LLM")
            try:
                from langchain.llms import Bedrock
                return Bedrock(
                    model_id="anthropic.claude-v2",
                    region_name=BEDROCK_REGION,
                    credentials_profile_name=None
                )
            except ImportError:
                logger.error("boto3 not installed. Install with: pip install boto3")
                raise
        elif LLM_PROVIDER == 'llama3':
            logger.info("Using local LLaMA 3 model (stub)")
            # TODO: Add actual LLaMA 3 integration
            return None
        elif LLM_PROVIDER == 'starcoder':
            logger.info("Using local StarCoder model (stub)")
            # TODO: Add actual StarCoder integration
            return None
        else:
            logger.warning("No valid LLM configuration found. Using fallback.")
            return None
    except Exception as e:
        logger.error(f"Failed to initialize LLM client: {e}")
        return None

# API call functions with error handling and validation
def call_predict(file: str, features: Dict) -> Dict:
    """Call the predict endpoint with validation"""
    try:
        if not file:
            raise ValueError("File path is required")
        
        resp = requests.post(
            f'{API_BASE}/predict', 
            json={'repo':'demo','file':file,'features':features}, 
            timeout=30
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API call failed: {e}")
        return {"error": f"API call failed: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"error": f"Unexpected error: {str(e)}"}

def call_generate(source: Dict) -> Dict:
    """Call the generate test endpoint with validation"""
    try:
        if not source:
            raise ValueError("Source information is required")
        
        resp = requests.post(
            f'{API_BASE}/generate_test', 
            json={'source':source}, 
            timeout=60
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API call failed: {e}")
        return {"error": f"API call failed: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"error": f"Unexpected error: {str(e)}"}

def call_run_tests(tests: List[str]) -> Dict:
    """Call the run tests endpoint with validation"""
    try:
        if not tests:
            raise ValueError("Test list is required")
        
        resp = requests.post(
            f'{API_BASE}/run_tests', 
            json={'tests':tests}, 
            timeout=600
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API call failed: {e}")
        return {"error": f"API call failed: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"error": f"Unexpected error: {str(e)}"}

def call_self_heal(payload: Dict) -> Dict:
    """Call the self heal endpoint with validation"""
    try:
        if not payload.get('error'):
            raise ValueError("Error message is required")
        
        resp = requests.post(
            f'{API_BASE}/self_heal', 
            json=payload, 
            timeout=60
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API call failed: {e}")
        return {"error": f"API call failed: {str(e)}"}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"error": f"Unexpected error: {str(e)}"}

# Enhanced tool wrappers with input validation
def tool_predict(args: Dict) -> str:
    """Predict defect risk for a file with validation"""
    try:
        request = PredictRequest(**args)
        result = call_predict(request.file, request.features)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Validation failed: {str(e)}"}, indent=2)

def tool_generate(args: Dict) -> str:
    """Generate test code with validation"""
    try:
        request = GenerateRequest(**args)
        result = call_generate(request.source)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Validation failed: {str(e)}"}, indent=2)

def tool_run(args: Dict) -> str:
    """Run tests with validation"""
    try:
        request = RunTestsRequest(**args)
        result = call_run_tests(request.tests)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Validation failed: {str(e)}"}, indent=2)

def tool_heal(args: Dict) -> str:
    """Attempt to self-heal failing selectors with validation"""
    try:
        request = SelfHealRequest(**args)
        result = call_self_heal(request.dict())
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Validation failed: {str(e)}"}, indent=2)

def _get_tools():
    try:
        from langchain.agents import Tool
    except Exception:
        # lightweight fallback: return simple descriptors; not used without LC
        class _T:
            def __init__(self, name, func, description):
                self.name, self.func, self.description = name, func, description
        Tool = _T  # type: ignore
    return [
        Tool(
            name="predict_defects", 
            func=tool_predict, 
            description="Predict defect risk for a file. Input: dict with 'file' (string) and 'features' (dict of metrics like loc, complexity, churn, num_devs)"
        ),
        Tool(
            name="generate_tests", 
            func=tool_generate, 
            description="Generate test code using AI. Input: dict with 'source' containing test requirements or specifications"
        ),
        Tool(
            name="run_tests", 
            func=tool_run, 
            description="Run tests in the test runner. Input: dict with 'tests' list of test file paths"
        ),
        Tool(
            name="self_heal", 
            func=tool_heal, 
            description="Attempt to self-heal failing selectors. Input: dict with 'error' (string) and optional 'dom' (string)"
        )
    ]

def initialize_agent_with_fallback():
    """Initialize agent with fallback to basic functionality if LLM or LangChain fails"""
    try:
        from langchain import OpenAI as LCOpenAI
        from langchain.agents import initialize_agent
        from langchain.agents.agent_types import AgentType
    except Exception as e:
        logger.warning(f"LangChain not available: {e}")
        return None

    llm = None
    try:
        if LLM_PROVIDER == 'openai' and OPENAI_KEY:
            llm = LCOpenAI(api_key=OPENAI_KEY, temperature=0, model=LLM_MODEL)
    except Exception as e:
        logger.error(f"Failed to init LLM: {e}")

    if llm:
        try:
            agent = initialize_agent(_get_tools(), llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
            logger.info("LangChain agent initialized successfully")
            return agent
        except Exception as e:
            logger.error(f"Failed to initialize LangChain agent: {e}")
    logger.warning("Using fallback agent without LLM")
    return None

# Initialize agent
agent = initialize_agent_with_fallback()

def run_agent(goal: str, context: Dict=None) -> str:
    """Run the LangChain agent with a free-form goal and optional context dict."""
    if context is None:
        context = {}
    
    if not agent:
        return "Agent not available. Install langchain and set OPENAI_API_KEY to enable."
    
    try:
        prompt = f"You are an automation testing assistant. Goal: {goal}. Context: {json.dumps(context)}"
        result = agent.run(prompt)
        
        # Save to memory
        try:
            from src.ai.memory.execution_memory import append_run
            append_run({'goal': goal, 'context': context, 'result': result})
        except Exception as e:
            logger.warning(f"Failed to save to memory: {e}")
        
        return result
    except Exception as e:
        error_msg = f"Agent execution failed: {str(e)}"
        logger.error(error_msg)
        return error_msg

if __name__ == '__main__':
    # simple CLI for local testing
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--goal', required=True)
    p.add_argument('--context', default='{}')
    args = p.parse_args()
    
    try:
        context = json.loads(args.context)
    except:
        context = {}
    
    print(run_agent(args.goal, context))
