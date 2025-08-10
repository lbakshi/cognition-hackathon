"""
Modal Integration for FastAPI Server
This module provides integration between the FastAPI server and Modal execution service.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modal_execution import modal_service
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ModalIntegration:
    """Integration layer between server endpoints and Modal execution"""
    
    def __init__(self):
        self.service = modal_service
    
    async def execute_experiment_from_api(
        self,
        plan_id: str,
        code_files: List[Dict[str, str]],
        spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute experiment from API request (for /execute endpoint)
        
        Args:
            plan_id: The plan ID from /plan endpoint
            code_files: List of generated code files from /codegen
            spec: Experiment specification
            
        Returns:
            Dict with job_id and status
        """
        # Convert code_files format to dict
        experiment_files = {}
        for file_info in code_files:
            path = file_info.get('path', '')
            content = file_info.get('content', '')
            
            # Extract filename from path
            filename = path.split('/')[-1] if '/' in path else path
            experiment_files[filename] = content
        
        # Submit to Modal
        job_id = await self.service.submit_experiment(
            experiment_files=experiment_files,
            experiment_spec=spec,
            experiment_id=plan_id
        )
        
        return {
            'job_id': job_id,
            'status': 'running'
        }
    
    def get_job_status_for_api(self, job_id: str) -> Dict[str, Any]:
        """
        Get job status for API response
        
        Args:
            job_id: Job ID to check
            
        Returns:
            Job status information
        """
        status = self.service.get_job_status(job_id)
        
        if not status:
            return {'error': 'Job not found'}
        
        # Convert to API format
        api_status = {
            'job_id': status['job_id'],
            'status': status['status'],
            'submitted_at': status['submitted_at']
        }
        
        if 'completed_at' in status:
            api_status['completed_at'] = status['completed_at']
        
        if status.get('error'):
            api_status['error'] = status['error']
        
        if status.get('results'):
            api_status['metrics'] = status['results']
        
        return api_status
    
    async def get_report_data(self, job_id: str) -> Dict[str, Any]:
        """
        Get data for report generation (for /report endpoint)
        
        Args:
            job_id: Job ID to get report data for
            
        Returns:
            Report data including metrics and artifacts
        """
        status = self.service.get_job_status(job_id)
        
        if not status:
            return {'error': 'Job not found'}
        
        if status.get('status') != 'success':
            return {'error': 'Job not completed successfully'}
        
        # Get artifacts
        artifacts = await self.service.get_job_artifacts(job_id)
        
        # Format for report generation
        report_data = {
            'job_id': job_id,
            'metrics': status.get('results', {}),
            'status': status['status'],
            'artifacts': artifacts
        }
        
        # Extract key metrics for LLM summary
        metrics_summary = {}
        if 'results' in status and status['results']:
            for model_name, model_results in status['results'].items():
                if isinstance(model_results, dict):
                    metrics_summary[model_name] = {
                        'accuracy': model_results.get('accuracy', 0),
                        'f1_score': model_results.get('f1_score', 0),
                        'loss': model_results.get('loss', 0),
                        'training_time': model_results.get('training_time', 0)
                    }
        
        report_data['metrics_summary'] = metrics_summary
        
        return report_data
    
    def list_all_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs for debugging/monitoring"""
        return self.service.list_jobs()

# Global integration instance for server
modal_integration = ModalIntegration()

# Convenience functions for server endpoints
async def execute_experiment(plan_id: str, code_files: List[Dict[str, str]], spec: Dict[str, Any]):
    """Execute experiment (for /execute endpoint)"""
    return await modal_integration.execute_experiment_from_api(plan_id, code_files, spec)

def get_job_status(job_id: str):
    """Get job status (for polling)"""
    return modal_integration.get_job_status_for_api(job_id)

async def get_report_data(job_id: str):
    """Get report data (for /report endpoint)"""
    return await modal_integration.get_report_data(job_id)