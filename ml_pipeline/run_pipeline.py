import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config_ml import (
    PROCESSED_DIR,
    TUNED_MODEL_DIR,
    REPORTS_DIR,
    LOGS_DIR,
)
from core.logger import setup_logger


class PipelineOrchestrator:

    def __init__(
        self,
        force_all: bool = False,
        skip_prep: bool = False,
        skip_tuning: bool = False,
        skip_eval: bool = False,
        n_trials: int = 100,
        verbose: bool = True
    ):
        self.force_all = force_all
        self.skip_prep = skip_prep
        self.skip_tuning = skip_tuning
        self.skip_eval = skip_eval
        self.n_trials = n_trials
        self.verbose = verbose
        
        # Setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = LOGS_DIR / f"pipeline_{timestamp}.log"
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger(
            "pipeline",
            log_file=log_path,
            level=20 if verbose else 30  # INFO or WARNING
        )
        
        self.results: Dict[str, Dict] = {}
        self.total_start_time = time.time()
        
    def _print_header(self, title: str) -> None:
        """Print formatted section header."""
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)
        self.logger.info(title)
        
    def _print_status(self, message: str, status: str = "INFO") -> None:
        """Print status message with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {
            "INFO": "ℹ️ ",
            "OK": "✅",
            "SKIP": "⏭️ ",
            "RUN": "🚀",
            "ERROR": "❌",
            "WARN": "⚠️ "
        }.get(status, "")
        
        print(f"[{timestamp}] {prefix} {message}")
        
        log_method = {
            "ERROR": self.logger.error,
            "WARN": self.logger.warning,
        }.get(status, self.logger.info)
        log_method(message)
    
    def _check_stage_completed(self, stage: str) -> Tuple[bool, str]:
        """
        Check if a pipeline stage has been completed.
        
        Args:
            stage: Stage name ('prep', 'tuning', 'eval')
        
        Returns:
            tuple: (is_completed, reason)
        """
        if stage == "prep":
            train_csv = PROCESSED_DIR / "train.csv"
            test_csv = PROCESSED_DIR / "test.csv"
            
            if train_csv.exists() and test_csv.exists():
                return True, f"Found {train_csv.name} and {test_csv.name}"
            return False, "Missing train.csv or test.csv"
        
        elif stage == "tuning":
            model_txt = TUNED_MODEL_DIR / "lgbm_tuned.txt"
            results_json = TUNED_MODEL_DIR / "evaluation_results.json"
            
            if model_txt.exists() and results_json.exists():
                return True, f"Found {model_txt.name} and {results_json.name}"
            return False, "Missing model or results files"
        
        elif stage == "eval":
            # Check for comprehensive testing outputs
            report_dir = REPORTS_DIR / "comprehensive_testing"
            if report_dir.exists():
                report_files = list(report_dir.glob("*.md")) + list(report_dir.glob("*.json"))
                if len(report_files) >= 2:
                    return True, f"Found {len(report_files)} report files"
            return False, "Missing evaluation reports"
        
        return False, "Unknown stage"
    
    def _run_command(
        self,
        command: list,
        stage_name: str,
        cwd: Optional[Path] = None
    ) -> bool:

        if cwd is None:
            cwd = Path(__file__).parent
        
        self._print_status(f"Running: {' '.join(command)}", "RUN")
        stage_start = time.time()
        
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True
            )
            
            stage_time = time.time() - stage_start
            
            # Log output if verbose
            if self.verbose and result.stdout:
                print(result.stdout)
            
            self._print_status(
                f"{stage_name} completed in {stage_time:.1f}s",
                "OK"
            )
            
            self.results[stage_name] = {
                "status": "success",
                "time_seconds": stage_time,
                "timestamp": datetime.now().isoformat()
            }
            
            return True
        
        except subprocess.CalledProcessError as e:
            stage_time = time.time() - stage_start
            
            self._print_status(
                f"{stage_name} FAILED after {stage_time:.1f}s",
                "ERROR"
            )
            
            print(f"\n❌ Error Output:\n{e.stderr}")
            self.logger.error(f"{stage_name} failed: {e.stderr}")
            
            self.results[stage_name] = {
                "status": "failed",
                "time_seconds": stage_time,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
            return False
    
    def run_stage_1_preparation(self) -> bool:

        self._print_header("STAGE 1: DATA PREPARATION")
        
        # Check if already completed
        if not self.force_all and not self.skip_prep:
            completed, reason = self._check_stage_completed("prep")
            if completed:
                self._print_status(f"Stage already completed: {reason}", "SKIP")
                self.results["preparation"] = {
                    "status": "skipped",
                    "reason": reason
                }
                return True
        
        if self.skip_prep:
            self._print_status("Stage skipped by user", "SKIP")
            self.results["preparation"] = {"status": "skipped", "reason": "User request"}
            return True
        
        # Run preparation
        return self._run_command(
            ["python", "training/prepare_dataset.py"],
            "Data Preparation"
        )
    
    def run_stage_2_tuning(self) -> bool:

        self._print_header("STAGE 2: HYPERPARAMETER TUNING")
        
        # Check dependencies
        train_csv = PROCESSED_DIR / "train.csv"
        if not train_csv.exists():
            self._print_status(
                "Cannot run tuning: train.csv not found. Run stage 1 first.",
                "ERROR"
            )
            return False
        
        # Check if already completed
        if not self.force_all and not self.skip_tuning:
            completed, reason = self._check_stage_completed("tuning")
            if completed:
                self._print_status(f"Stage already completed: {reason}", "SKIP")
                self.results["tuning"] = {
                    "status": "skipped",
                    "reason": reason
                }
                return True
        
        if self.skip_tuning:
            self._print_status("Stage skipped by user", "SKIP")
            self.results["tuning"] = {"status": "skipped", "reason": "User request"}
            return True
        
        # Run tuning
        self._print_status(f"Starting Optuna tuning with {self.n_trials} trials...", "INFO")
        self._print_status("This may take 5-10 minutes depending on your hardware", "INFO")
        
        # Note: tune_lgbm.py doesn't accept CLI args yet, always uses N_TRIALS=100
        # TODO: Add argparse to tune_lgbm.py to accept --n-trials parameter
        return self._run_command(
            ["python", "tuning/tune_lgbm.py"],
            "Hyperparameter Tuning"
        )
    
    def run_stage_3_evaluation(self) -> bool:

        self._print_header("STAGE 3: COMPREHENSIVE EVALUATION")
        
        # Check dependencies
        model_path = TUNED_MODEL_DIR / "lgbm_tuned.txt"
        if not model_path.exists():
            self._print_status(
                "Cannot run evaluation: model not found. Run stage 2 first.",
                "ERROR"
            )
            return False
        
        # Check if already completed
        if not self.force_all and not self.skip_eval:
            completed, reason = self._check_stage_completed("eval")
            if completed:
                self._print_status(f"Stage already completed: {reason}", "SKIP")
                self.results["evaluation"] = {
                    "status": "skipped",
                    "reason": reason
                }
                return True
        
        if self.skip_eval:
            self._print_status("Stage skipped by user", "SKIP")
            self.results["evaluation"] = {"status": "skipped", "reason": "User request"}
            return True
        
        # Run evaluation
        return self._run_command(
            ["python", "scripts/evaluation/comprehensive_testing.py"],
            "Comprehensive Evaluation"
        )
    
    def run_full_pipeline(self) -> bool:

        self._print_header("🚀 COMPLETE ML PIPELINE ORCHESTRATOR")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Configuration:")
        print(f"  Force re-run: {self.force_all}")
        print(f"  Optuna trials: {self.n_trials}")
        print(f"  Skip stages: prep={self.skip_prep}, tuning={self.skip_tuning}, eval={self.skip_eval}")
        
        # Run stages in sequence
        stages = [
            ("Stage 1", self.run_stage_1_preparation),
            ("Stage 2", self.run_stage_2_tuning),
            ("Stage 3", self.run_stage_3_evaluation),
        ]
        
        for stage_name, stage_func in stages:
            success = stage_func()
            if not success:
                self._print_status(f"{stage_name} failed. Pipeline halted.", "ERROR")
                self._print_summary(success=False)
                return False
        
        # All stages completed
        self._print_summary(success=True)
        return True
    
    def _print_summary(self, success: bool) -> None:
        """Print pipeline execution summary."""
        total_time = time.time() - self.total_start_time
        
        self._print_header("📊 PIPELINE SUMMARY")
        
        if success:
            print("✅ Pipeline completed successfully!")
        else:
            print("❌ Pipeline failed")
        
        print(f"\nTotal runtime: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print("\nStage Results:")
        
        for stage, result in self.results.items():
            status_icon = {
                "success": "✅",
                "failed": "❌",
                "skipped": "⏭️"
            }.get(result["status"], "❓")
            
            stage_time = result.get("time_seconds", 0)
            print(f"  {status_icon} {stage:20s}: {result['status']:8s} ({stage_time:.1f}s)")
        
        print("\n" + "=" * 70)
        
        if success:
            print("\n📁 Output Locations:")
            print(f"  Processed data:  {PROCESSED_DIR}")
            print(f"  Trained model:   {TUNED_MODEL_DIR}")
            print(f"  Reports:         {REPORTS_DIR}")
            print(f"  Logs:            {LOGS_DIR}")
            print("\n🎉 All results saved! Check the reports directory for visualizations.")


def main():
    """Main entry point for pipeline orchestrator."""
    parser = argparse.ArgumentParser(
        description="Complete ML Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python run_pipeline.py
  
  # Force re-run everything
  python run_pipeline.py --force-all
  
  # Skip data preparation (use existing processed data)
  python run_pipeline.py --skip-prep
  
  # Quick test with fewer trials
  python run_pipeline.py --n-trials 20
  
  # Re-run only evaluation
  python run_pipeline.py --skip-prep --skip-tuning --force-all
        """
    )
    
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Force re-run all stages even if completed"
    )
    
    parser.add_argument(
        "--skip-prep",
        action="store_true",
        help="Skip data preparation stage"
    )
    
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip hyperparameter tuning stage"
    )
    
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation stage"
    )
    
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials (default: 100)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator(
        force_all=args.force_all,
        skip_prep=args.skip_prep,
        skip_tuning=args.skip_tuning,
        skip_eval=args.skip_eval,
        n_trials=args.n_trials,
        verbose=not args.quiet
    )
    
    # Run pipeline
    success = orchestrator.run_full_pipeline()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
