"""
Automated dental scan processing monitor.
If it finds raw data in patient folders:
1) Applies scanTransformMatrix (found in the Json file) to the mesh;
2) Rotates the scan by 180 degrees around Y axis;
3) Rotates the scan by 90 degrees around X axis;
4) Shifts the scan towards the center of mass and saves the offset in a file;
Automatically runs segmentation and auto bonding.
Tracks individual files to process new files added to existing patient folders.
Usage:
    python application/monitor.py \
        --data-root /workspace/application/data/ \
        --seg-config /workspace/application/configs/Pt_semseg_app.py \
        --seg-weight /workspace/application/weights/segmentator_best.pth \
        --bond-config /workspace/application/configs/Pt_regressor_app.py \
        --bond-weight /workspace/application/weights/regressor_best.pth \
        --check-interval 10
"""
import pointcept
import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Set, List
from preprocessor import Preprocessor
import debugpy

# NOTE: This script requires 'trimesh' and 'numpy'.
# Install them with: pip install trimesh numpy
try:
    import trimesh
    import numpy as np
except ImportError:
    print("Error: 'trimesh' and 'numpy' are required. Please install them using 'pip install trimesh numpy'")
    sys.exit(1)


class ScanMonitor:
    def __init__(
        self,
        data_root: Path,
        seg_config: Path,
        seg_weight: Path,
        bond_config: Path,
        bond_weight: Path,
        check_interval: int = 10,
        status_file: str = "processing_status.json",
        no_visuals: bool = False
    ):
        self.data_root = Path(data_root)
        self.seg_config = Path(seg_config)
        self.seg_weight = Path(seg_weight)
        self.bond_config = Path(bond_config)
        self.bond_weight = Path(bond_weight)
        self.check_interval = check_interval
        self.status_file = self.data_root / status_file  # Status file inside data_root
        self.no_visuals = no_visuals
        self.prep = Preprocessor() 

        # Validate paths
        if not self.data_root.exists():
            raise ValueError(f"Data root does not exist: {self.data_root}")
        if not self.seg_config.exists():
            raise ValueError(f"Segmentation config does not exist: {self.seg_config}")
        if not self.seg_weight.exists():
            raise ValueError(f"Segmentation weight does not exist: {self.seg_weight}")
        if not self.bond_config.exists():
            raise ValueError(f"Bond config does not exist: {self.bond_config}")
        if not self.bond_weight.exists():
            raise ValueError(f"Bond weight does not exist: {self.bond_weight}")
        
        self.status = self.load_status()
        print(f"✅ Monitor initialized")
        print(f"   Data root: {self.data_root}")
        print(f"   Check interval: {self.check_interval}s")
        print(f"   Status file: {self.status_file}")
    
    def load_status(self) -> Dict:
        """
        Load processing status from JSON file.
        This function is called at the beginning of each
        iteration in the main processing loop, so that
        any manual edits of the file will be detected.
        """
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Error loading status file: {e}")
                # fallback to old status if a missedit happens (manual for exaxmple)
                if self.status: return self.status
        return {}
    
    def save_status(self):
        """Save processing status to JSON file."""
        try:
            with open(self.status_file, 'w') as f:
                json.dump(self.status, f, indent=4)
        except Exception as e:
            print(f"⚠️  Error saving status file: {e}")
    
    def find_patient_directories(self) -> Set[str]:
        """Find all patient directories that contain a 'raw_data' subdirectory or STL files."""
        patient_dirs = set()
        
        for item in self.data_root.iterdir():
            if item.is_dir() and item.name != "__pycache__":
                # A patient dir is one that contains raw data to be processed,
                # or already processed STL files for the next pipeline steps.
                has_raw_data = (item / "raw_data").is_dir()
                has_stl_files = any(item.glob("*.stl"))
                
                if has_raw_data or has_stl_files:
                    patient_dirs.add(item.name)
        
        return patient_dirs
    
    def get_stl_files(self, patient_dir: Path) -> List[str]:
        """Get list of STL files in patient directory."""
        stl_files = list(patient_dir.glob("*.stl"))
        return sorted([f.name for f in stl_files])
    
    def get_unprocessed_files(self, patient_id: str, patient_dir: Path) -> List[str]:
        """Get list of STL files that haven't been processed yet."""
        current_files = set(self.get_stl_files(patient_dir))
        
        if patient_id not in self.status:
            return list(current_files)
        
        processed_files = set(self.status[patient_id].get("processed_files", []))
        failed_files = set(self.status[patient_id].get("failed_files", []))
        
        # Return files that are neither processed nor failed
        unprocessed = current_files - processed_files - failed_files
        return sorted(list(unprocessed))
    
    def get_scan_info(self, files: List[str]) -> Dict:
        """Get information about scan files."""
        has_lower = any("lower" in f.lower() for f in files)
        has_upper = any("upper" in f.lower() for f in files)
        
        return {
            "num_files": len(files),
            "has_lower": has_lower,
            "has_upper": has_upper,
            "files": files
        }
    
    def has_new_raw_files(self, patient_dir: Path) -> bool:
        """Check if there are new, unprocessed files in the raw_data directory."""
        raw_data_dir = patient_dir / "raw_data"
        if not raw_data_dir.is_dir():
            return False

        # Case-insensitive glob for STL files
        raw_stls = list(raw_data_dir.glob('[sS][tT][eE][mM]_*.[sS][tT][lL]'))
        for raw_stl in raw_stls:
            # If the processed file doesn't exist in the parent directory, it's new.
            if not (patient_dir / raw_stl.name).exists():
                return True
        return False

    def should_process(self, patient_id: str, patient_dir: Path) -> bool:
        """Check if patient directory has new raw files or unprocessed processed files."""
        # Check 1: Are there new raw files to preprocess?
        if self.has_new_raw_files(patient_dir):
            return True
        
        # Check 2: Are there processed files waiting for segmentation/bonding?
        unprocessed_for_pipeline = self.get_unprocessed_files(patient_id, patient_dir)
        return len(unprocessed_for_pipeline) > 0
    
    def run_segmentation(self, patient_id: str, patient_dir: Path) -> bool:
        """Run segmentation script for patient directory."""
        print(f"\n{'='*80}")
        print(f"🦷 Running SEGMENTATION for patient {patient_id}")
        print(f"{'='*80}")

        cmd = [
            "xvfb-run", "-a",
            sys.executable,
            "application/segment_scan.py",
            "--config-file", str(self.seg_config),
            "--options",
            f"data_folder={patient_dir}",
            f"weight={self.seg_weight}"
        ]

        if self.no_visuals:
            cmd.append("--no-visuals")
        
        try:
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.data_root.parent.parent)  # Points to /workspace
            result = subprocess.run(
                cmd,
                env=env,
                cwd=str(self.data_root.parent.parent),  # Run from /workspace
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            print(f"✅ Segmentation completed for {patient_id}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Segmentation failed for {patient_id}")
            print(f"   Error: {e}")
            if e.stdout:
                print("STDOUT:", e.stdout)
            if e.stderr:
                print("STDERR:", e.stderr)
            return False
    
    def run_bond_prediction(self, patient_id: str, patient_dir: Path) -> bool:
        """Run bond prediction script for patient directory."""
        print(f"\n{'='*80}")
        print(f"📍 Running BOND PREDICTION for patient {patient_id}")
        print(f"{'='*80}")
        
        cmd = [
            sys.executable,
            "bond.py",
            "--config-file", str(self.bond_config),
            "--options",
            f"data_folder={patient_dir}",
            f"weight={self.bond_weight}"
        ]
        if self.no_visuals:
            cmd.append("--no-visuals")
        
        try:
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.data_root.parent.parent)
            result = subprocess.run(
                cmd,
                env=env,
                cwd=self.data_root.parent,  # Run from project root
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            print(f"✅ Bond prediction completed for {patient_id}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Bond prediction failed for {patient_id}")
            print(f"   Error: {e}")
            if e.stdout:
                print("STDOUT:", e.stdout)
            if e.stderr:
                print("STDERR:", e.stderr)
            return False
    
    def process_patient(self, patient_id: str, patient_dir: Path):
        """Process a patient through the full pipeline: pre-processing, segmentation, bonding."""
        timestamp = datetime.now().isoformat()
        
        # Initialize patient status if not exists
        if patient_id not in self.status:
            self.status[patient_id] = {
                "processed_files": [],
                "failed_files": [],
                "processing_history": []
            }
        
        # --- Stage 1: Pre-processing from raw_data ---
        raw_data_dir = patient_dir / "raw_data"
        if raw_data_dir.is_dir():
            # Case-insensitive glob for STL files
            raw_stls = list(raw_data_dir.glob('[sS][tT][eE][mM]_*.[sS][tT][lL]'))            
            preprocess_success, raw_files_handled = self.prep.preprocess_raw_scans(patient_id, patient_dir) 
            if not preprocess_success:
                print(f"❌ Pre-processing failed for {patient_id}. Aborting.")
                processing_entry = {
                    "started_at": timestamp,
                    "files_to_process": raw_files_handled,
                    "status": "failed",
                    "failed_at": datetime.now().isoformat(),
                    "error": "Pre-processing raw scans failed"
                }
                self.status[patient_id]["processing_history"].append(processing_entry)
                self.status[patient_id]["failed_files"].extend(raw_files_handled)
                self.status[patient_id]["failed_files"] = list(set(self.status[patient_id]["failed_files"]))
                self.save_status()
                return

        # --- Stage 2: Segmentation and Bond Prediction ---
        # Get unprocessed files (e.g., upper.stl, lower.stl that were just created)
        unprocessed_files = self.get_unprocessed_files(patient_id, patient_dir)
        
        if not unprocessed_files:
            print(f"  No new files for segmentation/bonding for {patient_id}")
            return
        
        scan_info = self.get_scan_info(unprocessed_files)
        
        # Add processing entry to history
        processing_entry = {
            "started_at": timestamp,
            "files_to_process": unprocessed_files,
            "num_files": scan_info["num_files"],
            "has_lower": scan_info["has_lower"],
            "has_upper": scan_info["has_upper"]
        }
        
        print(f"\n{'#'*80}")
        print(f"# Processing Patient: {patient_id}")
        print(f"# Started at: {timestamp}")
        print(f"# New files to process: {unprocessed_files}")
        print(f"{'#'*80}")
        
        # Run segmentation
        seg_success = self.run_segmentation(patient_id, patient_dir)
        
        if not seg_success:
            processing_entry["status"] = "failed"
            processing_entry["failed_at"] = datetime.now().isoformat()
            processing_entry["error"] = "Segmentation failed"
            self.status[patient_id]["processing_history"].append(processing_entry)
            # Mark files as failed
            self.status[patient_id]["failed_files"].extend(unprocessed_files)
            self.status[patient_id]["failed_files"] = list(set(self.status[patient_id]["failed_files"]))
            self.save_status()
            return
        
        # Run bond prediction
        bond_success = self.run_bond_prediction(patient_id, patient_dir)
        
        if not bond_success:
            processing_entry["status"] = "failed"
            processing_entry["failed_at"] = datetime.now().isoformat()
            processing_entry["error"] = "Bond prediction failed"
            self.status[patient_id]["processing_history"].append(processing_entry)
            # Mark files as failed
            self.status[patient_id]["failed_files"].extend(unprocessed_files)
            self.status[patient_id]["failed_files"] = list(set(self.status[patient_id]["failed_files"]))
            self.save_status()
            return
        
        # Mark files as successfully processed
        processing_entry["status"] = "completed"
        processing_entry["completed_at"] = datetime.now().isoformat()
        self.status[patient_id]["processing_history"].append(processing_entry)
        self.status[patient_id]["processed_files"].extend(unprocessed_files)
        self.status[patient_id]["processed_files"] = list(set(self.status[patient_id]["processed_files"]))
        self.save_status()
        
        print(f"\n{'='*80}")
        print(f"✅ PIPELINE COMPLETED for patient {patient_id}")
        print(f"   Processed files: {unprocessed_files}")
        print(f"   Total processed files: {len(self.status[patient_id]['processed_files'])}")
        print(f"{'='*80}\n")
    
    def run(self):
        """Main monitoring loop."""
        print(f"\n{'='*80}")
        print(f"🔍 Starting Dental Scan Monitor")
        print(f"{'='*80}\n") 
        iteration = 0 
        try:
            while True:
                iteration += 1
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
                print(f"\n[{current_time}] Check #{iteration}") 
                # Find all patient directories
                print(f"\n[{current_time}] Loading status file...", flush=True)
                self.status = self.load_status()
                patient_dirs = self.find_patient_directories() 
                if not patient_dirs:
                    print("  No patient directories found")
                else:
                    print(f"  Found {len(patient_dirs)} patient directories") 
                # Process new/unprocessed files
                processed_any = False
                for patient_id in sorted(patient_dirs):
                    patient_dir = self.data_root / patient_id 
                    if self.should_process(patient_id, patient_dir):
                        print(f"  Processing {patient_id}: Found new files or tasks.")
                        processed_any = True
                        self.process_patient(patient_id, patient_dir)
                    else:
                        if patient_id in self.status:
                            processed_count = len(self.status[patient_id].get("processed_files", []))
                            failed_count = len(self.status[patient_id].get("failed_files", []))
                            print(f"  Skipping {patient_id} (processed: {processed_count}, failed: {failed_count})")
                        else:
                            print(f"  Skipping {patient_id} (no files)")
                
                if not processed_any:
                    print(f"  No new files to process. Waiting {self.check_interval}s...")
                
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            total_processed = sum(len(v.get("processed_files", [])) for v in self.status.values())
            total_failed = sum(len(v.get("failed_files", [])) for v in self.status.values())
            print("\n\n⚠️  Monitor stopped by user")
            print(f"Total processed files: {total_processed}")
            print(f"Total failed files: {total_failed}")
            print(f"Patients tracked: {len(self.status)}")
        except Exception as e:
            print(f"\n\n❌ Monitor error: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Monitor and automatically process dental scan directories"
    )
    parser.add_argument("--data-root", type=str, required=True, help="Root directory containing patient folders")
    parser.add_argument("--seg-config",type=str, required=True, help="Path to segmentation config file")
    parser.add_argument("--seg-weight",type=str, required=True,help="Path to segmentation model weights")
    parser.add_argument("--bond-config",type=str,required=True,help="Path to bond prediction config file")
    parser.add_argument("--bond-weight",type=str,required=True,help="Path to bond prediction model weights")
    parser.add_argument("--check-interval",type=int,default=10,help="Interval in seconds between checks (default: 10)")
    parser.add_argument("--status-file",type=str,default="processing_status.json",help="Name of status file (default: processing_status.json)")
    parser.add_argument("--prod", action="store_true", help="Run in production mode (no visuals)") 
    parser.add_argument("--debug", action="store_true", help="Wait for debugger to attach")
    args = parser.parse_args()
    if args.debug:
        print("Hello, happy debugging.")
        debugpy.listen(("0.0.0.0", 5681))
        print(">>> Debugger is listening on port 5681. Waiting for client to attach...")
        debugpy.wait_for_client()
        print(">>> Debugger attached. Resuming execution.")
    monitor = ScanMonitor(
        data_root=args.data_root,
        seg_config=args.seg_config,
        seg_weight=args.seg_weight,
        bond_config=args.bond_config,
        bond_weight=args.bond_weight,
        check_interval=args.check_interval,
        status_file=args.status_file,
        no_visuals=args.prod
    ) 
    monitor.run()

if __name__ == "__main__":
    main()
