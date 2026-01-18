"""
HachimiNetV1 Installation & Setup Validation Script
Checks all dependencies, configurations, and model integrity
"""

import sys
import os
import torch
import yaml
from pathlib import Path
from utils.paths import get_config_path, get_project_root

def check_imports():
    """Verify all required packages are installed"""
    print("\n" + "="*60)
    print("CHECKING IMPORTS")
    print("="*60)
    
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'yaml': 'PyYAML',
        'h5py': 'h5py',
        'nibabel': 'Nibabel',
        'nilearn': 'Nilearn',
        'sklearn': 'Scikit-learn',
        'tqdm': 'tqdm',
        'scipy': 'SciPy'
    }
    
    all_ok = True
    for module_name, friendly_name in required_packages.items():
        try:
            __import__(module_name)
            print(f"✓ {friendly_name:20s} OK")
        except ImportError:
            print(f"✗ {friendly_name:20s} MISSING")
            all_ok = False
    
    return all_ok


def check_directories():
    """Verify required directories exist"""
    print("\n" + "="*60)
    print("CHECKING DIRECTORIES")
    print("="*60)
    
    project_root = get_project_root()
    
    required_dirs = [
        'configs',
        'data',
        'data/H5',
        'models',
        'models/components',
        'processing',
        'training',
        'training/evaluator.py'  # Actually a file
    ]
    
    all_ok = True
    for d in required_dirs:
        path = project_root / d
        if d.endswith('.py'):
            exists = path.is_file()
        else:
            exists = path.is_dir()
        
        status = "✓" if exists else "✗"
        print(f"{status} {d}")
        if not exists:
            all_ok = False
    
    return all_ok


def check_config():
    """Validate configuration file"""
    print("\n" + "="*60)
    print("CHECKING CONFIGURATION")
    print("="*60)
    
    config_path = get_config_path()
    
    if not config_path.exists():
        print(f"✗ Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        print(f"✓ Config file loaded successfully")
        
        # Check required keys
        required_keys = {
            'paths': ['raw_fmri_dir', 'raw_eeg_dir', 'processed_dir', 'template_dir'],
            'data_specs': ['tr', 'voxel_resolution', 'n_ica_components', 'activation_threshold', 'eeg_history_sec', 'hrf_delay_sec']
        }
        
        for section, keys in required_keys.items():
            if section not in cfg:
                print(f"✗ Missing section: {section}")
                return False
            
            for key in keys:
                if key not in cfg[section]:
                    print(f"✗ Missing key: {section}.{key}")
                    return False
            
            print(f"✓ Section '{section}' complete")
        
        return True
    
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return False


def check_models():
    """Test model initialization"""
    print("\n" + "="*60)
    print("CHECKING MODELS")
    print("="*60)
    
    try:
        from models.classifier_net import PhysicsE2fNet
        from models.components.temporal import EEGEncoder
        from models.loss import CompoundLoss
        
        print("✓ Model imports successful")
        
        # Test EEGEncoder
        encoder = EEGEncoder(input_channels=64, time_len=300, n_components=64)
        test_eeg = torch.randn(2, 300, 64)
        weights = encoder(test_eeg)
        assert weights.shape == (2, 64), f"Encoder output shape mismatch: {weights.shape}"
        print(f"✓ EEGEncoder works: input {test_eeg.shape} -> output {weights.shape}")
        
        # Test CompoundLoss
        loss_fn = CompoundLoss(num_classes=3)
        test_logits = torch.randn(2, 3, 53, 63, 52)
        test_targets = torch.randint(0, 3, (2, 53, 63, 52))
        loss, details = loss_fn(test_logits, test_targets)
        print(f"✓ CompoundLoss works: loss={loss.item():.4f}")
        
        # Test PhysicsE2fNet (without loading basis to avoid file dependency)
        basis = torch.randn(64, 53, 63, 52)
        model = PhysicsE2fNet(n_ica_components=64, eeg_channels=64, eeg_time_len=300)
        model.spatial_basis = basis  # Override with test basis
        
        test_eeg = torch.randn(2, 300, 64)
        logits = model(test_eeg)
        assert logits.shape == (2, 3, 53, 63, 52), f"Model output shape mismatch: {logits.shape}"
        print(f"✓ PhysicsE2fNet works: input {test_eeg.shape} -> output {logits.shape}")
        
        return True
    
    except Exception as e:
        print(f"✗ Model check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_data_loader():
    """Test data loader (if data exists)"""
    print("\n" + "="*60)
    print("CHECKING DATA LOADER")
    print("="*60)
    
    try:
        from data.loaders import FMRIEEGDataset
        print("✓ DataLoader import successful")
        
        # Try to instantiate (will fail gracefully if no data)
        try:
            ds = FMRIEEGDataset(config_path=str(get_config_path()))
            if len(ds) == 0:
                print("⚠ DataLoader found no processed data (expected)")
                print("  Run: python processing/quantize_data.py")
            else:
                print(f"✓ DataLoader found {len(ds)} samples")
        except FileNotFoundError:
            print("⚠ Processed data directory empty (expected before first run)")
        
        return True
    
    except Exception as e:
        print(f"✗ DataLoader check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_gpu():
    """Check CUDA availability"""
    print("\n" + "="*60)
    print("CHECKING GPU")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  PyTorch Version: {torch.__version__}")
        return True
    else:
        print("⚠ CUDA not available - training will be slow (CPU mode)")
        print(f"  PyTorch Version: {torch.__version__}")
        return False


def main():
    """Run all checks"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*15 + "HachimiNetV1 Validation Script" + " "*13 + "║")
    print("╚" + "="*58 + "╝")
    
    results = {
        'Imports': check_imports(),
        'Directories': check_directories(),
        'Configuration': check_config(),
        'Models': check_models(),
        'DataLoader': check_data_loader(),
        'GPU': check_gpu()
    }
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for check_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} {check_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL CHECKS PASSED - Ready to train!")
        print("\nNext steps:")
        print("1. Prepare your data (fMRI NIfTI + EEG PT files)")
        print("2. Run: python processing/run_ica.py")
        print("3. Run: python processing/quantize_data.py")
        print("4. Run: python main.py")
    else:
        print("✗ SOME CHECKS FAILED - Fix issues above")
        print("\nTo install missing packages:")
        print("pip install torch nibabel nilearn scikit-learn tqdm pyyaml")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
