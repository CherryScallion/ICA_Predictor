# HachimiNetV1 - Setup Checklist

## ✅ Installation Checklist

- [ ] Python 3.7+ installed
- [ ] PyTorch installed (`pip install torch`)
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Run validation: `python validate_setup.py` (should pass all checks)

## ✅ Data Preparation Checklist

- [ ] fMRI data in NIfTI format (.nii.gz)
- [ ] EEG data in PyTorch format (.pt tensor)
- [ ] Temporal alignment verified (T_fmri ≈ T_eeg)
- [ ] Create directories:
  ```bash
  mkdir -p data/processed
  mkdir -p data/templates
  mkdir -p checkpoints
  ```

## ✅ Configuration Checklist

- [ ] Update `configs/config.yaml` with your paths
  - Set `raw_fmri_dir` to your fMRI location
  - Set `raw_eeg_dir` to your EEG location
  - Verify `voxel_resolution` matches your data
  
- [ ] Review data parameters:
  - `tr`: Your fMRI repetition time (typically 2.0s)
  - `n_ica_components`: Start with 64
  - `activation_threshold`: Default 1.0 (adjust based on your needs)
  - `eeg_history_sec`: Window length (default 6.0s)
  - `hrf_delay_sec`: HRF delay (default 4.0s)

## ✅ Processing Pipeline Checklist

### Step 1: Extract ICA Basis
- [ ] Run: `python processing/run_ica.py`
- [ ] Wait for completion (10-30 min depending on data size)
- [ ] Verify outputs:
  - `data/templates/ica_basis.pt` exists
  - `data/templates/gray_mask.pt` exists
  - Check file sizes are reasonable

### Step 2: Quantize Data
- [ ] Run: `python processing/quantize_data.py`
- [ ] Verify outputs in `data/processed/`:
  - `sub*_eeg.pt` files exist
  - `sub*_labels.pt` files exist
  - At least 10+ samples for training

## ✅ Training Checklist

- [ ] Confirm validation script passes: `python validate_setup.py`
- [ ] Review `configs/config.yaml` one more time
- [ ] Ensure GPU is available (or accept CPU training)
- [ ] Run training: `python main.py`
- [ ] Monitor:
  - Loss decreasing over time
  - No NaN/Inf values
  - Checkpoints saved to `checkpoints/`

## ✅ Expected Outputs

After successful training:
```
checkpoints/
├── model_ep5.pth      (First checkpoint)
├── model_ep10.pth
├── model_ep15.pth
├── ...
└── model_ep50.pth     (Final model)
```

## ✅ Troubleshooting Checklist

- [ ] If validation fails, check error message in validation report
- [ ] If data loading fails:
  - Verify file names match naming convention
  - Check tensor shapes are correct
  - Confirm processed directory has data

- [ ] If training fails:
  - Check GPU memory (reduce batch_size if needed)
  - Verify config paths are absolute or correct relative paths
  - Check basis file was loaded successfully

- [ ] If convergence is poor:
  - Increase `warmup_epochs` in trainer
  - Adjust `activation_threshold` in config
  - Review class distribution in labels

## 📞 Quick Commands

```bash
# Validate setup
python validate_setup.py

# Extract ICA basis
python processing/run_ica.py

# Quantize data
python processing/quantize_data.py

# Train model
python main.py

# View help
python main.py --help
```

## 🎯 Success Criteria

After completing this checklist:
- ✅ All validation checks pass
- ✅ Data processing completes without errors
- ✅ Training runs with decreasing loss
- ✅ Checkpoints saved successfully

**If all criteria met → Project is working correctly!**
