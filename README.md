# Conditional Attention-Driven Sketch Generation

**Short Description:**  
This Colab notebook implements a class-conditioned, attention-enhanced Seq2Seq LSTM model in PyTorch to generate vector stroke sketches from QuickDraw data. The workflow includes data extraction from NDJSON files, Δ-sequence preprocessing, coordinate standardization, custom learnable-weight loss, model training with early stopping, and animated visualization of generated sketches.

## Notebook Usage

1. **Mount Google Drive**  
   Mount your Drive to access dataset and save outputs:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Set Paths & Hyperparameters**  
   Edit the top section of the notebook to configure:
   ```python
   base_dir = "/content/drive/MyDrive/dl-3_data"        # NDJSON files
   model_saver_path = "/content/drive/MyDrive/output"   # Checkpoints & pickles
   image_saver_path = "/content/drive/MyDrive/images"   # Generated visuals
   max_records_per_file = 4000
   epoch = 40
   batch_size = 64
   hidden_size = 64
   class_emb_dim = 8
   learning_rate = 0.0001952
   ```

3. **Data Extraction & Visualization**  
   - Extract up to `max_records_per_file` sketches per category.  
   - Display sample sketches per class.

4. **Preprocessing**  
   - Convert absolute coordinates to Δx, Δy with pen flags.  
   - Save processed sequences (`processed_data.pkl`).  
   - Standardize Δx, Δy across dataset.  
   - Save scaler (`scaler.pkl`).

5. **Dataset & DataLoader**  
   - Load standardized data.  
   - Pad sequences and split into train/val/test.  
   - Wrap into `SketchDataset` and PyTorch `DataLoader`.

6. **Model Training**  
   - Define `ConditionalSketchModel` with Class Embedding, Attention, LSTM encoder–decoder.  
   - Use `LossWithLearnableWeights` combining stroke and pen losses.  
   - Train with early stopping based on validation loss.  
   - Save best model checkpoint.

7. **Inference & Visualization**  
   - Load trained model & scaler.  
   - Use reference sketches or dummy inputs to generate new strokes.  
   - Plot and save static images and animated GIFs of generated sketches.

## File Structure

This repository contains a single Jupyter notebook (`.ipynb`) converted into this Python script:

```
Conditional_Sketch_Generator.ipynb  # Full end-to-end pipeline in Colab
```

## Dependencies

- Python 3.7+
- PyTorch
- scikit-learn
- numpy, pandas, matplotlib, plotly
- PIL (Pillow)
- joblib

Install via:
```bash
pip install torch torchvision scikit-learn numpy pandas matplotlib plotly pillow joblib
```

## Acknowledgments

Based on Google’s QuickDraw dataset and inspiration from SketchRNN architectures (but fully custom class-conditioned and attention-enhanced).

## License

MIT License © Rishi Chhabra
