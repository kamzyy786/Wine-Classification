"""Reproduce Milestone 1 data checks and leakage-safe feature processing."""
from pathlib import Path
import argparse, json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, default=Path('data/raw/wine.csv'))
    parser.add_argument('--out', type=Path, default=Path('outputs'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.raw)
    X = df.drop(columns='target')
    y = df.target
    assert df.shape == (178, 14), 'Unexpected source shape'
    assert set(y) == {0, 1, 2}, 'Unexpected labels'
    assert all(pd.api.types.is_numeric_dtype(X[c]) for c in X)
    assert np.isfinite(X.to_numpy()).all(), 'Review invalid values before processing'
    stats = X.describe().T
    stats.to_csv(args.out / 'numeric_summary.csv')
    q1, q3 = X.quantile(.25), X.quantile(.75)
    flags = ((X < q1 - 1.5*(q3-q1)) | (X > q3 + 1.5*(q3-q1)))
    quality = pd.DataFrame({'missing':df.isna().sum(), 'iqr_flags':flags.sum().reindex(df.columns)})
    quality.to_csv(args.out / 'quality_checks.csv')
    # Duplicate checks precede splitting, so repeated records cannot cross sets.
    assert not X.duplicated().any(), 'Group or remove repeated measurements before splitting'
    ids = np.arange(len(df))
    train, rest = train_test_split(ids, test_size=.30, stratify=y, random_state=42)
    val, test = train_test_split(rest, test_size=.50, stratify=y.iloc[rest], random_state=42)
    splits = {'train':train, 'validation':val, 'test':test}
    # Only training samples determine imputation and scaling parameters.
    imputer = SimpleImputer(strategy='median').fit(X.iloc[train])
    scaler = StandardScaler().fit(imputer.transform(X.iloc[train]))
    arrays = {}
    split_counts = {}
    for name, idx in splits.items():
        values = scaler.transform(imputer.transform(X.iloc[idx])).astype('float32')
        labels = y.iloc[idx].to_numpy(dtype='int64')
        arrays[name+'_X'], arrays[name+'_y'] = values, labels
        pd.DataFrame({'row_id':idx, 'target':labels}).to_csv(args.out / (name+'_rows.csv'), index=False)
        split_counts[name] = {str(k):int(v) for k,v in y.iloc[idx].value_counts().sort_index().items()}
    np.savez(args.out / 'processed_arrays.npz', **arrays)
    # PyTorch tensors are the inputs for a later classification model.
    import torch
    torch.save({k:torch.from_numpy(v) for k,v in arrays.items()}, args.out / 'processed_tensors.pt')
    parameters = {'features':list(X.columns), 'median':imputer.statistics_.tolist(),
                  'mean':scaler.mean_.tolist(), 'scale':scaler.scale_.tolist()}
    (args.out / 'preprocessing.json').write_text(json.dumps(parameters, indent=2))
    summary = {'n_rows':len(df), 'duplicates_full':int(df.duplicated().sum()),
               'duplicates_features':int(X.duplicated().sum()),
               'nonpositive_cells':int((X <= 0).sum().sum()),
               'class_counts':{str(k):int(v) for k,v in y.value_counts().sort_index().items()},
               'split_counts':split_counts, 'iqr_flagged_rows':int(flags.any(axis=1).sum()),
               'raw_sha256':hashlib.sha256(args.raw.read_bytes()).hexdigest()}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    y.value_counts().sort_index().plot.bar(ax=axes[0], color='#426c86', rot=0)
    axes[0].set(title='Cultivar classes', xlabel='Encoded label', ylabel='Samples')
    axes[1].hist(X.alcohol, bins=12, color='#426c86', edgecolor='white')
    axes[1].set(title='Alcohol distribution', xlabel='Source alcohol value', ylabel='Samples')
    axes[2].hist(X.proline, bins=12, color='#426c86', edgecolor='white')
    axes[2].set(title='Proline distribution', xlabel='Source proline value', ylabel='Samples')
    fig.tight_layout(); fig.savefig(args.out / 'distributions.png', dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(X.corr(), vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_xticks(range(13), X.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(13), X.columns, fontsize=7)
    fig.colorbar(im, ax=ax, label='Pearson correlation')
    fig.tight_layout(); fig.savefig(args.out / 'correlations.png', dpi=160); plt.close(fig)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
