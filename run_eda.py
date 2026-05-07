#!/usr/bin/env python3
"""
Zero-Trust Deepfake Voice Defense — Comprehensive EDA
======================================================
Runs full exploratory data analysis on ALL datasets present on the HPC:
  - data/release_in_the_wild/  (genuine/ + fake/)
  - data/LA/                   (ASVspoof 2019 LA)
  - data/synthetic/            (gtts, gtts_batch2, elevenlabs, etc.)
  - data/balanced/             (balanced dataset)
  - data/custom/               (custom genuine + synthetic)

Produces 10 publication-quality plots saved to logs/eda_*/
"""

import os, sys, random, json, warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import soundfile as sf
import librosa
import librosa.display

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO     = Path.home() / 'Zero-Trust-Deepfake-Voice-Defense'
DATA     = REPO / 'data'
LOG_DIR  = REPO / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)

NAVY  = '#1F3864'
BLUE  = '#2E75B6'
LIGHT = '#BDD7EE'
GREEN = '#1B5E20'
RED   = '#B71C1C'
WHITE = 'white'

AUDIO_EXTS = {'.wav', '.flac', '.mp3', '.ogg', '.m4a'}

# ═════════════════════════════════════════════════════════════════════════════
# DATASET DISCOVERY
# ═════════════════════════════════════════════════════════════════════════════

def find_audio(directory):
    """Recursively find all audio files in a directory."""
    files = []
    d = Path(directory)
    if not d.exists():
        return files
    for f in d.rglob('*'):
        if f.suffix.lower() in AUDIO_EXTS:
            files.append(str(f))
    return sorted(files)


def discover_datasets():
    """
    Discover all datasets on the HPC and classify files as genuine or synthetic.
    Returns: dict of {dataset_name: {'genuine': [...], 'synthetic': [...]}}
    """
    datasets = {}

    # ── 1. In-The-Wild ────────────────────────────────────────────────────────
    itw = DATA / 'release_in_the_wild'
    if itw.exists():
        g = find_audio(itw / 'genuine')
        s = find_audio(itw / 'fake')
        if not s:
            s = find_audio(itw / 'spoof')
        if g or s:
            datasets['In-The-Wild'] = {'genuine': g, 'synthetic': s}

    # ── 2. ASVspoof 2019 LA ────────────────────────────────────────────────────
    la = DATA / 'LA'
    if la.exists():
        # Try to read protocol files for labels
        proto_dir = la / 'ASVspoof2019_LA_cm_protocols'
        if not proto_dir.exists():
            proto_dir = la / 'ASVspoof2019_LA_asv_protocols'

        genuine_la, spoof_la = [], []
        if proto_dir.exists():
            for proto_file in proto_dir.glob('*.txt'):
                try:
                    with open(proto_file) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                utt_id = parts[1]
                                label  = parts[4]
                                # Search for the file
                                for split in ['train', 'dev', 'eval']:
                                    for ext in ['.flac', '.wav']:
                                        fp = la / f'ASVspoof2019_LA_{split}' / 'flac' / f'{utt_id}{ext}'
                                        if fp.exists():
                                            if label == 'bonafide':
                                                genuine_la.append(str(fp))
                                            else:
                                                spoof_la.append(str(fp))
                                            break
                except Exception:
                    pass

        # Fallback: scan all flac files
        if not genuine_la and not spoof_la:
            all_la = find_audio(la)
            genuine_la = all_la  # treat as genuine if no protocol

        if genuine_la or spoof_la:
            datasets['ASVspoof 2019 LA'] = {'genuine': genuine_la, 'synthetic': spoof_la}

    # ── 3. Synthetic folder ────────────────────────────────────────────────────
    synth = DATA / 'synthetic'
    if synth.exists():
        all_synth = []
        sources = {}
        for subdir in sorted(synth.iterdir()):
            if subdir.is_dir():
                files = find_audio(subdir)
                if files:
                    sources[subdir.name] = files
                    all_synth.extend(files)
        if all_synth:
            datasets['Synthetic (TTS)'] = {'genuine': [], 'synthetic': all_synth,
                                            'sources': sources}

    # ── 4. Balanced dataset ────────────────────────────────────────────────────
    balanced = DATA / 'balanced'
    if balanced.exists():
        bg = find_audio(balanced / 'genuine')
        bs = find_audio(balanced / 'synthetic')
        if not bg:
            bg = find_audio(balanced / 'real')
        if bg or bs:
            datasets['Balanced'] = {'genuine': bg, 'synthetic': bs}

    # ── 5. Custom dataset ──────────────────────────────────────────────────────
    custom = DATA / 'custom'
    if custom.exists():
        cg = find_audio(custom / 'genuine')
        cs = find_audio(custom / 'synthetic')
        if not cg:
            cg = find_audio(custom / 'real')
        if cg or cs:
            datasets['Custom'] = {'genuine': cg, 'synthetic': cs}

    # ── 6. Any other folders ───────────────────────────────────────────────────
    known = {'release_in_the_wild', 'LA', 'synthetic', 'balanced', 'custom', 'new_datasets'}
    for folder in sorted(DATA.iterdir()):
        if folder.is_dir() and folder.name not in known:
            files = find_audio(folder)
            if len(files) > 10:
                datasets[f'Other: {folder.name}'] = {
                    'genuine': files, 'synthetic': []
                }

    return datasets


# ═════════════════════════════════════════════════════════════════════════════
# AUDIO UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def load_wav(path, target_sr=16000):
    wav, sr = sf.read(path, dtype='float32', always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav, target_sr


def safe_info(path):
    try:
        info = sf.info(path)
        return info.duration, info.samplerate, info.channels
    except Exception:
        return None, None, None


def extract_features(files, n=150):
    """Extract audio features from a sample of files."""
    durations, centroids, zcrs, rms_vals, mfccs = [], [], [], [], []
    sample = random.sample(files, min(n, len(files)))
    for fp in sample:
        try:
            dur, sr, _ = safe_info(fp)
            if dur:
                durations.append(dur)
            wav, sr = load_wav(fp)
            centroids.append(float(librosa.feature.spectral_centroid(y=wav, sr=sr).mean()))
            zcrs.append(float(librosa.feature.zero_crossing_rate(wav).mean()))
            rms_vals.append(float(librosa.feature.rms(y=wav).mean()))
            mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=40).mean(axis=1)
            mfccs.append(mfcc)
        except Exception:
            pass
    return {
        'durations':  np.array(durations),
        'centroids':  np.array(centroids),
        'zcrs':       np.array(zcrs),
        'rms':        np.array(rms_vals),
        'mfccs':      np.array(mfccs) if mfccs else np.zeros((1, 40)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Dataset Overview (all datasets, counts)
# ═════════════════════════════════════════════════════════════════════════════

def plot_dataset_overview(datasets):
    print("  Generating Plot 1: Dataset Overview...")
    ds_names, g_counts, s_counts = [], [], []
    for name, info in datasets.items():
        g = len(info.get('genuine', []))
        s = len(info.get('synthetic', []))
        if g + s > 0:
            ds_names.append(name)
            g_counts.append(g)
            s_counts.append(s)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(WHITE)
    fig.suptitle('Dataset Overview — All Datasets on HPC',
                 fontsize=14, fontweight='bold', color=NAVY)

    # Stacked bar
    x = np.arange(len(ds_names))
    w = 0.6
    b1 = axes[0].bar(x, g_counts, w, label='Genuine', color=BLUE, edgecolor='white')
    b2 = axes[0].bar(x, s_counts, w, bottom=g_counts, label='Synthetic',
                     color=RED, edgecolor='white', alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ds_names, rotation=25, ha='right', fontsize=9)
    axes[0].set_ylabel('Number of Files')
    axes[0].set_title('File Counts per Dataset', fontweight='bold', color=NAVY)
    axes[0].legend()
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    for bar, val in zip(b1, g_counts):
        if val > 0:
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()/2,
                         f'{val:,}', ha='center', va='center',
                         fontsize=8, color='white', fontweight='bold')
    for bar, gv, sv in zip(b2, g_counts, s_counts):
        if sv > 0:
            axes[0].text(bar.get_x()+bar.get_width()/2, gv+sv/2,
                         f'{sv:,}', ha='center', va='center',
                         fontsize=8, color='white', fontweight='bold')

    # Pie chart — total genuine vs synthetic across all datasets
    total_g = sum(g_counts)
    total_s = sum(s_counts)
    total   = total_g + total_s
    wedges, texts, autotexts = axes[1].pie(
        [total_g, total_s],
        labels=[f'Genuine\n{total_g:,}', f'Synthetic\n{total_s:,}'],
        colors=[BLUE, RED],
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    for t in autotexts:
        t.set_fontsize(12)
        t.set_fontweight('bold')
        t.set_color('white')
    axes[1].set_title(f'Overall Class Balance\n(Total: {total:,} files)',
                      fontweight='bold', color=NAVY)

    plt.tight_layout()
    path = str(LOG_DIR / 'eda_01_dataset_overview.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print(f"    Saved: {path}")
    return path


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Duration Distribution (per dataset)
# ═════════════════════════════════════════════════════════════════════════════

def plot_duration_distribution(datasets):
    print("  Generating Plot 2: Duration Distribution...")
    valid_ds = {k: v for k, v in datasets.items()
                if len(v.get('genuine', [])) + len(v.get('synthetic', [])) > 0}

    ncols = min(3, len(valid_ds))
    nrows = (len(valid_ds) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6*ncols, 4*nrows), squeeze=False)
    fig.patch.set_facecolor(WHITE)
    fig.suptitle('Audio Duration Distribution — All Datasets',
                 fontsize=14, fontweight='bold', color=NAVY)

    for idx, (name, info) in enumerate(valid_ds.items()):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        g_files = info.get('genuine', [])
        s_files = info.get('synthetic', [])

        if g_files:
            g_durs = [safe_info(f)[0] for f in random.sample(g_files, min(200, len(g_files)))]
            g_durs = [d for d in g_durs if d]
            ax.hist(g_durs, bins=30, alpha=0.7, color=BLUE,
                    label=f'Genuine ({len(g_files):,})', edgecolor='white')
        if s_files:
            s_durs = [safe_info(f)[0] for f in random.sample(s_files, min(200, len(s_files)))]
            s_durs = [d for d in s_durs if d]
            ax.hist(s_durs, bins=30, alpha=0.7, color=RED,
                    label=f'Synthetic ({len(s_files):,})', edgecolor='white')

        ax.set_title(name, fontweight='bold', color=NAVY, fontsize=10)
        ax.set_xlabel('Duration (s)')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide unused axes
    for idx in range(len(valid_ds), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    plt.tight_layout()
    path = str(LOG_DIR / 'eda_02_duration_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print(f"    Saved: {path}")
    return path


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Mel Spectrograms (genuine vs synthetic, one row per dataset)
# ═════════════════════════════════════════════════════════════════════════════

def plot_mel_spectrograms(datasets):
    print("  Generating Plot 3: Mel Spectrograms...")
    ds_with_both = {k: v for k, v in datasets.items()
                    if v.get('genuine') and v.get('synthetic')}

    if not ds_with_both:
        print("    Skipped: no dataset has both genuine and synthetic files.")
        return None

    n_ds   = min(len(ds_with_both), 3)
    n_cols = 6  # 3 genuine + 3 synthetic
    fig, axes = plt.subplots(n_ds, n_cols, figsize=(20, 4.5 * n_ds))
    if n_ds == 1:
        axes = axes[np.newaxis, :]
    fig.patch.set_facecolor(WHITE)
    fig.suptitle('Log-Mel Spectrograms: Genuine vs Synthetic per Dataset',
                 fontsize=14, fontweight='bold', color=NAVY)

    for row, (name, info) in enumerate(list(ds_with_both.items())[:n_ds]):
        g3 = random.sample(info['genuine'],   min(3, len(info['genuine'])))
        s3 = random.sample(info['synthetic'], min(3, len(info['synthetic'])))

        # Row label
        axes[row][0].set_ylabel(name, fontsize=10, fontweight='bold',
                                color=NAVY, labelpad=10)

        for col, fp in enumerate(g3):
            try:
                wav, sr = load_wav(fp)
                mel = librosa.power_to_db(
                    librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=128), ref=np.max)
                librosa.display.specshow(mel, ax=axes[row][col], sr=sr,
                                         x_axis='time', y_axis='mel', cmap='magma')
                title = f'Genuine {col+1}' if row == 0 else ''
                axes[row][col].set_title(title, color=GREEN, fontweight='bold', fontsize=9)
            except Exception as e:
                axes[row][col].text(0.5, 0.5, f'Error\n{str(e)[:30]}',
                                    transform=axes[row][col].transAxes,
                                    ha='center', va='center', fontsize=7)

        for col, fp in enumerate(s3):
            try:
                wav, sr = load_wav(fp)
                mel = librosa.power_to_db(
                    librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=128), ref=np.max)
                im = librosa.display.specshow(mel, ax=axes[row][col+3], sr=sr,
                                              x_axis='time', y_axis='mel', cmap='magma')
                title = f'Synthetic {col+1}' if row == 0 else ''
                axes[row][col+3].set_title(title, color=RED, fontweight='bold', fontsize=9)
            except Exception as e:
                axes[row][col+3].text(0.5, 0.5, f'Error\n{str(e)[:30]}',
                                      transform=axes[row][col+3].transAxes,
                                      ha='center', va='center', fontsize=7)

    plt.tight_layout()
    path = str(LOG_DIR / 'eda_03_mel_spectrograms.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print(f"    Saved: {path}")
    return path


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Spectral Features Comparison (all datasets side by side)
# ═════════════════════════════════════════════════════════════════════════════

def plot_spectral_features(datasets, features_cache):
    print("  Generating Plot 4: Spectral Features Comparison...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(WHITE)
    fig.suptitle('Spectral Features: Genuine vs Synthetic — All Datasets',
                 fontsize=14, fontweight='bold', color=NAVY)

    bp_kw = dict(patch_artist=True, widths=0.35,
                 medianprops=dict(color=NAVY, linewidth=2))
    colors = [BLUE, RED]

    for ax_idx, (feature, label) in enumerate([
        ('centroids', 'Spectral Centroid (Hz)'),
        ('zcrs',      'Zero Crossing Rate'),
        ('rms',       'RMS Energy'),
    ]):
        positions_g, positions_s = [], []
        data_g,      data_s      = [], []
        labels = []
        offset = 0

        for name, cache in features_cache.items():
            g_data = cache['genuine'].get(feature, np.array([]))
            s_data = cache['synthetic'].get(feature, np.array([]))
            if len(g_data) > 2:
                positions_g.append(offset + 1)
                data_g.append(g_data)
            if len(s_data) > 2:
                positions_s.append(offset + 2)
                data_s.append(s_data)
            labels.append((offset + 1.5, name[:12]))
            offset += 3.5

        if data_g:
            bp_g = axes[ax_idx].boxplot(data_g, positions=positions_g,
                                         boxprops=dict(facecolor=BLUE), **bp_kw)
        if data_s:
            bp_s = axes[ax_idx].boxplot(data_s, positions=positions_s,
                                         boxprops=dict(facecolor=RED), **bp_kw)

        axes[ax_idx].set_xticks([x for x, _ in labels])
        axes[ax_idx].set_xticklabels([l for _, l in labels],
                                      rotation=25, ha='right', fontsize=8)
        axes[ax_idx].set_title(label, fontweight='bold', color=NAVY)
        axes[ax_idx].spines['top'].set_visible(False)
        axes[ax_idx].spines['right'].set_visible(False)

    # Legend
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=BLUE, label='Genuine'),
                  Patch(facecolor=RED,  label='Synthetic')]
    axes[2].legend(handles=legend_els, loc='upper right', fontsize=9)

    plt.tight_layout()
    path = str(LOG_DIR / 'eda_04_spectral_features.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print(f"    Saved: {path}")
    return path


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 5 — MFCC Heatmaps (one per dataset)
# ═════════════════════════════════════════════════════════════════════════════

def plot_mfcc_heatmaps(datasets, features_cache):
    print("  Generating Plot 5: MFCC Heatmaps...")
    ds_list = list(features_cache.keys())
    n = len(ds_list)
    fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n), squeeze=False)
    fig.patch.set_facecolor(WHITE)
    fig.suptitle('MFCC Heatmap Comparison — All Datasets',
                 fontsize=14, fontweight='bold', color=NAVY)

    for row, name in enumerate(ds_list):
        cache = features_cache[name]
        g_mfcc = cache['genuine'].get('mfccs', np.zeros((10, 40)))
        s_mfcc = cache['synthetic'].get('mfccs', np.zeros((10, 40)))

        im0 = axes[row][0].imshow(g_mfcc.T, aspect='auto', origin='lower', cmap='coolwarm')
        axes[row][0].set_title(f'{name} — Genuine', color=GREEN, fontweight='bold')
        axes[row][0].set_xlabel('Sample index')
        axes[row][0].set_ylabel('MFCC Coefficient')
        plt.colorbar(im0, ax=axes[row][0], shrink=0.8)

        im1 = axes[row][1].imshow(s_mfcc.T, aspect='auto', origin='lower', cmap='coolwarm')
        axes[row][1].set_title(f'{name} — Synthetic', color=RED, fontweight='bold')
        axes[row][1].set_xlabel('Sample index')
        axes[row][1].set_ylabel('MFCC Coefficient')
        plt.colorbar(im1, ax=axes[row][1], shrink=0.8)

    plt.tight_layout()
    path = str(LOG_DIR / 'eda_05_mfcc_heatmaps.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print(f"    Saved: {path}")
    return path


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 6 — Synthetic Sources Breakdown
# ═════════════════════════════════════════════════════════════════════════════

def plot_synthetic_sources(datasets):
    print("  Generating Plot 6: Synthetic Sources Breakdown...")
    sources = {}
    for name, info in datasets.items():
        if 'sources' in info:
            for src, files in info['sources'].items():
                sources[src] = len(files)
        elif info.get('synthetic'):
            sources[name] = len(info['synthetic'])

    if not sources:
        print("    Skipped: no source breakdown available.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(WHITE)
    fig.suptitle('Synthetic Audio Sources Breakdown',
                 fontsize=14, fontweight='bold', color=NAVY)

    names  = list(sources.keys())
    counts = list(sources.values())
    colors_list = plt.cm.Blues(np.linspace(0.4, 0.9, len(names)))

    axes[0].barh(names, counts, color=colors_list, edgecolor='white')
    for i, (n, c) in enumerate(zip(names, counts)):
        axes[0].text(c + max(counts)*0.01, i, f'{c:,}', va='center', fontsize=9)
    axes[0].set_title('Files per Source', fontweight='bold', color=NAVY)
    axes[0].set_xlabel('Number of Files')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    wedges, texts, autotexts = axes[1].pie(
        counts, labels=names, colors=colors_list, autopct='%1.1f%%',
        startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_fontweight('bold')
    axes[1].set_title(f'Source Proportions\n(Total: {sum(counts):,})',
                      fontweight='bold', color=NAVY)

    plt.tight_layout()
    path = str(LOG_DIR / 'eda_06_synthetic_sources.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print(f"    Saved: {path}")
    return path


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 7 — Cross-Dataset Feature Comparison (violin plots)
# ═════════════════════════════════════════════════════════════════════════════

def plot_cross_dataset_violin(features_cache):
    print("  Generating Plot 7: Cross-Dataset Violin Plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor(WHITE)
    fig.suptitle('Cross-Dataset Feature Distribution (Violin Plots)',
                 fontsize=14, fontweight='bold', color=NAVY)

    for ax_idx, (feature, label) in enumerate([
        ('centroids', 'Spectral Centroid (Hz)'),
        ('zcrs',      'Zero Crossing Rate'),
        ('rms',       'RMS Energy'),
    ]):
        all_data, positions, tick_labels, colors_v = [], [], [], []
        pos = 1
        for name, cache in features_cache.items():
            short = name[:10]
            for label_type, color, data_key in [
                ('G', BLUE, 'genuine'),
                ('S', RED,  'synthetic'),
            ]:
                d = cache[data_key].get(feature, np.array([]))
                if len(d) > 3:
                    all_data.append(d)
                    positions.append(pos)
                    tick_labels.append(f'{short}\n{label_type}')
                    colors_v.append(color)
                    pos += 1
            pos += 0.5

        if all_data:
            parts = axes[ax_idx].violinplot(all_data, positions=positions,
                                             showmedians=True, showextrema=True)
            for i, (pc, col) in enumerate(zip(parts['bodies'], colors_v)):
                pc.set_facecolor(col)
                pc.set_alpha(0.7)
            parts['cmedians'].set_color(NAVY)
            parts['cmedians'].set_linewidth(2)
            axes[ax_idx].set_xticks(positions)
            axes[ax_idx].set_xticklabels(tick_labels, fontsize=7)

        axes[ax_idx].set_title(label, fontweight='bold', color=NAVY)
        axes[ax_idx].spines['top'].set_visible(False)
        axes[ax_idx].spines['right'].set_visible(False)

    plt.tight_layout()
    path = str(LOG_DIR / 'eda_07_cross_dataset_violin.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print(f"    Saved: {path}")
    return path


# ═════════════════════════════════════════════════════════════════════════════
# PLOT 8 — MFCC Coefficient Mean Profiles
# ═════════════════════════════════════════════════════════════════════════════

def plot_mfcc_profiles(features_cache):
    print("  Generating Plot 8: MFCC Mean Profiles...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(WHITE)
    fig.suptitle('MFCC Mean Coefficient Profiles — All Datasets',
                 fontsize=14, fontweight='bold', color=NAVY)

    line_colors = plt.cm.tab10(np.linspace(0, 1, len(features_cache)))

    for ax_idx, label_type in enumerate(['genuine', 'synthetic']):
        title_col = GREEN if label_type == 'genuine' else RED
        axes[ax_idx].set_title(f'{label_type.capitalize()} — Mean MFCC Profile',
                               fontweight='bold', color=title_col)
        axes[ax_idx].set_xlabel('MFCC Coefficient Index')
        axes[ax_idx].set_ylabel('Mean Value')
        axes[ax_idx].spines['top'].set_visible(False)
        axes[ax_idx].spines['right'].set_visible(False)
        axes[ax_idx].axhline(0, color='gray', linewidth=0.5, linestyle='--')

        for (name, cache), color in zip(features_cache.items(), line_colors):
            mfccs = cache[label_type].get('mfccs', np.zeros((1, 40)))
            if mfccs.shape[0] > 1:
                mean_profile = mfccs.mean(axis=0)
                std_profile  = mfccs.std(axis=0)
                x = np.arange(len(mean_profile))
                axes[ax_idx].plot(x, mean_profile, label=name[:14],
                                  color=color, linewidth=2)
                axes[ax_idx].fill_between(x,
                                          mean_profile - std_profile,
                                          mean_profile + std_profile,
                                          alpha=0.12, color=color)

        axes[ax_idx].legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    path = str(LOG_DIR / 'eda_08_mfcc_profiles.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=WHITE)
    plt.close()
    print(f"    Saved: {path}")
    return path


# ═════════════════════════════════════════════════════════════════════════════
# SUMMARY JSON
# ═════════════════════════════════════════════════════════════════════════════

def save_eda_summary(datasets, features_cache):
    summary = {}
    for name, info in datasets.items():
        g = info.get('genuine', [])
        s = info.get('synthetic', [])
        cache = features_cache.get(name, {})
        g_feat = cache.get('genuine', {})
        s_feat = cache.get('synthetic', {})

        entry = {
            'genuine_count':   len(g),
            'synthetic_count': len(s),
            'total':           len(g) + len(s),
        }
        for feat_key in ['durations', 'centroids', 'zcrs', 'rms']:
            for label, feat_dict in [('genuine', g_feat), ('synthetic', s_feat)]:
                arr = feat_dict.get(feat_key, np.array([]))
                if len(arr) > 0:
                    entry[f'{label}_{feat_key}_mean'] = float(np.mean(arr))
                    entry[f'{label}_{feat_key}_std']  = float(np.std(arr))
        summary[name] = entry

    out_path = str(LOG_DIR / 'eda_summary.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"    Saved: {out_path}")
    return out_path


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print('=' * 60)
    print('  ZERO-TRUST DEEPFAKE VOICE DEFENSE — FULL EDA')
    print('=' * 60)

    # ── Discover all datasets ──────────────────────────────────────────────
    print('\n[1/4] Discovering datasets...')
    datasets = discover_datasets()

    if not datasets:
        print('  No datasets found. Check DATA path:', DATA)
        return

    print(f'  Found {len(datasets)} dataset(s):')
    for name, info in datasets.items():
        g = len(info.get('genuine', []))
        s = len(info.get('synthetic', []))
        print(f'    {name:25s}: {g:,} genuine  {s:,} synthetic')

    # ── Extract features ────────────────────────────────────────────────────
    print('\n[2/4] Extracting audio features (150 samples per class per dataset)...')
    features_cache = {}
    for name, info in datasets.items():
        print(f'  Processing: {name}')
        g_files = info.get('genuine',   [])
        s_files = info.get('synthetic', [])
        features_cache[name] = {
            'genuine':   extract_features(g_files, n=150) if g_files else {},
            'synthetic': extract_features(s_files, n=150) if s_files else {},
        }
    print('  Feature extraction complete')

    # ── Generate plots ──────────────────────────────────────────────────────
    print('\n[3/4] Generating plots...')
    saved_plots = []
    saved_plots.append(plot_dataset_overview(datasets))
    saved_plots.append(plot_duration_distribution(datasets))
    saved_plots.append(plot_mel_spectrograms(datasets))
    saved_plots.append(plot_spectral_features(datasets, features_cache))
    saved_plots.append(plot_mfcc_heatmaps(datasets, features_cache))
    saved_plots.append(plot_synthetic_sources(datasets))
    saved_plots.append(plot_cross_dataset_violin(features_cache))
    saved_plots.append(plot_mfcc_profiles(features_cache))

    saved_plots = [p for p in saved_plots if p]

    # ── Save summary ─────────────────────────────────────────────────────────
    print('\n[4/4] Saving EDA summary JSON...')
    save_eda_summary(datasets, features_cache)

    # ── Final report ─────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('  EDA COMPLETE')
    print('=' * 60)
    print(f'\n  Datasets analyzed : {len(datasets)}')
    total_files = sum(len(v.get('genuine',[])) + len(v.get('synthetic',[]))
                      for v in datasets.values())
    print(f'  Total files found : {total_files:,}')
    print(f'  Plots generated   : {len(saved_plots)}')
    print(f'\n  Saved to: {LOG_DIR}')
    print()
    for p in saved_plots:
        size_kb = os.path.getsize(p) / 1024
        print(f'    {os.path.basename(p):45s} ({size_kb:.0f} KB)')
    print()


if __name__ == '__main__':
    main()
