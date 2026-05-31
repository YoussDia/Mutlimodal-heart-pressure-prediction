import os
import re
import numpy as np
import pandas as pd
from ecg_segmentation_2 import (
    load_mat_file as load_segment_mat,
    detect_peaks_ecg,
    detect_PR_interval,
    detect_QT_interval,
    detect_rpeaks_hybrid,
    # pacing spike removal not used anymore
)
from scg_segmentation import detect_scg_events, compute_scg_intervals

DEFAULT_ECG_FS = 1000
ANNOTATION_FOLDER = "annotations"

def get_sampling_rate_from_time(time, default_fs=DEFAULT_ECG_FS):
    """Retourne la fréquence d'échantillonnage estimée à partir d'un vecteur de temps."""
    if time is None or len(time) < 2:
        return default_fs

    diffs = np.diff(time)
    valid_diffs = diffs[diffs > 0]
    if len(valid_diffs) == 0:
        return default_fs

    return int(round(1.0 / np.median(valid_diffs)))


def get_base_patient_id(patient_id):
    """
    Extrait l'identifiant de base TRM### depuis un nom de segment.

    Exemple:
        TRM163-RHC1_segment_15 -> TRM163
    """
    match = re.search(r'TRM\d+', patient_id)
    return match.group(0) if match else patient_id


def find_annotation_file(patient_id):
    """
    Retourne le chemin du fichier d'annotation correspondant.
    """
    exact_path = os.path.join(ANNOTATION_FOLDER, f"{patient_id}.txt")
    if os.path.exists(exact_path):
        return exact_path

    base_id = get_base_patient_id(patient_id)
    if base_id != patient_id:
        # Chercher un fichier d'annotation commençant par le même patient de base
        for fname in os.listdir(ANNOTATION_FOLDER):
            if fname.startswith(base_id) and fname.lower().endswith('.txt'):
                return os.path.join(ANNOTATION_FOLDER, fname)

    return None


def extract_patient_class(patient_id):
    """
    Extrait la classe du patient depuis le fichier d'annotation.
    
    Args:
        patient_id (str): ID du patient (ex: "TRM107.RHC1" ou "TRM163-RHC1_segment_15")
    
    Returns:
        int: Classe du patient (I→1, II→2, III→3) ou None si non trouvé
    """
    annotation_path = find_annotation_file(patient_id)
    if not annotation_path:
        return None
    
    try:
        with open(annotation_path, 'r') as f:
            content = f.read()
        
        # Chercher "class I", "class II", "class III"
        match = re.search(r'class\s+([IVX]+)', content, re.IGNORECASE)
        if match:
            roman_numeral = match.group(1).upper()
            # Convertir chiffres romains en entiers
            roman_to_int = {"I": 1, "II": 2, "III": 3, "IV": 4}
            return roman_to_int.get(roman_numeral, None)
    except Exception as e:
        print(f"  ⚠ Erreur lors de la lecture de l'annotation pour {patient_id}: {e}")
    
    return None


def load_segment_data(mat_path):
    """Charge un segment .mat et renvoie ECG, SCG, temps et fréquence."""
    data = load_segment_mat(mat_path)

    ecg_clean = np.asarray(data.get("ECG_clean", []))
    ecg_clean = np.ravel(ecg_clean) if ecg_clean is not None else np.array([])

    scg_clean = data.get("SCG_clean", None)
    if scg_clean is not None:
        scg_clean = np.asarray(scg_clean)
        scg_clean = np.ravel(scg_clean)

    time = np.asarray(data.get("time", []))
    fs = int(data.get("fs", DEFAULT_ECG_FS))
    if time is None or len(time) == 0:
        time = np.arange(len(ecg_clean)) / fs
    else:
        time = np.ravel(time)

    patient = str(data.get("patient", "")).strip()

    return {
        "ECG_clean": ecg_clean,
        "SCG_clean": scg_clean,
        "time": time,
        "fs": fs,
        "patient": patient,
    }

# pacing spike cleaning removed: we no longer detect or remove pacer spikes
def _identity_clean(ecg_segment, fs):
    """Placeholder: do not modify ECG; pacer spike logic removed."""
    return ecg_segment


def compute_ecg_beat_rows(ecg_segment, time_segment, fs):
    """Retourne une ligne par battement ECG avec PR, QT, RR, HR et statuts par segment."""
    rows = []
    if len(ecg_segment) == 0:
        return rows, {
            "n_beats": 0,
            "mean_pr_ms": np.nan,
            "median_pr_ms": np.nan,
            "mean_qt_ms": np.nan,
            "median_qt_ms": np.nan,
            "mean_rr_s": np.nan,
            "median_rr_s": np.nan,
            "mean_hr_bpm": np.nan,
            "median_hr_bpm": np.nan,
        }
    ecg_segment = _identity_clean(ecg_segment, fs)
    r_peaks = detect_rpeaks_hybrid(ecg_segment, fs)
    if isinstance(r_peaks, tuple):
        r_peaks = r_peaks[0]
    r_peaks = np.asarray(r_peaks, dtype=int)

    rr_s_values = []
    hr_bpm_values = []
    pr_ms_values = []
    qt_ms_values = []

    for i, r_idx in enumerate(r_peaks):
        pr_ms = np.nan
        qt_ms = np.nan
        rr_s = np.nan
        hr_bpm = np.nan

        if i > 0:
            rr_s = float((r_idx - r_peaks[i-1]) / fs)
            if rr_s > 0:
                hr_bpm = float(60.0 / rr_s)
                rr_s_values.append(rr_s)
                hr_bpm_values.append(hr_bpm)

        p_idx = detect_peaks_ecg(ecg_segment, r_idx, fs=fs, window_ms=200, offset_ms=80, name_peak="P")
        q_idx = detect_peaks_ecg(ecg_segment, r_idx, fs=fs, window_ms=80, offset_ms=10, name_peak="Q")
        t_idx = detect_peaks_ecg(ecg_segment, r_idx, fs=fs, window_ms=450, offset_ms=150, name_peak="T")

        if p_idx is not None and q_idx is not None:
            onset_p, onset_q = detect_PR_interval(ecg_segment, p_idx, q_idx, fs)
            if 0 <= onset_p < onset_q < len(time_segment):
                pr_ms = float((time_segment[onset_q] - time_segment[onset_p]) * 1000.0)
                pr_ms_values.append(pr_ms)

        if q_idx is not None and t_idx is not None:
            onset_q, offset_t = detect_QT_interval(ecg_segment, q_idx, t_idx, fs)
            if 0 <= onset_q < offset_t < len(time_segment):
                qt_ms = float((time_segment[offset_t] - time_segment[onset_q]) * 1000.0)
                qt_ms_values.append(qt_ms)

        rows.append({
            "beat_index": i,
            # r_sample removed
            "r_time_s": float(time_segment[r_idx]) if r_idx < len(time_segment) else np.nan,
            "pr_ms": pr_ms,
            "qt_ms": qt_ms,
            "rr_s": rr_s,
            "hr_bpm": hr_bpm,
        })

    summary = {
        "n_beats": len(r_peaks),
        "mean_pr_ms": float(np.mean(pr_ms_values)) if len(pr_ms_values) > 0 else np.nan,
        "median_pr_ms": float(np.median(pr_ms_values)) if len(pr_ms_values) > 0 else np.nan,
        "mean_qt_ms": float(np.mean(qt_ms_values)) if len(qt_ms_values) > 0 else np.nan,
        "median_qt_ms": float(np.median(qt_ms_values)) if len(qt_ms_values) > 0 else np.nan,
        "mean_rr_s": float(np.mean(rr_s_values)) if len(rr_s_values) > 0 else np.nan,
        "median_rr_s": float(np.median(rr_s_values)) if len(rr_s_values) > 0 else np.nan,
        "mean_hr_bpm": float(np.mean(hr_bpm_values)) if len(hr_bpm_values) > 0 else np.nan,
        "median_hr_bpm": float(np.median(hr_bpm_values)) if len(hr_bpm_values) > 0 else np.nan,
        "r_peaks": r_peaks,
    }

    return rows, summary


def compute_scg_beat_rows(scg_segment, r_peaks, fs):
    """Retourne une ligne par battement SCG avec PEP, ET, IVCT, IVRT et événements détectés."""
    rows = []
    summary = {
        # scg_* counts removed
        "mean_pep_ms": np.nan,
        "median_pep_ms": np.nan,
        "mean_et_ms": np.nan,
        "median_et_ms": np.nan,
        "mean_ivct_ms": np.nan,
        "median_ivct_ms": np.nan,
        "mean_ivrt_ms": np.nan,
        "median_ivrt_ms": np.nan,
    }

    if scg_segment is None or len(scg_segment) == 0 or len(r_peaks) == 0:
        return rows, summary

    scg_events = detect_scg_events(scg_segment, r_peaks, fs)

    pep_values = []
    et_values = []
    ivct_values = []
    ivrt_values = []

    for i, r_idx in enumerate(r_peaks):
        mc_onset = scg_events["MC_onset"][i] if i < len(scg_events["MC_onset"]) else None
        ao_onset = scg_events["AO_onset"][i] if i < len(scg_events["AO_onset"]) else None
        ac_onset = scg_events["AC_onset"][i] if i < len(scg_events["AC_onset"]) else None
        mo_onset = scg_events["MO_onset"][i] if i < len(scg_events["MO_onset"]) else None

        pep_ms = np.nan
        et_ms = np.nan
        ivct_ms = np.nan
        ivrt_ms = np.nan

        if ao_onset is not None and ao_onset > r_idx:
            pep_ms = float((ao_onset - r_idx) / fs * 1000.0)
            pep_values.append(pep_ms)

        if ao_onset is not None and ac_onset is not None and ac_onset > ao_onset:
            et_ms = float((ac_onset - ao_onset) / fs * 1000.0)
            et_values.append(et_ms)

        if mc_onset is not None and ao_onset is not None and ao_onset > mc_onset:
            ivct_ms = float((ao_onset - mc_onset) / fs * 1000.0)
            ivct_values.append(ivct_ms)

        if ac_onset is not None and mo_onset is not None and mo_onset > ac_onset:
            ivrt_ms = float((mo_onset - ac_onset) / fs * 1000.0)
            ivrt_values.append(ivrt_ms)

        rows.append({
            "pep_ms": pep_ms,
            "et_ms": et_ms,
            "ivct_ms": ivct_ms,
            "ivrt_ms": ivrt_ms,
        })

    # counts removed; compute only mean/median intervals
    summary["mean_pep_ms"] = float(np.mean(pep_values)) if len(pep_values) > 0 else np.nan
    summary["median_pep_ms"] = float(np.median(pep_values)) if len(pep_values) > 0 else np.nan
    summary["mean_et_ms"] = float(np.mean(et_values)) if len(et_values) > 0 else np.nan
    summary["median_et_ms"] = float(np.median(et_values)) if len(et_values) > 0 else np.nan
    summary["mean_ivct_ms"] = float(np.mean(ivct_values)) if len(ivct_values) > 0 else np.nan
    summary["median_ivct_ms"] = float(np.median(ivct_values)) if len(ivct_values) > 0 else np.nan
    summary["mean_ivrt_ms"] = float(np.mean(ivrt_values)) if len(ivrt_values) > 0 else np.nan
    summary["median_ivrt_ms"] = float(np.median(ivrt_values)) if len(ivrt_values) > 0 else np.nan

    return rows, summary

def create_dataset_from_patients(processed_folder, start_time=0, window_s=30, fs=None):
    """
    Crée un dataset avec une ligne par battement ECG/SCG et les valeurs exactes des intervalles.

    Args:
        processed_folder (str): Dossier contenant les fichiers .mat
        start_time (float): Temps de début de la fenêtre en secondes
        window_s (float): Durée de la fenêtre en secondes
        fs (int|None): Fréquence d'échantillonnage ; si None, on utilise le fs contenu dans le .mat

    Returns:
        pd.DataFrame: Dataset avec une ligne par battement de segment
    """
    all_rows = []

    for fname in sorted(os.listdir(processed_folder)):
        if not fname.lower().endswith(".mat"):
            continue

        mat_path = os.path.join(processed_folder, fname)
        segment_name = os.path.splitext(fname)[0]

        print(f"\nTraitement du segment: {segment_name}")

        try:
            data = load_segment_data(mat_path)
        except Exception as e:
            print(f"  ⚠ Erreur lors du chargement du fichier : {e}")
            continue

        ecg_clean = data["ECG_clean"]
        scg_clean = data.get("SCG_clean")
        time = data["time"]
        segment_fs = int(data["fs"])
        patient_id = data.get("patient") or segment_name
        patient_class = extract_patient_class(patient_id)

        if patient_class is not None:
            print(f"  ✓ Classe: {patient_class}")
        else:
            print(f"  ⚠ Classe non trouvée pour {patient_id}")

        if len(ecg_clean) == 0:
            print("  ⚠ ECG vide, segment ignoré")
            continue

        start_idx = int(start_time * segment_fs)
        end_idx = start_idx + int(window_s * segment_fs)
        if end_idx > len(ecg_clean):
            end_idx = len(ecg_clean)

        ecg_segment = ecg_clean[start_idx:end_idx]
        time_segment = time[start_idx:end_idx]
        scg_segment = None
        if scg_clean is not None and len(scg_clean) > start_idx:
            scg_segment = scg_clean[start_idx:end_idx]

        ecg_rows, ecg_summary = compute_ecg_beat_rows(ecg_segment, time_segment, segment_fs)
        scg_rows, scg_summary = compute_scg_beat_rows(scg_segment, ecg_summary["r_peaks"], segment_fs)

        for idx, beat_row in enumerate(ecg_rows):
            scg_row = scg_rows[idx] if idx < len(scg_rows) else {
                "pep_ms": np.nan,
                "et_ms": np.nan,
                "ivct_ms": np.nan,
                "ivrt_ms": np.nan,
            }

            row = {
                "patient_id": patient_id,
                "segment_name": segment_name,
                "classe": patient_class,
                "beat_index": beat_row["beat_index"],
                "r_time_s": beat_row["r_time_s"],
                "pr_ms": beat_row["pr_ms"],
                "qt_ms": beat_row["qt_ms"],
                "rr_s": beat_row["rr_s"],
                "hr_bpm": beat_row["hr_bpm"],
                "mean_pr_ms": ecg_summary["mean_pr_ms"],
                "median_pr_ms": ecg_summary["median_pr_ms"],
                "mean_qt_ms": ecg_summary["mean_qt_ms"],
                "median_qt_ms": ecg_summary["median_qt_ms"],
                "mean_rr_s": ecg_summary["mean_rr_s"],
                "median_rr_s": ecg_summary["median_rr_s"],
                "mean_hr_bpm": ecg_summary["mean_hr_bpm"],
                "median_hr_bpm": ecg_summary["median_hr_bpm"],
                "pep_ms": scg_row["pep_ms"],
                "et_ms": scg_row["et_ms"],
                "ivct_ms": scg_row["ivct_ms"],
                "ivrt_ms": scg_row["ivrt_ms"],
                "mean_pep_ms": scg_summary["mean_pep_ms"],
                "median_pep_ms": scg_summary["median_pep_ms"],
                "mean_et_ms": scg_summary["mean_et_ms"],
                "median_et_ms": scg_summary["median_et_ms"],
                "mean_ivct_ms": scg_summary["mean_ivct_ms"],
                "median_ivct_ms": scg_summary["median_ivct_ms"],
                "mean_ivrt_ms": scg_summary["mean_ivrt_ms"],
                "median_ivrt_ms": scg_summary["median_ivrt_ms"],
            }
            all_rows.append(row)

        print(f"  Beats détectés: {ecg_summary['n_beats']}")
        pr_measured = int(sum(1 for v in ecg_rows if not np.isnan(v.get('pr_ms'))))
        qt_measured = int(sum(1 for v in ecg_rows if not np.isnan(v.get('qt_ms'))))
        print(f"  PR intervals measured: {pr_measured}, QT intervals measured: {qt_measured}")
        print(f"  SCG rows: {len(scg_rows)}, mean PEP(ms): {scg_summary.get('mean_pep_ms', np.nan)}, mean ET(ms): {scg_summary.get('mean_et_ms', np.nan)}")

    df = pd.DataFrame(all_rows)
    columns_order = [
        "patient_id", "segment_name", "classe", "beat_index", "r_time_s",
        "pr_ms", "qt_ms", "rr_s", "hr_bpm",
        "mean_pr_ms", "median_pr_ms",
        "mean_qt_ms", "median_qt_ms",
        "mean_rr_s", "median_rr_s", "mean_hr_bpm", "median_hr_bpm",
        "pep_ms", "et_ms", "ivct_ms", "ivrt_ms",
        "mean_pep_ms", "median_pep_ms", "mean_et_ms", "median_et_ms",
        "mean_ivct_ms", "median_ivct_ms", "mean_ivrt_ms", "median_ivrt_ms",
    ]
    df = df[[col for col in columns_order if col in df.columns]]
    print(f"\n{'='*60}")
    print(f"Dataset créé: {len(df)} battements")
    print(f"{'='*60}\n")

    return df

def save_dataset(df, output_path):
    """
    Sauvegarde le dataset dans un fichier CSV.

    Args:
        df (pd.DataFrame): Le dataset
        output_path (str): Chemin de sortie
    """
    df.to_csv(output_path, index=False)
    print(f"Dataset sauvegardé dans {output_path}")

# Exemple d'utilisation
if __name__ == "__main__":
    processed_folder = "15_segments_per_patient_30S"
    
    # Créer le dataset simplifié
    dataset = create_dataset_from_patients(
        processed_folder=processed_folder,
        start_time=0,
        window_s=30
    )
    
    print("\n" + "="*60)
    print("Aperçu du dataset:")
    print("="*60)
    print(dataset.head(10))
    print(f"\nStatistiques des features:")
    print(dataset.describe())
    
    save_dataset(dataset, "ecg_scg_segment_dataset.csv")