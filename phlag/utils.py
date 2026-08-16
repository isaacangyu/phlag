import time
import argparse

from functools import wraps

def count_lines(filepath):
    try:
        with open(filepath, "r") as f:
            line_count = sum(1 for line in f)
        return line_count
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e

def parse_pattern_string(pattern_str, block_size_bp=500000, total_span=None):
    """
    Parses pattern strings representing locus blocks, percentage intervals, or coordinate intervals.
    
    Supported formats:
      1. Standard token sequence: e.g., 'n1n2n3n4n5n6n8a1a2a3n10n11' (12-chunk pattern)
      2. Range-token sequence: e.g., 'n1-n6,a1-a3,n10-n11' or 'n1-n6_a1-a3_n10-n11'
      3. Percentage interval lists: e.g. '37-62,80-85' or '47-52' (marked as state 1 / anomaly)
      4. Prepended locus pattern + interval suffix: e.g., 'n4n6n3n8n9a3a1a2n1n10n12n2_45-55'
      5. Coordinate intervals: e.g. '0-3000000,3000000-4500000,4500000-6000000'

    Returns:
      (blocks, anomaly_intervals, total_length_bp)
    """
    import re
    blocks = []
    anomaly_intervals = []
    curr_pos = 0

    # 0. Check if there is a trailing/standalone explicit interval range suffix (e.g. '_45-55', ';45-55', '_37-62,80-85', '37-62;80-85')
    range_suffix_match = re.search(r'(?:^|[;_,])(\d+-\d+(?:[;_,]\d+-\d+)*)$', pattern_str)
    if range_suffix_match:
        suffix_str = range_suffix_match.group(1)
        coord_matches = re.findall(r'(\d+)-(\d+)', suffix_str)
        if coord_matches:
            all_vals = [int(v) for pair in coord_matches for v in pair]
            max_val = max(all_vals) if all_vals else 0
            first_start = int(coord_matches[0][0])
            
            # Case A: Percentage intervals (0-100) (e.g. 45-55 or 37-62,80-85)
            if max_val == 100 or (first_start >= 15 and max_val <= 100):
                span = total_span if total_span is not None and total_span > 0 else 6000000
                for s_str, e_str in coord_matches:
                    s_pct, e_pct = int(s_str) / 100.0, int(e_str) / 100.0
                    start_bp = s_pct * span
                    end_bp = e_pct * span
                    anomaly_intervals.append((start_bp, end_bp))
                    blocks.append(('a', f"{s_str}%-{e_str}%", end_bp - start_bp))
                return blocks, anomaly_intervals, span

            # Case B: Contiguous base-pair coordinate partition starting at 0
            elif coord_matches[0][0] == '0' and max_val > 100:
                for idx, (s_str, e_str) in enumerate(coord_matches):
                    s_bp, e_bp = int(s_str), int(e_str)
                    b_type = 'a' if (len(coord_matches) == 3 and idx == 1) or (len(coord_matches) > 3 and 0.25 <= (idx / len(coord_matches)) <= 0.75) else ('n' if idx % 2 == 0 else 'a')
                    length_bp = max(0, e_bp - s_bp)
                    blocks.append((b_type, f"{s_bp}-{e_bp}", length_bp))
                    if b_type == 'a':
                        anomaly_intervals.append((s_bp, e_bp))
                    curr_pos = max(curr_pos, e_bp)
                return blocks, anomaly_intervals, curr_pos

    # 1. Try range tokens with explicit 'n' or 'a' prefixes (e.g. n1-n6,a1-a3 or n1n2n3a1a2...)
    if re.search(r'[an]\d+', pattern_str, re.IGNORECASE):
        range_matches = re.findall(r'([an])(\d+)-(?:([an])?(\d+))', pattern_str, re.IGNORECASE)
        if range_matches:
            for b_type, start_str, _, end_str in range_matches:
                b_type = b_type.lower()
                start_id = int(start_str)
                end_id = int(end_str)
                
                if end_id >= start_id and (end_id - start_id) < 1000:
                    for chunk_id in range(start_id, end_id + 1):
                        blocks.append((b_type, str(chunk_id), block_size_bp))
                        if b_type == 'a':
                            anomaly_intervals.append((curr_pos, curr_pos + block_size_bp))
                        curr_pos += block_size_bp
                else:
                    length_bp = max(0, end_id - start_id)
                    blocks.append((b_type, f"{start_id}-{end_id}", length_bp))
                    if b_type == 'a':
                        anomaly_intervals.append((start_id, end_id))
                    curr_pos = max(curr_pos, end_id)
        else:
            # Fallback to individual chunk tokens like n1n2n3a1a2...
            token_matches = re.findall(r'([an])(\d+)', pattern_str, re.IGNORECASE)
            for b_type, b_id in token_matches:
                b_type = b_type.lower()
                blocks.append((b_type, b_id, block_size_bp))
                if b_type == 'a':
                    anomaly_intervals.append((curr_pos, curr_pos + block_size_bp))
                curr_pos += block_size_bp

    # 2. Chunk index ranges without 'a'/'n' prefix (e.g. 1-6,7-9,10-12)
    if not blocks:
        coord_matches = re.findall(r'(\d+)-(\d+)', pattern_str)
        if coord_matches:
            for idx, (s_str, e_str) in enumerate(coord_matches):
                s_val, e_val = int(s_str), int(e_str)
                b_type = 'a' if (len(coord_matches) == 3 and idx == 1) or (len(coord_matches) > 3 and 0.25 <= (idx / len(coord_matches)) <= 0.75) else ('n' if idx % 2 == 0 else 'a')
                for chunk_id in range(s_val, e_val + 1):
                    blocks.append((b_type, str(chunk_id), block_size_bp))
                    if b_type == 'a':
                        anomaly_intervals.append((curr_pos, curr_pos + block_size_bp))
                    curr_pos += block_size_bp

    return blocks, anomaly_intervals, curr_pos

def clean_locus_name(locus_str):
    """
    Cleans locus strings by stripping 'concat' patterns, prefixes, and file extension tokens.
    e.g. 'neoaves_N482_N477_rate090-time2599554_a200k-a700k_10M_concat' -> 'N482_N477_rate090-time2599554_a200k-a700k'
         'n1n2n3n4n5n6a1a2a3n8n10n11_10M_concat' -> 'n1n2n3n4n5n6a1a2a3n8n10n11'
         'Strigiformes_N297_rate090-time7099554_concat' -> 'Strigiformes_N297_rate090-time7099554'
    """
    import re
    if not locus_str:
        return ""
    s = str(locus_str).strip()
    s = re.sub(r'_(?:10m_|10m|10M_|10M_)?concat(?:_\w+)?$', '', s, flags=re.IGNORECASE)
    s = re.sub(r'_concat$', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^(?:neoaves_|avian_)', '', s, flags=re.IGNORECASE)
    return s

def get_short_sim_name(sim_name):
    """
    Strips redundant category suffixes/infixes from simulation names to match shortened folder names on disk.
    e.g. 'Nyctibiidae_10X_up' -> 'Nyctibiidae'
         'N109_10X_down' -> 'N109'
         'N252_recombination_down' -> 'N252'
         'N482_N477_admixture_rate090-time2599554' -> 'N482_N477_rate090-time2599554'
    """
    s = clean_locus_name(sim_name)
    s = s.replace("_10X_up", "").replace("_10X_down", "")
    s = s.replace("_recombination_up", "").replace("_recombination_down", "")
    s = s.replace("_admixture_", "_")
    return s


def get_simulation_node_name(file_path):
    """
    Extracts the simulation directory segment (the clade/node identifier, e.g.
    'Strigiformes_N297_rate090-time7099554' or 'N555') directly from a caster/phlag
    output path, by taking the path component two levels above 'caster' or 'phlag'
    (.../<node_name>/<pattern>/caster/scores.tsv or .../<node_name>/<pattern>/phlag/report.tsv).
    Returns None if the path doesn't match this convention.
    """
    import pathlib
    parts = pathlib.Path(file_path).parts
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx] in ("caster", "phlag") and idx >= 2:
            return clean_locus_name(parts[idx - 2])
    return None


def get_simulation_clade(caster_scores_path, sim_root=None):
    """
    Locates this run's mapping file (neoaves_<clade>_mapping.tsv) by matching the
    caster/phlag output path's simulation directory segment (see get_simulation_node_name)
    against leaf directories under the simulations root, and splits the mapping
    filename's clade token into name and number -- e.g. 'Strigiformes:3' becomes
    clade_name='Strigiformes', clade_number='3' (the number/letter after the colon is a
    disambiguator distinguishing multiple simulated instances of the same named clade,
    not part of the clade name itself; plain tokens like 'N555' have no colon, so
    clade_number is None).
    Returns (node_name, clade_name, clade_number); clade_name is None if no matching
    mapping file was found.
    """
    import re
    import pathlib
    node_name = get_simulation_node_name(caster_scores_path)
    if not node_name:
        return None, None, None

    if sim_root is None:
        sim_root = get_data_dir() / "simulations"
    sim_root = pathlib.Path(sim_root)
    if not sim_root.exists():
        return node_name, None, None

    for mapping_file in sim_root.rglob("neoaves_*_mapping.tsv"):
        if get_short_sim_name(mapping_file.parent.name) == node_name:
            m = re.match(r'neoaves_(.+)_mapping\.tsv$', mapping_file.name)
            if m:
                clade_name, _, clade_number = m.group(1).partition(":")
                return node_name, clade_name, (clade_number or None)
    return node_name, None, None


def resolve_fasta_from_scores_path(scores_path, sim_root=None):
    """
    Reverse-resolves a not-yet-computed scores.tsv-style output path back to its source
    concat FASTA, by matching the path's simulation-directory segment (the component two
    levels above 'caster', see get_simulation_node_name) against leaf directories under
    the simulations root, then requiring an exact match between the path's pattern-stem
    segment (the component directly above 'caster') and a FASTA filename in that leaf's
    concat/ subdirectory. Does not guess -- returns None if there is no exact match,
    including when the concat/ directory has multiple FASTAs and none match.
    """
    import pathlib
    parts = pathlib.Path(scores_path).parts
    caster_idx = None
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx] == "caster" and idx >= 2:
            caster_idx = idx
            break
    if caster_idx is None:
        return None

    node_name = clean_locus_name(parts[caster_idx - 2])
    pattern_stem = clean_locus_name(parts[caster_idx - 1])

    if sim_root is None:
        sim_root = get_data_dir() / "simulations"
    sim_root = pathlib.Path(sim_root)
    if not sim_root.exists():
        return None

    for concat_dir in sim_root.rglob("concat"):
        leaf_dir = concat_dir.parent
        if get_short_sim_name(leaf_dir.name) != node_name:
            continue
        for fasta_path in concat_dir.iterdir():
            if fasta_path.suffix.lower() in (".fa", ".fasta") and clean_locus_name(fasta_path.stem) == pattern_stem:
                return fasta_path
    return None


def get_clade_info_path(filename):
    """
    Resolves a file under the repo's clade-info/ reference-data directory
    (clade_cu.csv, population-information.tsv, 63K.tre), checking the repo root
    first and the configured data dir as a fallback. Returns None if not found.
    """
    import pathlib
    candidates = [
        get_repo_root() / "clade-info" / filename,
        get_data_dir() / "clade-info" / filename,
    ]
    return next((c for c in candidates if c.exists()), None)


def get_cu_branch_length(candidate_names, csv_path=None):
    """
    Looks up the coalescent-units branch length of an exact species name in
    clade-info/clade_cu.csv, trying each candidate in candidate_names in order.
    Does not infer via MRCA/internal-node reconstruction or fuzzy matching --
    only a literal species-name match against the CSV counts (the CSV is
    species/leaf-level only, so clade-level names like 'Strigiformes' will not
    match; this mirrors the exact-match-only policy the old tree lookup used).
    Returns (matched_name, branch_length), or (None, None) if no candidate
    matched or the CSV could not be resolved.
    """
    import csv
    import pathlib

    csv_path = pathlib.Path(csv_path) if csv_path else get_clade_info_path("clade_cu.csv")
    if not csv_path or not csv_path.exists():
        return None, None

    lengths = {}
    with open(csv_path, "r", newline="") as f:
        for row in csv.DictReader(f):
            species = row.get("species")
            if species and species not in lengths:
                try:
                    lengths[species] = float(row["cu_branch_length"])
                except (TypeError, ValueError):
                    lengths[species] = None

    for name in candidate_names:
        if name in lengths:
            return name, lengths[name]
    return None, None


def get_population_info(label, tsv_path=None):
    """
    Looks up a clade or species's row in clade-info/population-information.tsv
    by exact LABEL match (this file carries both clade-level rows, e.g.
    'Strigiformes', and species-level rows). Returns the row as a dict of
    column name -> raw string value, or None if no exact match / the file
    could not be resolved.
    """
    import csv
    import pathlib

    tsv_path = pathlib.Path(tsv_path) if tsv_path else get_clade_info_path("population-information.tsv")
    if not tsv_path or not tsv_path.exists():
        return None

    with open(tsv_path, "r", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("LABEL") == label:
                return row
    return None


def get_cu_branch_length_from_population_info(label, pop_info=None):
    """
    Computes a clade's own branch length in coalescent units as
    CU = NGEN / (2 * SIZE), using its row in clade-info/population-information.tsv
    (NGEN is that row's own branch duration in generations, not a cumulative
    root-to-tip figure -- see get_population_info's LABEL join).
    Pass an already-fetched pop_info dict (e.g. from get_population_info) to
    avoid re-reading the TSV; otherwise it's looked up by label.
    Returns None if there's no matching row, or NGEN/SIZE are missing,
    non-numeric, or SIZE is zero.
    """
    if pop_info is None:
        pop_info = get_population_info(label)
    if not pop_info:
        return None
    try:
        ngen = float(pop_info.get("NGEN"))
        size = float(pop_info.get("SIZE"))
    except (TypeError, ValueError):
        return None
    if size == 0:
        return None
    return ngen / (2 * size)


ADMIXTURE_DIVERGENCE_THRESHOLD_MYR = 3.5


def get_admixture_divergence_time(sim_name):
    """
    Extracts the admixture divergence time, in millions of years (Myr), from a
    simulation name or path carrying a 'time<NUM>' token.
    e.g. 'N482_N477_rate090-time2599554'                -> 2.599554
         '.../admixture/high/..._rate090-time7099554'   -> 7.099554
    Values above 1000 are read as raw years and normalized to Myr; values at or
    below 1000 are assumed to already be in Myr.
    Returns None when there is no 'time<NUM>' token to read.

    This is the raw quantity that get_simulation_categories() buckets into its
    'low'/'high' subcategory at ADMIXTURE_DIVERGENCE_THRESHOLD_MYR; both share
    this function so the parsing and the threshold each live in exactly one place.
    """
    import re
    m = re.search(r'time(\d+(?:\.\d+)?)', str(sim_name))
    if not m:
        return None
    val = float(m.group(1))
    return val / 1000000.0 if val > 1000 else val


def get_simulation_categories(sim_name):
    """
    Categorizes a simulation name or path into two subdirectory levels.
    Returns (level1, level2) or () if not a recognized 10X/recombination/admixture simulation.

    Always prefer passing a full path over a bare node name: the bare-stub
    fallbacks at the end map names like 'N717'/'N228' to a single fixed category,
    but those node names are reused across several category directories on disk,
    so a bare stub is genuinely ambiguous and the fallback will silently pick one.

    10X:
      up: '10X_up', '10X/up', 'Nyctibiidae', 'N497', 'N544', 'N554', 'N716' -> ('10X', 'up')
      down: '10X_down', '10X/down', 'N109', 'N498', 'N717' -> ('10X', 'down')

    recombination:
      up: 'recombination_up', 'recombination/up' -> ('recombination', 'up')
      down: 'recombination_down', 'recombination/down', 'N252' -> ('recombination', 'down')

    admixture:
      'admixture', or ('rate' and 'time') -> ('admixture', 'low' if time < 3.5 else 'high')
      Time is read by get_admixture_divergence_time() and compared against
      ADMIXTURE_DIVERGENCE_THRESHOLD_MYR; an unreadable/absent time falls back to 'low'.
    """
    s = str(sim_name)
    if "10X_down" in s or "10X/down" in s:
        return ("10X", "down")
    elif "10X_up" in s or "10X/up" in s:
        return ("10X", "up")
    elif "recombination_down" in s or "recombination/down" in s:
        return ("recombination", "down")
    elif "recombination_up" in s or "recombination/up" in s:
        return ("recombination", "up")
    elif "admixture" in s or ("rate" in s and "time" in s):
        t_val = get_admixture_divergence_time(s)
        sub2 = "low" if (t_val is None or t_val < ADMIXTURE_DIVERGENCE_THRESHOLD_MYR) else "high"
        return ("admixture", sub2)
    elif "10X" in s:
        if "down" in s or any(k in s for k in ["N109", "N498", "N717"]):
            return ("10X", "down")
        if "up" in s or any(k in s for k in ["Nyctibiidae", "N497", "N544", "N554", "N716"]):
            return ("10X", "up")
    elif "recombination" in s:
        if "down" in s or "N252" in s:
            return ("recombination", "down")
        if "up" in s:
            return ("recombination", "up")
    # Check node stubs
    if any(k in s for k in ["Nyctibiidae", "N497", "N544", "N554", "N716"]):
        return ("10X", "up")
    if any(k in s for k in ["N109", "N498", "N717"]):
        return ("10X", "down")
    if "N252" in s:
        return ("recombination", "down")
    return ()

def convert_pattern_to_interval(pattern_str):
    """
    Converts a pattern string (which may be a string pattern like 'n1n2n3n4n5n6a1a2a3n8n10n11'
    or 'n4n6n3n8n9a3a1a2n1n10n12n2_45-55') into an interval representation (e.g. '45-55').
    """
    if not pattern_str:
        return ""
    import re
    s = str(pattern_str).strip()

    # 1. Check for trailing explicit interval suffix e.g. '_45-55', ';45-55', '45-55'
    m_suffix = re.search(r'(?:^|[;_,])(\d+-\d+(?:[;_,]\d+-\d+)*)$', s)
    if m_suffix:
        return m_suffix.group(1)

    # 2. Check if string is already a pure interval (e.g., '45-55', '1-6,7-9,10-11', '37-62,80-85')
    if re.fullmatch(r'\d+-\d+(?:[;_,]\d+-\d+)*', s):
        return s

    # 3. Check for aXXk-aYYk_ZZM format
    m_ak = re.search(r'a(\d+)k-a(\d+)k_(\d+)M', s, re.IGNORECASE)
    if m_ak:
        start_k = int(m_ak.group(1))
        end_k = int(m_ak.group(2))
        total_m = int(m_ak.group(3))
        start_pct = int(round((start_k * 1000) / (total_m * 1000000) * 100))
        end_pct = int(round((end_k * 1000) / (total_m * 1000000) * 100))
        return f"{start_pct}-{end_pct}"

    # 4. Handle token patterns containing 'a' and 'n' (e.g. 'n1n2n3n4n5n6a1a2a3n8n10n11')
    if re.search(r'[an]\d+', s, re.IGNORECASE):
        # General conversion via parse_pattern_string
        try:
            blocks, anomaly_intervals, total_span = parse_pattern_string(s)
            if anomaly_intervals and total_span:
                parts = []
                for start_bp, end_bp in anomaly_intervals:
                    start_pct = int(round((start_bp / total_span) * 100))
                    end_pct = int(round((end_bp / total_span) * 100))
                    parts.append(f"{start_pct}-{end_pct}")
                if parts:
                    return ",".join(parts)
        except Exception:
            pass

    return s

def get_locus_description(file_path):
    """
    Extracts a full locus description (branch, params, change/pattern, category, window/step size)
    from a file path or filename, matching Caster's scatter plot locus format.
    e.g. 'N482_N477_rate090-time2599554_45-55_admixture_w50k_s1k'
    """
    if not file_path:
        return ""
    import re
    import pathlib

    file_path = pathlib.Path(file_path)
    parts = file_path.parts

    w_s_part = None
    cat_part = None
    sim_part = None
    pattern_stem = None

    for idx, p in enumerate(parts):
        if re.match(r'w\w+_s\w+', p):
            w_s_part = p
        elif p in ['admixture', '10X', 'recombination']:
            cat_part = p
            if idx + 1 < len(parts) and parts[idx + 1] in ['up', 'down', 'low', 'high']:
                cat_part = f"{p}_{parts[idx + 1]}"
        elif p not in ['scores.tsv', 'report.tsv', 'caster', 'phlag', 'store', 'gaussian', 'gmm', 'low', 'high', 'up', 'down', 'simulations']:
            if '_' in p or 'N' in p or 'Nyctibiidae' in p or 'Strigiformes' in p:
                if not sim_part and not re.match(r'^\d+-\d+(?:[;_,]\d+-\d+)*$', p):
                    sim_part = clean_locus_name(p)

    for idx, p in enumerate(parts):
        if p in ['phlag', 'caster'] and idx > 0:
            candidate = clean_locus_name(parts[idx - 1])
            if candidate not in ['low', 'high', 'up', 'down', 'admixture', '10X', 'recombination', 'gaussian', 'gmm', 'simulations']:
                if re.match(r'^\d+-\d+(?:[;_,]\d+-\d+)*$', candidate) or 'a' in candidate.lower() or 'n' in candidate.lower():
                    pattern_stem = candidate
                elif not sim_part:
                    sim_part = candidate
            if idx > 1 and not sim_part:
                candidate2 = clean_locus_name(parts[idx - 2])
                if candidate2 not in ['low', 'high', 'up', 'down', 'admixture', '10X', 'recombination', 'gaussian', 'gmm', 'simulations']:
                    sim_part = candidate2

    locus_tokens = []
    if sim_part:
        locus_tokens.append(clean_locus_name(sim_part))
    if pattern_stem and pattern_stem != sim_part:
        locus_tokens.append(convert_pattern_to_interval(pattern_stem))
    if cat_part and (not sim_part or cat_part not in sim_part):
        locus_tokens.append(cat_part)
    if w_s_part:
        locus_tokens.append(w_s_part)

    if locus_tokens:
        return "_".join(filter(None, locus_tokens))
    return clean_locus_name(file_path.stem)

def parse_filename_to_dir_structure(filename):
    """
    Parses a filename like 'null-neoaves_alt-Nyctibiidae_10X_up_n1n8a1n5_0_2m_w50k_s1k'
    or 'null-neoaves_alt-Nyctibiidae_10X_up_1-6,7-9,10-11_w50k_s1k'
    Returns a dict with 'null', 'alt', 'pattern', 'locus', 'window_step'.
    """
    import re
    pattern_regex = r'((?:[a-zA-Z0-9;_-]+[;_,])?(?:[an]?\d+-[an]?\d+[;_,]?)+|(?:[an]\d+)+|\d+-\d+(?:[;_,]\d+-\d+)*)'
    # Fallback structure
    m = re.search(r'null-(.*?)_alt-(.*?)_' + pattern_regex + r'_(\d+_\d+m)_(w\w+_s\w+)', filename)
    if m:
        alt_name = m.group(2)
        short_alt = get_short_sim_name(alt_name)
        cats = get_simulation_categories(alt_name)
        cat_prefix = f"{cats[0]}/{cats[1]}/" if cats else ""
        pattern_val = convert_pattern_to_interval(m.group(3))
        return {
            "null": m.group(1),
            "alt": alt_name,
            "pattern": pattern_val,
            "locus": m.group(4),
            "window_step": m.group(5),
            "relative_dir": f"{m.group(5)}/{cat_prefix}{short_alt}/{pattern_val}"
        }
    # Attempt parsing without locus chunk
    m2 = re.search(r'null-(.*?)_alt-(.*?)_' + pattern_regex + r'_(w\w+_s\w+)', filename)
    if m2:
        alt_name = m2.group(2)
        short_alt = get_short_sim_name(alt_name)
        cats = get_simulation_categories(alt_name)
        cat_prefix = f"{cats[0]}/{cats[1]}/" if cats else ""
        pattern_val = convert_pattern_to_interval(m2.group(3))
        return {
            "null": m2.group(1),
            "alt": alt_name,
            "pattern": pattern_val,
            "window_step": m2.group(4),
            "relative_dir": f"{m2.group(4)}/{cat_prefix}{short_alt}/{pattern_val}"
        }
    return None

def integer_pair(arg_str):
    try:
        parts = arg_str.split(",")
        if len(parts) != 2:
            raise ValueError(
                "Argument must contain exactly two integers separated by a comma."
            )
        int1 = int(parts[0].strip())
        int2 = int(parts[1].strip())
        return [int1, int2]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid integer pair format: {arg_str}. {e}")


def is_float(val):
    try:
        return float(val) == float(val)
    except (ValueError, TypeError):
        return False


def limited_float(min_val, max_val):
    def check_range(value):
        fvalue = float(value)
        if not (min_val <= fvalue <= max_val):
            raise ValueError(
                f"Argument must be between {min_val} and {max_val} (inclusive)."
            )
        return fvalue

    return check_range


def limited_int(min_val, max_val):
    def check_range(value):
        ivalue = int(value)
        if not (min_val <= ivalue <= max_val):
            raise ValueError(
                f"Argument must be between {min_val} and {max_val} (inclusive)."
            )
        return ivalue

    return check_range


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function {func.__name__} took {execution_time:.4f} seconds to execute.")
        return result

    return wrapper


def get_repo_root():
    import os
    import pathlib
    override = os.environ.get("PHLAG_REPO_ROOT")
    if override:
        return pathlib.Path(override)
    return pathlib.Path(__file__).resolve().parent.parent


def load_cli_config(tool_name):
    """
    Returns (variable_tokens, fixed_tokens) -- flat lists of raw CLI flag
    tokens -- for tool_name ("caster" or "phlag") from bench/config.json,
    sibling to phlag/ in whichever tree this module is physically running
    from. Deliberately resolved relative to this file rather than via
    get_repo_root() (which honors a PHLAG_REPO_ROOT override) -- that keeps
    config.json frozen together with the code when running from one of
    benchmark.py's run_all() source snapshots, instead of picking up live
    edits to the real repo's config.json mid-run. Missing file or missing
    tool key returns ([], []), matching get_data_dir()'s tolerance for a
    missing .env.
    """
    import json
    import pathlib

    config_path = pathlib.Path(__file__).resolve().parent.parent / "bench" / "config.json"
    if not config_path.exists():
        return [], []

    try:
        config = json.loads(config_path.read_text())
    except Exception:
        return [], []

    tool_config = config.get(tool_name, {})
    return tool_config.get("variable", []), tool_config.get("fixed", [])


def apply_cli_config(parser, argv, tool_name):
    """
    Parses argv with parser, with defaults sourced from config.json's
    "variable"/"fixed" sections for tool_name (see load_cli_config).

    Deliberately does NOT splice config tokens into argv before parsing --
    "variable" flags with nargs="*" (e.g. --plot) would then greedily swallow
    whatever token follows them, including the real positional path, if a
    config token list happened to be placed next to it. Instead:
      - "variable" tokens are parsed once through the same parser (with no
        positional present -- every positional here is nargs="?", so that's
        a valid parse on its own) and applied via parser.set_defaults(), so
        they only take effect for flags argv doesn't already specify.
      - "fixed" tokens are parsed the same way, but only the destinations
        they actually set (detected via a sentinel default) are then forced
        onto the final parsed args, overriding anything argv specified.

    Positional actions (option_strings == []) are excluded from the "fixed"
    destination set -- argparse always assigns nargs="?" positionals their
    own default when no positional token is present, even overwriting an
    already-sentineled namespace attribute, so they can't be sentinel-
    detected the way optional flags can. Config should never carry a
    positional's value anyway (paths always come from real argv).
    """
    import argparse

    variable_tokens, fixed_tokens = load_cli_config(tool_name)

    if variable_tokens:
        variable_ns, _ = parser.parse_known_args(variable_tokens)
        parser.set_defaults(**vars(variable_ns))

    args = parser.parse_args(argv)

    if fixed_tokens:
        sentinel = object()
        dests = [a.dest for a in parser._actions if a.dest != "help" and a.option_strings]
        empty_ns = argparse.Namespace(**{dest: sentinel for dest in dests})
        fixed_ns, _ = parser.parse_known_args(fixed_tokens, namespace=empty_ns)
        for key in dests:
            value = getattr(fixed_ns, key)
            if value is not sentinel:
                setattr(args, key, value)

    return args


def get_data_dir():
    """
    Returns the user data output directory where scores and HMM state files should be stored.
    Checks CONNECTION_DIR environment variable or .env file first, then defaults to /drive2/iang if present.
    Raises if neither is available -- there is no repo-relative fallback.
    """
    import os
    import pathlib
    conn_env = os.environ.get("CONNECTION_DIR")
    if not conn_env:
        try:
            env_file = get_repo_root() / ".env"
            if env_file.exists():
                for line in env_file.read_text().splitlines():
                    line = line.strip()
                    if line.startswith("CONNECTION_DIR="):
                        conn_env = line.split("=", 1)[1].strip().strip("\"'")
                        break
        except Exception:
            pass

    if conn_env:
        return pathlib.Path(conn_env)

    drive2_dir = pathlib.Path("/drive2/iang")
    if drive2_dir.exists():
        return drive2_dir

    raise RuntimeError(
        "CONNECTION_DIR is not set (env var or .env) and /drive2/iang does not exist. "
        "Set CONNECTION_DIR to the shared data directory."
    )


def get_phlag_output_base(base_path=None):
    """
    Ensures that output paths do not duplicate 'phlag/phlag' when base_path already ends with 'phlag'.
    """
    import pathlib
    if base_path is None:
        base_path = get_data_dir()
    p = pathlib.Path(base_path)
    if p.name.lower() == "phlag":
        return p
    return p / "phlag"


def find_mapping_file(leaf_dir):
    """
    Finds the single population mapping file (neoaves_*_mapping.tsv, falling back to
    *_mapping.tsv) directly inside a simulation leaf directory.
    Returns (mapping_path, None) if exactly one is found, else (None, reason).
    """
    import pathlib
    leaf_dir = pathlib.Path(leaf_dir)
    mapping_candidates = sorted(leaf_dir.glob("neoaves_*_mapping.tsv")) or sorted(leaf_dir.glob("*_mapping.tsv"))
    if len(mapping_candidates) == 0:
        return None, "no mapping file found"
    if len(mapping_candidates) > 1:
        return None, f"ambiguous mapping files ({len(mapping_candidates)} found)"
    return mapping_candidates[0], None


def resolve_input_file(path_input, default_subdirs=None, default_exts=None):
    """
    Resolves a user-provided file or directory path flexibly.
    Supports absolute paths, relative paths, filenames without directory,
    stem names without extension, directory names containing fasta files,
    and automatic searches within repo store/msa/simulations directories.
    """
    import pathlib

    if path_input is None:
        return None

    path_obj = pathlib.Path(path_input)

    # 1. Direct path check (file exists)
    if path_obj.exists() and path_obj.is_file():
        return path_obj.resolve()

    # 2. Build candidate filenames
    candidates = []
    if path_obj.suffix:
        candidates.append(path_obj)
    else:
        if default_exts:
            for ext in default_exts:
                if not ext.startswith("."):
                    ext = "." + ext
                candidates.append(pathlib.Path(str(path_obj) + ext))
        else:
            candidates.extend([
                pathlib.Path(str(path_obj) + ".fa"),
                pathlib.Path(str(path_obj) + ".fasta"),
                pathlib.Path(str(path_obj) + ".tsv"),
                pathlib.Path(str(path_obj) + ".txt"),
            ])

    # Direct candidate checks relative to CWD
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand.resolve()

    # 3. Build search directories
    repo_root = get_repo_root()
    data_dir = get_data_dir()

    search_bases = [
        pathlib.Path.cwd(),
        data_dir,
        repo_root,
        repo_root / "connection_dir",
        repo_root / "connection_dir" / "simulations",
        repo_root / "store",
        repo_root / "test",
        pathlib.Path.cwd() / "test",
        pathlib.Path("/drive2/iang"),
    ]

    if default_subdirs is None:
        default_subdirs = [
            "store/msa/concat", "store/msa", "store", 
            "msa/concat", "msa", "concat", "simulations"
        ]

    search_dirs = []
    for base in search_bases:
        if base.exists():
            search_dirs.append(base)
            for s in default_subdirs:
                sub_path = base / s
                if sub_path.exists():
                    search_dirs.append(sub_path)
            try:
                for sim_sub in list(base.glob("simulations/*/*/*/concat")) + list(base.glob("simulations/*/concat")):
                    if sim_sub.is_dir():
                        search_dirs.append(sim_sub)
            except Exception:
                pass

    # Deduplicate search_dirs
    seen_dirs = set()
    unique_dirs = []
    for d in search_dirs:
        try:
            resolved_d = d.resolve()
            if resolved_d not in seen_dirs:
                seen_dirs.add(resolved_d)
                unique_dirs.append(d)
        except Exception:
            pass

    for cand in candidates:
        cand_name = cand.name
        for sdir in unique_dirs:
            check_path = sdir / cand_name
            if check_path.exists() and check_path.is_file():
                return check_path.resolve()

    # 5. Directory resolution
    clean_name = path_obj.name
    if clean_name.startswith("alt-"):
        clean_name = clean_name[4:]
    parts = clean_name.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        clean_name = parts[0]

    cats = get_simulation_categories(clean_name) or get_simulation_categories(path_obj.name)
    short_clean = get_short_sim_name(clean_name)
    short_path_name = get_short_sim_name(path_obj.name)
    names_to_try = []
    for n in [short_clean, short_path_name, clean_name, path_obj.name]:
        if n and n not in names_to_try:
            names_to_try.append(n)

    dir_candidates = [path_obj]
    if cats:
        cat_rel = pathlib.Path(cats[0]) / cats[1]
        for sname in names_to_try:
            dir_candidates.extend([
                repo_root / "connection_dir" / "simulations" / cat_rel / sname,
                data_dir / "simulations" / cat_rel / sname,
                repo_root / "simulations" / cat_rel / sname,
                pathlib.Path("/drive2/iang/simulations") / cat_rel / sname,
            ])
    for sname in names_to_try:
        dir_candidates.extend([
            repo_root / "connection_dir" / "simulations" / sname,
            data_dir / "simulations" / sname,
            data_dir / sname,
            repo_root / "simulations" / sname,
            pathlib.Path("/drive2/iang/simulations") / sname,
        ])
    for dcand in dir_candidates:
        if dcand.exists() and dcand.is_dir():
            concat_dir = dcand / "concat"
            target_dirs = [concat_dir] if (concat_dir.exists() and concat_dir.is_dir()) else [dcand]
            for tdir in target_dirs:
                if tdir.name == "simulated" or tdir.parent.name == "simulated":
                    continue
                files = list(tdir.glob("*.fa")) + list(tdir.glob("*.fasta")) + list(tdir.glob("*.fa.gz"))
                # Exclude any files under a simulated subfolder
                files = [f for f in files if "simulated" not in f.parts]
                if files:
                    # Pull most recent FASTA file in concat/ or directory
                    files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                    return files[0].resolve()

    return path_obj


def get_most_recent_file(default_subdirs=None, default_exts=None, exclude_prefixes=None, target_dir_name=None):
    """
    Finds the most recently created or modified file in store/scores, store/msa/concat,
    or specified subdirectories, ignoring report output files.
    """
    import os
    import pathlib

    repo_root = get_repo_root()
    data_dir = get_data_dir()

    if exclude_prefixes is None:
        exclude_prefixes = ["report_", "gaussian_", "scores_", "em_", "walkthrough", "implementation_plan"]

    if default_subdirs is None:
        default_subdirs = ["store/scores", "scores", "store/msa/concat", "msa/concat"]

    search_dirs = []
    bases_to_search = [repo_root, pathlib.Path.cwd(), data_dir, pathlib.Path("/drive2/iang")]

    if target_dir_name:
        for base in bases_to_search:
            if base.exists():
                try:
                    for d in base.rglob(target_dir_name):
                        if d.is_dir():
                            search_dirs.append(d)
                except Exception:
                    pass
    
    for s in default_subdirs:
        for base in bases_to_search:
            sp = base / s
            if sp.exists():
                search_dirs.append(sp)

    # Also search recursively in model output directories (store/phlag, phlag, data_dir) for scores files
    for base in [get_phlag_output_base(repo_root / "store" / "phlag"), get_phlag_output_base(data_dir), repo_root / "phlag", data_dir]:
        if base.exists():
            try:
                for score_file in base.rglob("scores.tsv"):
                    if score_file.is_file():
                        search_dirs.append(score_file.parent)
                for score_file in base.rglob("*.tsv"):
                    if score_file.is_file() and not any(score_file.name.startswith(p) for p in exclude_prefixes):
                        search_dirs.append(score_file.parent)
            except Exception:
                pass

    seen_dirs = set()
    newest_file = None
    newest_mtime = -1.0

    for d in search_dirs:
        try:
            resolved_d = d.resolve()
            if target_dir_name and resolved_d.name != target_dir_name:
                continue
            if resolved_d in seen_dirs:
                continue
            seen_dirs.add(resolved_d)

            for item in d.iterdir():
                if item.is_file():
                    fname = item.name
                    if exclude_prefixes and any(fname.startswith(p) for p in exclude_prefixes):
                        continue
                    if default_exts and not any(fname.endswith(ext) for ext in default_exts):
                        continue
                    mtime = item.stat().st_mtime
                    if mtime > newest_mtime:
                        newest_mtime = mtime
                        newest_file = item.resolve()
        except Exception:
            pass

    return newest_file



