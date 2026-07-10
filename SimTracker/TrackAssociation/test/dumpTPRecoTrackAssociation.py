#!/usr/bin/env python3
"""Dump and summarise TrackingParticle <-> reco::Track associations from NanoAOD.

The producers in SimTracker/TrackAssociation/python/trackingParticleRecoTrackAssociationTables_cff.py
write two pairs of flat tables per event:

  TPAssoc      (one row per TrackingParticle)
    nTPAssocLinks[nTPAssoc]   number of reco tracks linked to each TP
    oTPAssocLinks[nTPAssoc]   offset into the TPAssocLinks sub-table
  TPAssocLinks (flat, one row per TP -> Track link)
    index[]   reco-track index
    score[]   association score

  TrackAssoc / TrackAssocLinks: same shape for the reverse direction.

This script reads those branches, prints the first K rows in each direction for
the first N events, and reports per-event and aggregated efficiency/fake/merge
counts. Requires `uproot` (and `numpy`); does not need a CMSSW environment.
"""

import argparse
import sys

import numpy as np
import uproot


def make_branches(prefix):
    """Build the list of expected branch names for a given FlatTable-name prefix.

    The offline NanoAOD tables are named TPAssoc/TPAssocLinks/TrackAssoc/TrackAssocLinks
    (prefix=""). The Phase-2 HLT NANO step uses HLT-prefixed names: hltTPAssoc, ...
    (prefix="hlt"). The branch layout produced by OneToManyWithQualityFlatTableProducer
    is identical apart from the prefix.
    """
    tp = f"{prefix}TPAssoc"
    tpl = f"{prefix}TPAssocLinks"
    tr = f"{prefix}TrackAssoc"
    trl = f"{prefix}TrackAssocLinks"
    return {
        "tp_n": f"{tp}_n{tpl}",
        "tp_o": f"{tp}_o{tpl}",
        "tp_lidx": f"{tpl}_index",
        "tp_lscore": f"{tpl}_score",
        "trk_n": f"{tr}_n{trl}",
        "trk_o": f"{tr}_o{trl}",
        "trk_lidx": f"{trl}_index",
        "trk_lscore": f"{trl}_score",
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="Path to a NanoAOD ROOT file containing the TPAssoc/TrackAssoc tables")
    p.add_argument("-e", "--events", type=int, default=5, help="Number of events to process (default: 5)")
    p.add_argument("-k", "--rows", type=int, default=5, help="Rows to dump per direction per event (default: 5)")
    p.add_argument("--tree", default="Events", help="TTree name (default: Events)")
    p.add_argument(
        "--prefix",
        default="",
        help="FlatTable-name prefix: empty for offline NANO (TPAssoc, TrackAssoc), "
        "'hlt' for Phase-2 HLT NANO (hltTPAssoc, hltTrackAssoc). Default: empty.",
    )
    p.add_argument("--no-dump", action="store_true", help="Skip per-event row dump; only print stats")
    p.add_argument(
        "--score-cut",
        type=float,
        default=None,
        help="Drop links with score < cut from both dump and stats (default: keep all)",
    )
    return p.parse_args(argv)


def slice_links(idx, counts, offsets, link_idx, link_score, score_cut):
    """Return the list of (target_index, score) pairs for entry `idx`."""
    o = int(offsets[idx])
    n = int(counts[idx])
    if n == 0:
        return []
    pairs = list(zip(link_idx[o:o + n].tolist(), link_score[o:o + n].tolist()))
    if score_cut is not None:
        pairs = [(i, s) for (i, s) in pairs if s >= score_cut]
    return pairs


def per_entry_counts(counts, offsets, link_score, score_cut):
    """Return an array of effective link counts after applying the optional score cut."""
    if score_cut is None:
        return counts.astype(np.int64, copy=False)
    # Recount only the entries with a score below the cut; vectorised over the flat array.
    kept = (link_score >= score_cut).astype(np.int64)
    eff = np.zeros_like(counts, dtype=np.int64)
    for idx in range(len(counts)):
        o = int(offsets[idx])
        n = int(counts[idx])
        if n:
            eff[idx] = int(kept[o:o + n].sum())
    return eff


def format_links(pairs, max_show=4):
    if not pairs:
        return "[]"
    head = ", ".join(f"({i:>4d}, {s:.3f})" for (i, s) in pairs[:max_show])
    tail = "" if len(pairs) <= max_show else f", ... (+{len(pairs) - max_show} more)"
    return f"[{head}{tail}]"


def dump_event(event_idx, event, branches, args):
    tp_n = event[branches["tp_n"]]
    tp_o = event[branches["tp_o"]]
    tp_lidx = event[branches["tp_lidx"]]
    tp_lscore = event[branches["tp_lscore"]]
    trk_n = event[branches["trk_n"]]
    trk_o = event[branches["trk_o"]]
    trk_lidx = event[branches["trk_lidx"]]
    trk_lscore = event[branches["trk_lscore"]]

    n_tp = len(tp_n)
    n_trk = len(trk_n)

    if not args.no_dump:
        print(f"=== Event {event_idx} ===")
        show_tp = min(args.rows, n_tp)
        print(f"  TP -> Track  (first {show_tp} of {n_tp} TPs)")
        for i in range(show_tp):
            pairs = slice_links(i, tp_n, tp_o, tp_lidx, tp_lscore, args.score_cut)
            print(f"    TP  {i:>4d}  -> {format_links(pairs)}")

        show_trk = min(args.rows, n_trk)
        print(f"  Track -> TP  (first {show_trk} of {n_trk} tracks)")
        for i in range(show_trk):
            pairs = slice_links(i, trk_n, trk_o, trk_lidx, trk_lscore, args.score_cut)
            tag = "  # fake" if not pairs else ("  # merged" if len(pairs) > 1 else "")
            print(f"    Trk {i:>4d}  -> {format_links(pairs)}{tag}")

    tp_eff_n = per_entry_counts(tp_n, tp_o, tp_lscore, args.score_cut)
    trk_eff_n = per_entry_counts(trk_n, trk_o, trk_lscore, args.score_cut)

    stats = {
        "n_tp": n_tp,
        "tp_matched": int((tp_eff_n > 0).sum()),
        "tp_dup_matched": int((tp_eff_n > 1).sum()),
        "tp_unmatched": int((tp_eff_n == 0).sum()),
        "n_trk": n_trk,
        "trk_matched": int((trk_eff_n > 0).sum()),
        "trk_merged": int((trk_eff_n > 1).sum()),
        "trk_fake": int((trk_eff_n == 0).sum()),
        "n_links_tp_side": int(tp_eff_n.sum()),
        "score_sum": float(
            tp_lscore[tp_lscore >= args.score_cut].sum()
            if args.score_cut is not None
            else tp_lscore.sum()
        ),
    }

    line = (
        f"  stats: TPs {stats['n_tp']:>6d} "
        f"(matched {stats['tp_matched']:>6d} / dup-matched {stats['tp_dup_matched']:>4d} / "
        f"unmatched {stats['tp_unmatched']:>6d})\n"
        f"         Trks {stats['n_trk']:>5d} "
        f"(matched {stats['trk_matched']:>6d} / merged {stats['trk_merged']:>4d} / "
        f"fake {stats['trk_fake']:>5d})"
    )
    print(line)
    print()
    return stats


def main(argv=None):
    args = parse_args(argv)
    branches = make_branches(args.prefix)
    branch_list = list(branches.values())

    try:
        f = uproot.open(args.input)
    except OSError as exc:
        print(f"error: cannot open {args.input}: {exc}", file=sys.stderr)
        return 2
    if args.tree not in f:
        print(f"error: tree '{args.tree}' not found in {args.input}", file=sys.stderr)
        return 2
    tree = f[args.tree]

    missing = [b for b in branch_list if b not in tree]
    if missing:
        hint = (
            "trackingParticleTrackAssociationTablesTask"
            if args.prefix == ""
            else "the Phase-2 HLT NANO 'Val' flavour (e.g. NANO:@Phase2HLTVal)"
        )
        print(
            "error: required branches are missing from the input file:\n  "
            + "\n  ".join(missing)
            + f"\n\nThis usually means the {args.prefix or 'TP'}Assoc / "
            f"{args.prefix or ''}TrackAssoc tables were not produced. "
            f"Was {hint} scheduled?",
            file=sys.stderr,
        )
        return 2

    totals = {
        "events": 0, "n_tp": 0, "tp_matched": 0, "tp_dup_matched": 0, "tp_unmatched": 0,
        "n_trk": 0, "trk_matched": 0, "trk_merged": 0, "trk_fake": 0,
        "n_links_tp_side": 0, "score_sum": 0.0,
    }

    arrays = tree.arrays(branch_list, entry_stop=args.events, library="np")
    n_to_process = len(arrays[branch_list[0]])

    for ev in range(n_to_process):
        event = {b: arrays[b][ev] for b in branch_list}
        stats = dump_event(ev, event, branches, args)
        totals["events"] += 1
        for k in stats:
            totals[k] += stats[k]

    mean_score = totals["score_sum"] / totals["n_links_tp_side"] if totals["n_links_tp_side"] else float("nan")
    cut_str = "no cut" if args.score_cut is None else f"score >= {args.score_cut}"
    print(f"=== Summary over {totals['events']} events ({cut_str}) ===")
    print(
        f"  TPs    : total {totals['n_tp']:>8d}  "
        f"matched {totals['tp_matched']:>8d}  "
        f"dup-matched {totals['tp_dup_matched']:>6d}  "
        f"unmatched {totals['tp_unmatched']:>8d}"
    )
    print(
        f"  Tracks : total {totals['n_trk']:>8d}  "
        f"matched {totals['trk_matched']:>8d}  "
        f"merged      {totals['trk_merged']:>6d}  "
        f"fake      {totals['trk_fake']:>8d}"
    )
    print(f"  Total kept links (TP-side): {totals['n_links_tp_side']}, mean score: {mean_score:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
