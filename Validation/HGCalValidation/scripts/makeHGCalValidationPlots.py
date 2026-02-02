#!/usr/bin/env python3

import os
import argparse
import datetime

from Validation.RecoTrack.plotting.validation import SeparateValidation, SimpleValidation, SimpleSample
from Validation.HGCalValidation.HGCalValidator_cff import hgcalValidator
import Validation.RecoTrack.plotting.plotting as plotting

simClustersIters = [hgcalValidator.label_SimClustersLevel.value(), "ticlSimTracksters"]

hitCalLabel = 'hitCalibration'
hitValLabel = 'hitValidation'
layerClustersLabel = 'layerClusters'
trackstersLabel = 'tracksters'
trackstersWithEdgesLabel = 'trackstersWithEdges'
candidatesLabel = 'candidates'
simLabel = 'simulation'
allLabel = 'all'
ticlVersions = [5]
ticlVersion = 5
collection_choices = [allLabel]
collection_choices.extend([hitCalLabel]+[hitValLabel]+[layerClustersLabel]+[trackstersLabel]+[trackstersWithEdgesLabel]+[candidatesLabel]+[simLabel])
tracksters = []

def _write_top_index(output_dir, entries):
    """Create a single top-level index.html linking to per-flavour reports."""
    index_path = os.path.join(output_dir, "index.html")
    lines = []
    lines.append("<!doctype html>")
    lines.append("<html lang='en'>")
    lines.append("<head>")
    lines.append("  <meta charset='utf-8'>")
    lines.append("  <meta name='viewport' content='width=device-width, initial-scale=1'>")
    lines.append("  <title>HGCal validation plots</title>")
    lines.append("  <style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem}ul{line-height:1.8}</style>")
    lines.append("</head>")
    lines.append("<body>")
    lines.append("  <h1>HGCal validation plots</h1>")
    lines.append("  <ul>")
    for label, rel_index in entries:
        lines.append(f"    <li><a href='{rel_index}'>{label}</a></li>")
    lines.append("  </ul>")
    lines.append("</body>")
    lines.append("</html>")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(opts):

    drawArgs = {}
    extendedFlag = False
    if opts.no_ratio:
        drawArgs["ratio"] = False
    if opts.separate:
        drawArgs["separate"] = True
    if opts.png:
        drawArgs["saveFormat"] = ".png"
    if opts.extended:
        extendedFlag = True
    if opts.verbose:
        plotting.verbose = True

    collections = [c.strip() for c in opts.collection.split(",")]
    for coll in collections:
        if coll not in collection_choices:
            raise ValueError(f"Unknown collection '{coll}'. Valid options: {collection_choices}.")

    import ROOT

    def _discover_subdirs(dqm_file, dqm_base):
        """Return direct subdirectory names under dqm_base in the DQM ROOT file."""
        f = ROOT.TFile.Open(dqm_file)
        if not f or f.IsZombie():
            print(f"Warning: could not open DQM file {dqm_file} to discover collections")
            return set()
        d = f.GetDirectory(dqm_base)
        if not d:
            print(f"Warning: base directory '{dqm_base}' not found in {dqm_file}")
            f.Close()
            return set()
        names = set()
        keys = d.GetListOfKeys()
        if keys:
            key = keys.First()
            while key:
                obj = key.ReadObj()
                if obj and obj.InheritsFrom("TDirectory"):
                    names.add(key.GetName())
                key = keys.After(key)
        f.Close()
        return names

    output_root = opts.outputDir[0]
    os.makedirs(output_root, exist_ok=True)

    filenames = [(f, f.replace(".root", "")) for f in opts.files]

    # Hardcoded Offline and HLT HGCalValidator base folders
    flavours = (
        ("offline", "DQMData/Run 1/HGCAL/Run summary/HGCalValidator/"),
        ("hlt", "DQMData/Run 1/HLT/Run summary/HGCAL/HGCalValidator/"),
    )

    # Import once, then reload each pass to avoid accumulating plot definitions
    import importlib
    import Validation.HGCalValidation.hgcalPlots as hgcalPlots

    top_index_entries = []

    for prefix, dqm_base in flavours:
        out_dir = os.path.join(output_root, prefix)
        os.makedirs(out_dir, exist_ok=True)

        # Reset plotters for each pass
        hgcalPlots = importlib.reload(hgcalPlots)

        # Override the HGCalValidator base (used at least by TICL candidates)
        if hasattr(hgcalPlots, "set_hgcVal_dqm"):
            hgcalPlots.set_hgcVal_dqm(dqm_base)
        else:
            hgcalPlots.hgcVal_dqm = dqm_base

        sample = SimpleSample(prefix, opts.html_sample, filenames)

        val = SimpleValidation([sample], out_dir, nProc=opts.jobs)
        if opts.separate:
            val = SeparateValidation([sample], out_dir)

        htmlReport = val.createHtmlReport(
            validationName=f"{opts.html_validation_name[0]} ({prefix})"
        )

        # Discover which collections exist under this base in the DQM file
        discovered_subdirs = _discover_subdirs(opts.files[0], dqm_base)
        present_lower = {n.lower(): n for n in discovered_subdirs}

        # Start from known default collections, but keep only the ones that actually exist
        trackstersIters = []

        # Add any other folder that looks like a trackster collection (case-insensitive match)
        for name in sorted(discovered_subdirs, key=str.lower):
            if ("trackster" in name.lower()) or ("candidate" in name.lower()) and name not in trackstersIters:
                trackstersIters.append(name)

        # Detect candidates presence similarly
        has_candidates = any("candidat" in n.lower() for n in discovered_subdirs)

        if opts.verbose:
            print(f"Discovered under {dqm_base}: {sorted(discovered_subdirs, key=str.lower)}")
            print(f"Using tracksters collections for {prefix}: {trackstersIters}")
            print(f"Candidates present for {prefix}: {has_candidates}")

        # layerClusters
        def plot_LC():
            hgclayclus = [hgcalPlots.hgcalLayerClustersPlotter]
            hgcalPlots.append_hgcalLayerClustersPlots(hgcalValidator.label_layerClustersPlots.value(), "Layer Clusters", extendedFlag)
            val.doPlots(hgclayclus, plotterDrawArgs=drawArgs)

        # simClusters
        def plot_SC():
            hgcsimclus = [hgcalPlots.hgcalSimClustersPlotter]
            for i_iter in simClustersIters:
                hgcalPlots.append_hgcalSimClustersPlots(i_iter, i_iter)
            val.doPlots(hgcsimclus, plotterDrawArgs=drawArgs)

        # tracksters
        def plot_Tst():
            hgctrackster = [hgcalPlots.hgcalTrackstersPlotter]
            for tracksterCollection in trackstersIters:
                print("Searching for tracksters collection in DQM files: ", tracksterCollection)
                hgcalPlots.append_hgcalTrackstersPlots(tracksterCollection, tracksterCollection)
            val.doPlots(hgctrackster, plotterDrawArgs=drawArgs)

        # trackstersWithEdges
        def plot_TstEdges():
            plot_Tst()
            for tracksterCollection in trackstersIters:
                hgctracksters = [hgcalPlots.create_hgcalTrackstersPlotter(sample.files(), tracksterCollection, tracksterCollection)]
                val.doPlots(hgctracksters, plotterDrawArgs=drawArgs)

        # caloParticles
        def plot_CP():
            particletypes = {"pion-":"-211", "pion+":"211", "pion0": "111",
                             "muon-": "-13", "muon+":"13",
                             "electron-": "-11", "electron+": "11", "photon": "22",
                             "kaon0L": "310", "kaon0S": "130",
                             "kaon-": "-321", "kaon+": "321"}
            hgcaloPart = [hgcalPlots.hgcalCaloParticlesPlotter]
            for i_part, i_partID in particletypes.items():
                hgcalPlots.append_hgcalCaloParticlesPlots(sample.files(), i_partID, i_part)
            val.doPlots(hgcaloPart, plotterDrawArgs=drawArgs)

        # hitValidation
        def plot_hitVal():
            hgchit = [hgcalPlots.hgcalHitPlotter]
            hgcalPlots.append_hgcalHitsPlots('HGCalSimHitsV', "Simulated Hits")
            hgcalPlots.append_hgcalHitsPlots('HGCalRecHitsV', "Reconstruced Hits")
            hgcalPlots.append_hgcalDigisPlots('HGCalDigisV', "Digis")
            val.doPlots(hgchit, plotterDrawArgs=drawArgs)

        # hitCalibration
        def plot_hitCal():
            hgchitcalib = [hgcalPlots.hgcalHitCalibPlotter]
            val.doPlots(hgchitcalib, plotterDrawArgs=drawArgs)

        # candidates
        def plotCand():
            if not has_candidates:
                print(f"Skipping candidates for {prefix}: no folder matching 'candidate' found under {dqm_base}")
                return
            candidate_labels = [n for n in discovered_subdirs if "candidate" in n.lower()]
            ticlcand = [hgcalPlots.hgcalTICLCandPlotter(candidate_labels)]
            val.doPlots(ticlcand, plotterDrawArgs=drawArgs)

        plotDict = {
            hitCalLabel: [plot_hitCal],
            hitValLabel: [plot_hitVal],
            layerClustersLabel: [plot_LC],
            trackstersLabel: [plot_Tst],
            trackstersWithEdgesLabel: [plot_TstEdges],
            simLabel: [plot_SC, plot_CP],
            candidatesLabel: [plotCand],
        }

        if (allLabel not in collections):
            for coll in collections:
                for task in plotDict[coll]:
                    task()
        else:
            for label in plotDict:
                if (label == trackstersLabel):
                    continue  # already run in trackstersWithEdges
                for task in plotDict[label]:
                    task()

        if opts.no_html:
            print("Plots created into directory '%s'." % out_dir)
        else:
            htmlReport.write()
            print("Plots and HTML report created into directory '%s'." % out_dir)
            top_index_entries.append((prefix, f"{prefix}/index.html"))

    if (not opts.no_html) and top_index_entries:
        _write_top_index(output_root, top_index_entries)
        print("Top-level index written to '%s'." % os.path.join(output_root, "index.html"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create set of HGCal validation plots from one or more DQM files.")
    parser.add_argument("files", metavar="file", type=str, nargs="+",
                        default = "DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root",
                        help="DQM file to plot the validation plots from")
    parser.add_argument("-o", "--outputDir", type=str, default=["plots"], nargs="+",
                        help="Plot output directories (default: 'plots'")
    parser.add_argument("--subdirprefix", type=str, default=["HLT","offline"], nargs="+",
                        help="Prefix for subdirectories inside outputDir (default: 'HLT, offline')")
    parser.add_argument("--no-ratio", action="store_true", default = False,
                        help="Disable ratio pads")
    parser.add_argument("--separate", action="store_true", default = False,
                        help="Save all plots separately instead of grouping them")
    parser.add_argument("--png", action="store_true", default = True,
                        help="Save plots in PNG instead of PDF")
    parser.add_argument("--no-html", action="store_true", default = False,
                        help="Disable HTML page generation")
    parser.add_argument("--html-sample", default=os.environ.get('CMSSW_VERSION', 'CMSSW'),
                        help="Sample name for HTML page generation (default: CMSSW version)")
    parser.add_argument("--html-validation-name", type=str, default=["TICL Validation",""], nargs="+",
                        help="Validation name for HTML page generation (enters to <title> element) (default 'TICL Validation')")
    parser.add_argument("--collection", default=layerClustersLabel,
                        help="Choose output plots collections among possible choices: {collection_choices}")
    parser.add_argument("--extended", action="store_true", default = False,
                        help="Include extended set of plots (e.g. bunch of distributions; default off)")
    parser.add_argument("--jobs", default=0, type=int,
                        help="Number of jobs to run in parallel for generating plots. Default is 0 i.e. run number of cpu cores jobs.")
    parser.add_argument("--ticlv", choices=ticlVersions, default=5, type=int,
                        help="TICL Version. Default 5.")
    parser.add_argument("--verbose", action="store_true", default = False,
                        help="Be verbose")

    opts = parser.parse_args()

    for f in opts.files:
        if not os.path.exists(f):
            parser.error("DQM file %s does not exist" % f)
    if len(opts.files) == 0:
        parser.error("No DQM files specified")
    else:
        main(opts)
