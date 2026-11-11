import argparse
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)
import pandas as pd
import ROOT
import numpy as np

FIELDS = [
    "det",
    "iscalib",
    "ncellsMapped",
    "nHits",
    "nHitsIT",
    "nHitsOOT",
    "nHitsUnknown",
    "nHitsRejected",
    "en",
    "enIT",
    "enOOT",
    "enUnknown",
    "enRejected",
]

def main():
    parser = argparse.ArgumentParser(description="Plot HGCal digi accumulator counters.")
    parser.add_argument(
        "-i",
        "--input",
        help="DIGI step file",
    )
    parser.add_argument(
        "--acc-monitor-branch",
        default="hgcaldigiHGCalSimHitAccumulatorMonitors_hgCalSimHitAccumulator_HGCalSimHitAccumulatorCounters_TEST",
        help="ROOT branch name for the accumulator monitor collection",
    )
    args = parser.parse_args()

    #loop over the events
    chain = ROOT.TChain("Events")
    chain.AddFile(args.input)
    chain.SetAlias("acc_monitor", args.acc_monitor_branch)
    rows = []
    for event_index in range(chain.GetEntries()):
        chain.GetEntry(event_index)
        for monitor in chain.acc_monitor.product():
            rows.append(
                [event_index]
                + [getattr(monitor, field) for field in FIELDS]
            )

    #convert to a dataframe
    data = pd.DataFrame(rows, columns=["event"] + FIELDS)
    data['f_unknown'] = data["nHitsUnknown"] / data['nHits']
    data['f_IT'] = data["nHitsIT"]/(data["nHitsIT"]+data["nHitsOOT"]+data["nHitsRejected"])
    data['fen_IT'] = data["enIT"]/(data["enIT"]+data["enOOT"]+data["enRejected"])

    for field,xlab,bins in [
        ('f_unknown','Fraction of unknown hits',np.linspace(0,1,100)),
        ('f_IT','Fraction of in-time hits',np.linspace(0,1,100)),
        ('fen_IT','Fraction of in-time energy',np.linspace(0,1,100))
    ]:

        fig, ax = plt.subplots(figsize=(6, 6))
        for (det, iscalib), group in data.groupby(["det", "iscalib"]):

            x = group[field].values
            is_real = ~np.isnan( x )
            n_real = is_real.sum()
            print(det,iscalib,field,n_real,len(x))
            if n_real==0: continue
            ax.hist(
                x[is_real],
                histtype='step',
                bins=bins,
                linewidth=2,
                label=f"det={det}, iscalib={iscalib}",
            )

        ax.set_xlabel("Event")
        ax.set_ylabel(xlab)
        ax.grid(True)
        ax.legend(loc="best", fontsize=10)
        fig.tight_layout()

        filename = f"{field}.png"
        fig.savefig(filename)
        plt.close(fig)

if __name__ == "__main__":
    main()