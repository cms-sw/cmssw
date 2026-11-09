import argparse
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)
import pandas as pd
import ROOT

FIELDS = [
    "det",
    "iscalib",
    "ncellsMapped",
    "nHits",
    "nHitsIT",
    "nHitsOOT",
    "nHitsUnknown",
    "nHitsRejected",
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
    print(data.head())

    for field in FIELDS:
        if field in ("det", "iscalib"):
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        for (det, iscalib), group in data.groupby(["det", "iscalib"]):
            ax.plot(
                group["event"],
                group[field],
                linewidth=1,
                label=f"det={det}, iscalib={iscalib}",
            )

        ax.set_xlabel("Event")
        ax.set_ylabel(field)
        ax.grid(True)
        ax.legend(loc="best", fontsize=10)
        fig.tight_layout()

        filename = f"{field}.png"
        fig.savefig(filename)
        plt.close(fig)

if __name__ == "__main__":
    main()