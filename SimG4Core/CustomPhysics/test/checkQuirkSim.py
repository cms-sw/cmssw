#!/usr/bin/env python3
# Checks the SimTrack/SimVertex/SimHit bookkeeping of quirk events:
# exactly two quirk SimTracks, their hits per collection, secondaries attached.
import argparse
import sys
import ROOT
from DataFormats.FWLite import Events, Handle

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Quirk SIM bookkeeping check')
parser.add_argument("inputFile", type=str, help="SIM output file")
parser.add_argument("--pdgId", type=int, default=17, help="quirk PDG code")
args = parser.parse_args()

hitColls = ["TrackerHitsPixelBarrelLowTof", "TrackerHitsPixelBarrelHighTof",
            "TrackerHitsPixelEndcapLowTof", "TrackerHitsPixelEndcapHighTof",
            "TrackerHitsTIBLowTof", "TrackerHitsTIBHighTof", "TrackerHitsTOBLowTof", "TrackerHitsTOBHighTof",
            "TrackerHitsTIDLowTof", "TrackerHitsTIDHighTof", "TrackerHitsTECLowTof", "TrackerHitsTECHighTof",
            "MuonDTHits", "MuonCSCHits", "MuonRPCHits"]
caloColls = ["EcalHitsEB", "EcalHitsEE", "HcalHits"]

ROOT.gSystem.Load("libFWCoreFWLite")
ROOT.FWLiteEnabler.enable()
trkH = Handle("std::vector<io_v1::SimTrack>")
vtxH = Handle("std::vector<io_v1::SimVertex>")
hitH = Handle("std::vector<io_v1::PSimHit>")
caloH = Handle("std::vector<io_v1::PCaloHit>")

ok = True
for iev, ev in enumerate(Events(args.inputFile)):
    ev.getByLabel("g4SimHits", trkH)
    ev.getByLabel("g4SimHits", vtxH)
    trks = trkH.product()
    vtxs = vtxH.product()
    quirks = [t for t in trks if abs(t.type()) == args.pdgId]
    qids = set(t.trackId() for t in quirks)
    # secondaries whose parent is a quirk
    nDaughters = 0
    for t in trks:
        iv = t.vertIndex()
        if iv >= 0 and iv < len(vtxs) and not vtxs[iv].noParent() and vtxs[iv].parentIndex() in qids:
            nDaughters += 1
    print("event %d: %d SimTracks, %d SimVertices, %d quirk SimTracks %s, %d tracks from quirk vertices" %
          (iev, len(trks), len(vtxs), len(quirks), sorted(qids), nDaughters))
    for q in quirks:
        m = q.momentum()
        print("   quirk id %d pdg %d p=(%.1f, %.1f, %.1f) GeV genpart %d vertex %d" %
              (q.trackId(), q.type(), m.px(), m.py(), m.pz(), q.genpartIndex(), q.vertIndex()))
    if len(quirks) != 2 or len(qids) != 2:
        ok = False
    for c in hitColls:
        ev.getByLabel("g4SimHits", c, hitH)
        if not hitH.isValid():
            continue
        hits = hitH.product()
        qh = [h for h in hits if h.trackId() in qids]
        if len(qh) == 0:
            continue
        eloss = sum(h.energyLoss() for h in qh) * 1e6
        tofs = [h.tof() for h in qh]
        print("   %-32s quirk hits %4d (of %5d)  sum Eloss %9.1f keV  tof %.2f-%.2f ns" %
              (c, len(qh), len(hits), eloss, min(tofs), max(tofs)))
    for c in caloColls:
        ev.getByLabel("g4SimHits", c, caloH)
        if caloH.isValid():
            hits = caloH.product()
            qh = [h for h in hits if h.geantTrackId() in qids]
            print("   %-32s quirk hits %4d (of %5d)" % (c, len(qh), len(hits)))
print("RESULT", "OK" if ok else "FAIL: not exactly two quirk SimTracks per event")
sys.exit(0 if ok else 1)
