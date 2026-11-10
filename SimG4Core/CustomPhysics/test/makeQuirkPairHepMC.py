#!/usr/bin/env python3
# Writes a HepMC2 IO_GenEvent file with one quirk pair per event at the origin.
import argparse
import math
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Quirk pair gun as a HepMC2 ascii file')
parser.add_argument("--output", type=str, default="quirkPair.hepmc", help="output file")
parser.add_argument("--nEvents", type=int, default=10, help="number of events")
parser.add_argument("--mass", type=float, default=250., help="quirk mass in GeV")
parser.add_argument("--pdgId", type=int, default=17, help="PDG code of the quirk")
parser.add_argument("--p1", type=float, nargs=3, default=[100., -60., 1.], help="quirk momentum in GeV (pz = 0 is dropped by the SimG4Core Generator eta cut)")
parser.add_argument("--p2", type=float, nargs=3, default=[50., 60., 1.], help="antiquirk momentum in GeV")
args = parser.parse_args()

def particle(barcode, pdg, p, m):
    e = math.sqrt(p[0]**2 + p[1]**2 + p[2]**2 + m**2)
    pt = math.hypot(p[0], p[1])
    theta = math.atan2(pt, p[2])
    phi = math.atan2(p[1], p[0])
    return "P %d %d %.10e %.10e %.10e %.10e %.10e 1 %.10e %.10e 0 0\n" % (barcode, pdg, p[0], p[1], p[2], e, m, theta, phi)

with open(args.output, "w") as f:
    f.write("\nHepMC::Version 2.06.09\nHepMC::IO_GenEvent-START_EVENT_LISTING\n")
    for i in range(args.nEvents):
        f.write("E %d -1 -1.0 -1.0 -1.0 0 -1 1 0 0 0 1 1.0\n" % (i + 1))
        f.write("U GEV MM\n")
        f.write("V -1 0 0 0 0 0 0 2 0\n")
        f.write(particle(1, args.pdgId, args.p1, args.mass))
        f.write(particle(2, -args.pdgId, args.p2, args.mass))
    f.write("HepMC::IO_GenEvent-END_EVENT_LISTING\n")
