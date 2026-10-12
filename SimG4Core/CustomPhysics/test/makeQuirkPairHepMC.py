#!/usr/bin/env python3
# Writes a HepMC3 ascii file with one quirk pair per event at the origin.
import argparse
import math
import sys

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Quirk pair gun as a HepMC3 ascii file')
parser.add_argument("--output", type=str, default="quirkPair.hepmc3", help="output file")
parser.add_argument("--nEvents", type=int, default=10, help="number of events")
parser.add_argument("--mass", type=float, default=250., help="quirk mass in GeV")
parser.add_argument("--pdgId", type=int, default=17, help="PDG code of the quirk")
parser.add_argument("--p1", type=float, nargs=3, default=[100., -60., 1.], help="quirk momentum in GeV (pz = 0 is dropped by the SimG4Core Generator eta cut)")
parser.add_argument("--p2", type=float, nargs=3, default=[50., 60., 1.], help="antiquirk momentum in GeV")
args = parser.parse_args()

def fourVector(p, m):
    return [p[0], p[1], p[2], math.sqrt(p[0]**2 + p[1]**2 + p[2]**2 + m**2)]

def particle(pid, vertex, pdg, v, m, status):
    return "P %d %d %d %.10e %.10e %.10e %.10e %.10e %d\n" % (pid, vertex, pdg, v[0], v[1], v[2], v[3], m, status)

q1, q2 = fourVector(args.p1, args.mass), fourVector(args.p2, args.mass)
pair = [a + b for a, b in zip(q1, q2)]
mPair = math.sqrt(max(pair[3]**2 - pair[0]**2 - pair[1]**2 - pair[2]**2, 0.))

with open(args.output, "w") as f:
    f.write("HepMC::Version 3.02.06\nHepMC::Asciiv3-START_EVENT_LISTING\n")
    for i in range(args.nEvents):
        f.write("E %d 1 3 @ 0 0 0 0\n" % (i + 1))
        f.write("U GEV MM\n")
        # the pair system (status 4, not simulated) decays at the origin into the two quirks
        f.write(particle(1, 0, 32, pair, mPair, 4))
        f.write("V -1 0 [1] @ 0 0 0 0\n")
        f.write(particle(2, -1, args.pdgId, q1, args.mass, 1))
        f.write(particle(3, -1, -args.pdgId, q2, args.mass, 1))
    f.write("HepMC::Asciiv3-END_EVENT_LISTING\n")
