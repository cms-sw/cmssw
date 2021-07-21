#!/usr/bin/env python3

from __future__ import print_function
import ROOT

from Validation.RecoTrack.plotting.ntuple import *

# The purpose of this file is to demonstrate the printout features of the ntuple library

def main():
    ntuple = TrackingNtuple("trackingNtuple.root")

    printTrack = TrackPrinter(trackingParticlePrinter=TrackingParticlePrinter())
    printTP = TrackingParticlePrinter(trackPrinter=TrackPrinter())

    for event in ntuple:
        print("Event", event.eventIdStr())
        print("Fake tracks")
        for track in event.tracks():
            if track.nMatchedTrackingParticles() == 0:
                printTrack(track)
                print()

        print("Duplicate tracks")
        for tp in event.trackingParticles():
            if tp.nMatchedTracks() >= 2:
                printTP(tp)
                print()
        print()

        if event.entry() >= 1:
            break

if __name__ == "__main__":
    main()
