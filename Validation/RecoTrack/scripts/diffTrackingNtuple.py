#!/usr/bin/env python3

from __future__ import print_function
import argparse
import itertools
import collections

import Validation.RecoTrack.plotting.ntuple as ntuple

def body(opts, ev1, ev2, printTrack):
    print(ev1.eventIdStr())

    tracks1 = ev1.tracks()
    tracks2 = ev2.tracks()

    singleTrack = (opts.track is not None)
    if singleTrack:
        tracks1 = [tracks1[opts.track]]

    diff = ntuple.diffTrackListsGeneric(printTrack, tracks1, tracks2, ignoreAdditionalLst2=singleTrack)
    if diff.hasDifference():
        print(str(diff))
        print()

def inOrder(opts, ntpl1, ntpl2, *args, **kwargs):
    if opts.entry is not None:
        ev1 = ntpl1.getEvent(opts.entry)
        ev2 = ntpl2.getEvent(opts.entry)

        if ev1.eventId() != ev2.eventId():
            raise Exception("Events are out of order, entry %d file1 has %s and file %s. Use --outOfOrder option instead." % (ev1.entry(), ev1.eventIdStr(), ev2.eventIdStr()))

        body(opts, ev1, ev2, *args, **kwargs)
        return

    for i, (ev1, ev2) in enumerate(itertools.izip(ntpl1, ntpl2)):
        if opts.maxEvents >= 0 and i >= opts.maxEvents:
            break

        if ev1.eventId() != ev2.eventId():
            raise Exception("Events are out of order, entry %d file1 has %s and file %s. Use --outOfOrder option instead." % (ev1.entry(), ev1.eventIdStr(), ev2.eventIdStr()))

        body(opts, ev1, ev2, *args, **kwargs)


def outOfOrder(opts, ntpl1, ntpl2, *args, **kwargs):
    if opts.entry is not None:
        raise Exception("--entry does not make sense with --outOfOrder")

    events2 = collections.OrderedDict()
    for ev2 in ntpl2:
        events2[ev2.eventIdStr()] = ev2.entry()

    for i, ev1 in enumerate(ntpl1):
        if opts.maxEvents >= 0 and i >= opts.maxEvents:
            break

        if not ev1.eventId() in events2:
            print("-", ev1.eventIdStr())
            continue

        ev2 = ntpl2.getEvent(events2[ev1.eventIdStr()])
        events2.remove(ev1.eventId())

        body(opts, ev1, ev2, *args, **kwargs)


    for eventIdStr in events2.iterkeys():
        print("+", eventIdStr)

def main(opts):
    ntpl1 = ntuple.TrackingNtuple(opts.file1)
    ntpl2 = ntuple.TrackingNtuple(opts.file2)

    print("--- %s" % opts.file1)
    print("+++ %s" % opts.file2)

    printTrack = ntuple.TrackPrinter(trackingParticlePrinter=ntuple.TrackingParticlePrinter(parentage=False), diffForTwiki=opts.twiki)

    if opts.outOfOrder:
        outOfOrder(opts, ntpl1, ntpl2, printTrack)
    else:
        inOrder(opts, ntpl1, ntpl2, printTrack)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified diff of two TrackingNtuple files (clusters i.e. hits and TrackingParticles are assumed to be the same")
    parser.add_argument("file1", type=str,
                        help="File1")
    parser.add_argument("file2", type=str,
                        help="File2")
    parser.add_argument("--outOfOrder", action="store_true",
                        help="Set this if events are in different order in the files")
    parser.add_argument("--twiki", action="store_true",
                        help="Additional twiki-friendly diff formatting")
    parser.add_argument("--maxEvents", type=int, default=-1,
                        help="Maximum number of events to process (default: -1 for all events)")
    parser.add_argument("--entry", type=int, default=None,
                        help="Make diff only for this entry")
    parser.add_argument("--track", type=int,
                        help="Make diff only for this track (indexing from FILE1; only if --entry is given)")

    opts = parser.parse_args()

    if opts.track is not None and opts.entry is None:
        parser.error("With --track need --entry, which was not given")
    main(opts)
