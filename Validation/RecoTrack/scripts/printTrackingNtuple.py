#!/usr/bin/env python

import argparse

import Validation.RecoTrack.plotting.ntuple as ntuple


def findEvent(ntpl, event):
    eventId = event.split(":")
    if len(eventId) != 3:
        raise Exception("Malformed eventId %s, is not run:lumi:event" % eventId)
    eventId = (int(eventId[0]), int(eventId[1]), int(eventId[2]))

    for ev in ntpl:
        if ev.eventId() == eventId:
            return ev

    raise Exception("Did not find event %s from file %s" % (eventId, ntpl.file().GetPath()))

def main(opts):
    if opts.track is None and opts.trackingParticle is None and opts.seed is None and opts.pixelHit is None and opts.stripHit is None:
        return

    ntpl = ntuple.TrackingNtuple(opts.file)

    if opts.entry is not None:
        event = ntpl.getEvent(opts.entry)
        print event.eventIdStr()
    elif opts.event is not None:
        event = findEvent(ntpl, opts.event)
        print "Entry %d" % event.entry()

    hasHits = ntpl.hasHits()
    hasSeeds = ntpl.hasSeeds()

    if not hasSeeds and opts.seed is not None:
        print "Ntuple %s does not have seeds saved!" % opts.file
        return
    if not hasHits and (opts.pixelHit is not None or opts.stripHit is not None):
        print "Ntuple %s does not have hits saved!" % opts.file
        return

    seedArgs = dict(hits=hasHits, bestMatchingTrackingParticle=hasHits)
    trackArgs = dict(hits=hasHits, bestMatchingTrackingParticle=hasHits)
    tpArgs = dict(hits=hasHits, bestMatchingTrack=hasHits)
    if not hasSeeds:
        trackArgs["seedPrinter"] = None
        tpArgs["seedPrinter"] = None
    elif not hasHits:
        trackArgs["seedPrinter"] = ntuple.SeedPrinter(**seedArgs)
        tpArgs["seedPrinter"] = ntuple.SeedPrinter(**seedArgs)

    printSeed = ntuple.SeedPrinter(trackingParticles=True, trackingParticlePrinter=ntuple.TrackingParticlePrinter(**tpArgs), **seedArgs)
    printTrack = ntuple.TrackPrinter(trackingParticlePrinter=ntuple.TrackingParticlePrinter(**tpArgs), **trackArgs)
    printTrackingParticle = ntuple.TrackingParticlePrinter(trackPrinter=ntuple.TrackPrinter(**trackArgs), **tpArgs)

    if opts.track is not None:
        trk = event.tracks()[opts.track]
        printTrack(trk)

    if opts.trackingParticle is not None:
        tp = event.trackingParticles()[opts.trackingParticle]
        printTrackingParticle(tp)

    if opts.seed is not None:
        seeds = event.seeds()
        if opts.seedIteration is not None:
            seed = seeds.seedForAlgo(getattr(ntuple.Algo, opts.seedIteration), opts.seed)
        else:
            seed = seeds[opts.seed]
        printSeed(seed)

    if opts.pixelHit is not None:
        hit = event.pixelHits()[opts.pixelHit]
        print "Pixel hit %d tracks" % opts.pixelHit
        for t in hit.tracks():
            printTrack(t)
        if hasSeeds:
            print "Pixel hit %d seeds" % opts.pixelHit
            for t in hit.seeds():
                printSeed(s)

    if opts.stripHit is not None:
        hit = event.stripHits()[opts.stripHit]
        print "Strip hit %d tracks" % opts.stripHit
        for t in hit.tracks():
            printTrack(t)
        if hasSeeds:
            print "Strip hit %d seeds" % opts.stripHit
            for t in hit.seeds():
                printSeed(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print information from a TrackingNtuple file")
    parser.add_argument("file", type=str,
                        help="Input file")

    parser.add_argument("--entry", type=int,
                        help="Entry in a file to print information for (conflicts with --event)")
    parser.add_argument("--event", type=str,
                        help="Event in a file to print information for, in a format run:lumi:event (conflicts with --entry)")

    parser.add_argument("--track", type=int,
                        help="Index of a track to print information for")
    parser.add_argument("--trackingParticle", type=int,
                        help="Index of a TrackingParticle to print information for")
    parser.add_argument("--seed", type=int,
                        help="Index of a seed to print information for. If --seedIteration is specified, the index is within the iteration. Without --seedIteration it is used as a global index.")
    parser.add_argument("--seedIteration", type=str,
                        help="Seed iteration, used optionally with --seed")
    parser.add_argument("--pixelHit", type=int,
                        help="Index of a pixel hit")
    parser.add_argument("--stripHit", type=int,
                        help="Index of a strip hit")

    opts = parser.parse_args()

    if opts.entry is None and opts.event is None:
        parser.error("Need either --entry or --event, neither was given")
    if opts.entry is not None and opts.event is not None:
        parser.error("--entry and --event conflict, please give only one of them")

    main(opts)


