#!/usr/bin/env python3

from __future__ import print_function
import ROOT

from Validation.RecoTrack.plotting.ntuple import *

# The purpose of this file is to demonstrate mainly the links between
# tracks, hits, seeds, and TrackingParticles.

def main():
    ntuple = TrackingNtuple("trackingNtuple.root")

    tot_nevents = 0
    tot_pv_ntracks = 0

    tot_ntracks = 0
    tot_hptracks = 0
    tot_fakes = 0
    tot_fakes_ninvalidhits = 0
    tot_fakes_ninvalidhits_missing = 0
    tot_fakes_ninvalidhits_inactive = 0
    tot_fakes_npixhits = 0
    tot_fakes_nstrhits = 0
    tot_fakes_npixhits_true = 0
    tot_fakes_nstrhits_true = 0
    tot_fakes_npixhits_tps = 0
    tot_duplicates = 0
    tot_secondaries = 0
    tot_novertex = 0

    tot_tps = 0
    tot_recoed = 0
    tot_tp_dups = 0
    tot_tp_nonrecohits = 0.0

    tot_pix = 0
    tot_pix_ntracks = 0
    tot_pix_nseeds = 0
    tot_pix_eloss = 0.0
    tot_pix_signal = 0
    tot_pix_itpu = 0
    tot_pix_ootpu = 0
    tot_pix_noise = 0
    tot_str = 0
    tot_str_ntracks = 0
    tot_str_nseeds = 0
    tot_str_signal = 0
    tot_str_itpu = 0
    tot_str_ootpu = 0
    tot_str_noise = 0
    tot_glu = 0
    tot_glu_nseeds = 0

    tot_seeds = 0
    tot_seeds_true = 0
    tot_seeds_lowPtTriplet = 0
    tot_seeds_pixelhits = 0
    tot_seeds_striphits = 0
    tot_seeds_gluedhits = 0
    tot_seeds_notrack = 0
    tot_track_seeds_true = 0

    for event in ntuple:
        #print "Event", event.entry()
        tot_nevents += 1

        vertices = event.vertices()
        tot_pv_ntracks += vertices[0].nTracks()

        # links from TrackingParticles to tracks
        tps = event.trackingParticles()
        tot_tps += len(tps)
        neff = 0
        ndups = 0
        for tp in tps:
            if tp.nMatchedTracks() >= 1:
                neff += 1
                if tp.nMatchedTracks() > 1:
                    ndups += 1

                # links from TrackingParticles to reco hits
                #print "TP", tp.index()
                #for hit in tp.hits():
                #    print " %s %d x %f y %f tof %f" % (hit.layerStr(), hit.detId(), hit.x(), hit.y(), hit.tof())

            if ntuple.hasHits():
                nhits = 0
                nhits_noreco = 0
                # links from TrackingParticles to SimHits
                for hit in tp.simHits():
                    nhits += 1
                    # links from SimHits to RecHits
                    if hit.nRecHits() == 0:
                        nhits_noreco += 1
                if nhits > 0:
                    tot_tp_nonrecohits += float(nhits_noreco)/nhits

        tot_recoed += neff
        tot_tp_dups += ndups

        # links from tracks to TrackingParticles
        tracks = event.tracks()
        ntracks = len(tracks) # also tracks.size() works
        tot_ntracks += ntracks
        nfakes = 0
        nfakes_invalidhits = 0
        nfakes_pixhits = 0
        nfakes_strhits = 0
        nfakes_pixhits_true = 0
        nfakes_strhits_true = 0
        nfakes_pixhits_tps = 0
        ndups = 0
        nsecondaries = 0
        for track in tracks:
            if track.isHP():
                tot_hptracks += 1

            # link from track to vertex
            if not track.vertex().isValid():
                tot_novertex += 1

            if track.nMatchedTrackingParticles() == 0:
                #print "Track", track.index(), " is fake"

                nfakes += 1


                # links from tracks to hits
                if ntuple.hasHits():
                    pix_simTrkIds = set()

                    #for hit in track.hits():
                    #    print hit.layerStr()

                    for hit in track.pixelHits():
                        nfakes_pixhits += 1
                        if hit.nSimHits() >= 1:
                            # links from hits to SimHits
                            nfakes_pixhits_true += 1
                            for simHit in hit.simHits():
                                pix_simTrkIds.add(simHit.trackingParticle().index())
                    nfakes_pixhits_tps += len(pix_simTrkIds)

                    for hit in track.stripHits():
                        nfakes_strhits += 1
                        if hit.nSimHits() >= 1:
                            nfakes_strhits_true += 1

                    for hit in track.invalidHits():
                        tot_fakes_ninvalidhits += 1
                        if hit.type() == InvalidHit.Type.missing:
                            tot_fakes_ninvalidhits_missing += 1
                        elif hit.type() == InvalidHit.Type.inactive:
                            tot_fakes_ninvalidhits_inactive += 1
            else:
                for tpInfo in track.matchedTrackingParticleInfos():
                    tp = tpInfo.trackingParticle()
                    if tp.nMatchedTracks() > 1:
                        ndups += 1
                        break

                    # TrackinParticle <-> TrackingVertex links
                    if tp.parentVertex().nSourceTrackingParticles() > 0:
                        nsecondaries += 1

        tot_fakes += nfakes
        tot_fakes_npixhits += nfakes_pixhits
        tot_fakes_nstrhits += nfakes_strhits
        tot_fakes_npixhits_true += nfakes_pixhits_true
        tot_fakes_nstrhits_true += nfakes_strhits_true
        tot_fakes_npixhits_tps += nfakes_pixhits_tps
        tot_duplicates += ndups
        tot_secondaries += nsecondaries

        # hits
        if ntuple.hasHits():
            # links from hits to tracks
            for hit in event.pixelHits():
                tot_pix += 1
                # hit -> track links
                for track in hit.tracks():
                    tot_pix_ntracks += 1

                if hit.simType() == HitSimType.Signal:
                    tot_pix_signal += 1
                elif hit.simType() == HitSimType.ITPileup:
                    tot_pix_itpu += 1
                elif hit.simType() == HitSimType.OOTPileup:
                    tot_pix_ootpu += 1
                elif hit.simType() == HitSimType.Noise:
                    tot_pix_noise += 1
                else:
                    raise Exception("Got unknown hit sim type value %d" % hit.simType())

                # hit -> SimHit links
                for simHit in hit.simHits():
                    tot_pix_eloss += simHit.eloss()

            for hit in event.stripHits():
                tot_str += 1
                # hit -> track links
                for track in hit.tracks():
                    tot_str_ntracks += 1

                if hit.simType() == HitSimType.Signal:
                    tot_str_signal += 1
                elif hit.simType() == HitSimType.ITPileup:
                    tot_str_itpu += 1
                elif hit.simType() == HitSimType.OOTPileup:
                    tot_str_ootpu += 1
                elif hit.simType() == HitSimType.Noise:
                    tot_str_noise += 1
                else:
                    raise Exception("Got unknown hit sim type value %d" % hit.simType())

            tot_glu += len(event.gluedHits())

            # hit -> seed links
            if ntuple.hasSeeds():
                for hit in event.pixelHits():
                    for seed in hit.seeds():
                        tot_pix_nseeds += 1
                for hit in event.stripHits():
                    for seed in hit.seeds():
                        tot_str_nseeds += 1
                for hit in event.gluedHits():
                    for seed in hit.seeds():
                        tot_glu_nseeds += 1

        # seeds
        if ntuple.hasSeeds():
            seeds = event.seeds()
            nseeds = len(seeds)
            tot_seeds += nseeds

            # finding seeds of a particular iteration
            tot_seeds_lowPtTriplet += seeds.nSeedsForAlgo(5) # = lowPtTripletStep

            # links from seeds to TrackingParticles and tracks
            ntrue = 0
            for seed in seeds:
                if seed.nMatchedTrackingParticles() >= 1:
                    ntrue += 1

                if not seed.track().isValid():
                    tot_seeds_notrack += 1
            tot_seeds_true = ntrue

            # links from seeds to hits
            for seed in seeds:
                for hit in seed.hits():
                    if isinstance(hit, PixelHit):
                        tot_seeds_pixelhits += 1
                    elif isinstance(hit, StripHit):
                        tot_seeds_striphits += 1
                    elif isinstance(hit, GluedHit):
                        tot_seeds_gluedhits += 1

            # links from tracks to seeds
            ntracktrue = 0
            for track in tracks:
                seed = track.seed()
                if seed.nMatchedTrackingParticles() >= 1:
                    ntracktrue += 1
            tot_track_seeds_true += ntracktrue


    print("Processed %d events" % tot_nevents)
    print("On average %f tracks from PV" % (float(tot_pv_ntracks)/tot_nevents))
    print("On average %f TrackingParticles" % (float(tot_tps)/tot_nevents))
    print(" with %f %% reconstructed" % (float(tot_recoed)/tot_tps * 100))
    print("  of which %f %% were reconstructed at least twice" % (float(tot_tp_dups)/tot_recoed * 100))
    print(" average fraction of non-recoed hits %f %%" % (tot_tp_nonrecohits/tot_tps * 100))
    print("On average %f tracks" % (float(tot_ntracks)/tot_nevents))
    print(" with %f %% being high purity" % (float(tot_hptracks)/tot_ntracks * 100))
    print(" with %f %% were not used in any vertex fit" % (float(tot_novertex)/tot_ntracks * 100))
    print(" with %f %% of true tracks being secondaries" % (float(tot_secondaries)/(tot_ntracks-tot_fakes) * 100))
    print(" with fake rate %f %%" % (float(tot_fakes)/tot_ntracks * 100))
    if tot_fakes_npixhits > 0:
        print("  on average %f %% of pixel hits are true" % (float(tot_fakes_npixhits_true)/tot_fakes_npixhits * 100))
        print("   pixel hits from %f TrackingParticles/track" % (float(tot_fakes_npixhits_tps)/tot_fakes))
        print("  on average %f %% of strip hits are true" % (float(tot_fakes_nstrhits_true)/tot_fakes_nstrhits * 100))
        print("  on average %f %% of hits are invalid" % (float(tot_fakes_ninvalidhits)/(tot_fakes_npixhits+tot_fakes_nstrhits) * 100))
        print("   of those, %f %% were missing" % (float(tot_fakes_ninvalidhits_missing)/tot_fakes_ninvalidhits * 100))
        print("   of those, %f %% were invalid" % (float(tot_fakes_ninvalidhits_inactive)/tot_fakes_ninvalidhits * 100))
    print(" with duplicate rate %f %%" % (float(tot_duplicates)/tot_ntracks * 100))
    if tot_seeds > 0:
        print(" of which %f %% had a true seed" % (float(tot_track_seeds_true)/tot_ntracks * 100))
        print("On average %f seeds" % (float(tot_seeds)/tot_nevents))
        print(" of which %f were from lowPtTripletStep" % (float(tot_seeds_lowPtTriplet)/tot_nevents))
        print(" of which %f %% were true" % (float(tot_seeds_true)/tot_seeds * 100))
        print(" of which %f %% did not produce a track" % (float(tot_seeds_notrack)/tot_seeds * 100))
        print(" on average %f pixel hits / seed" % (float(tot_seeds_pixelhits)/tot_seeds))
        print(" on average %f strip hits / seed" % (float(tot_seeds_striphits)/tot_seeds))
        print(" on average %f glued hits / seed" % (float(tot_seeds_gluedhits)/tot_seeds))
    if tot_pix > 0:
        print("On average %f pixel hits" % (float(tot_pix)/tot_nevents))
        print(" on average %f tracks per hit" % (float(tot_pix_ntracks)/tot_pix))
        print(" on average %f seeds per hit" % (float(tot_pix_nseeds)/tot_pix))
        print(" on average %f energy loss per hit" % (tot_pix_eloss/tot_pix))
        print(" on average %f %% from hard scatter" % (float(tot_pix_signal)/tot_pix*100))
        print(" on average %f %% from in-time pileup" % (float(tot_pix_itpu)/tot_pix*100))
        print(" on average %f %% from out-of-time pileup" % (float(tot_pix_ootpu)/tot_pix*100))
        print(" on average %f %% from noise" % (float(tot_pix_noise)/tot_pix*100))
        print("On average %f strip hits" % (float(tot_str)/tot_nevents))
        print(" on average %f tracks per hit" % (float(tot_str_ntracks)/tot_str))
        print(" on average %f seeds per hit" % (float(tot_str_nseeds)/tot_str))
        print(" on average %f %% from hard scatter" % (float(tot_str_signal)/tot_str*100))
        print(" on average %f %% from in-time pileup" % (float(tot_str_itpu)/tot_str*100))
        print(" on average %f %% from out-of-time pileup" % (float(tot_str_ootpu)/tot_str*100))
        print(" on average %f %% from noise" % (float(tot_str_noise)/tot_str*100))
        print("On average %f glued hits" % (float(tot_glu)/tot_nevents))
        print(" on average %f seeds per hit" % (float(tot_glu_nseeds)/tot_glu))


if __name__ == "__main__":
    main()
