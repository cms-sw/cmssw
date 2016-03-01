#!/usr/bin/env python

import ROOT

# The purpose of this file is to demonstrate mainly the links between
# tracks, hits, seeds, and TrackingParticles.

def main():
    inputFile = ROOT.TFile.Open("trackingNtuple.root")
    tree = inputFile.Get("trackingNtuple/tree")
    entries = tree.GetEntriesFast()

    tot_nevents = 0
    tot_pv_ntracks = 0

    tot_ntracks = 0
    tot_hptracks = 0
    tot_fakes = 0
    tot_fakes_ninvalidhits = 0
    tot_fakes_npixhits = 0
    tot_fakes_nstrhits = 0
    tot_fakes_npixhits_true = 0
    tot_fakes_nstrhits_true = 0
    tot_fakes_npixhits_tps = 0
    tot_duplicates = 0

    tot_tps = 0
    tot_recoed = 0
    tot_tp_dups = 0

    tot_seeds = 0
    tot_seeds_true = 0
    tot_seeds_lowPtTriplet = 0
    tot_track_seeds_true = 0

    for jentry in xrange(entries):
        # get the next tree in the chain and verify
        ientry = tree.LoadTree( jentry )
        if ientry < 0: break
        # copy next entry into memory and verify
        nb = tree.GetEntry( jentry )
        if nb <= 0: continue

        #print "Event", jentry
        tot_nevents += 1

        tot_pv_ntracks += tree.vtx_trkIdx[0].size()

        # links from TrackingParticles to tracks
        ntps = tree.sim_px.size()
        tot_tps += ntps
        neff = 0
        ndups = 0
        for itp in xrange(ntps):
            if tree.sim_trkIdx[itp].size() >= 1:
                neff += 1
                if tree.sim_trkIdx[itp].size() > 1:
                    ndups += 1
        tot_recoed += neff
        tot_tp_dups += ndups

        # links from tracks to TrackingParticles
        ntracks = tree.trk_px.size()
        tot_ntracks += ntracks
        nfakes = 0
        nfakes_invalidhits = 0
        nfakes_pixhits = 0
        nfakes_strhits = 0
        nfakes_pixhits_true = 0
        nfakes_strhits_true = 0
        nfakes_pixhits_tps = 0
        ndups = 0
        for itrack in xrange(ntracks):
            if track.isHP():
                tot_hptracks += 1

            if tree.trk_simIdx[itrack].size() == 0:
                #print "Track", itrack, " is fake"

                nfakes += 1

                # links from tracks to hits
                if hasattr(tree, "pix_nSimTrk"):
                    pix_simTrkIds = set()

                    for ihit in tree.trk_pixelIdx[itrack]:
                        if ihit == -1:
                            nfakes_invalidhits += 1
                            continue
                        nfakes_pixhits += 1
                        if tree.pix_nSimTrk[ihit] >= 1:
                            nfakes_pixhits_true += 1
                        pix_simTrkIds.add(tree.pix_simTrkIdx[ihit]) # currently the index of only the "first" matched TP is stored
                    nfakes_pixhits_tps += len(pix_simTrkIds)

                    for ihit in tree.trk_stripIdx[itrack]:
                        if ihit == -1:
                            nfakes_invalidhits += 1
                            continue
                        nfakes_strhits += 1
                        if tree.str_nSimTrk[ihit] >= 1:
                            nfakes_strhits_true += 1
            else:
                for itp in tree.trk_simIdx[itrack]:
                    if tree.sim_trkIdx[itp].size() > 1:
                        ndups += 1
                        break
        tot_fakes += nfakes
        tot_fakes_ninvalidhits += nfakes_invalidhits
        tot_fakes_npixhits += nfakes_pixhits
        tot_fakes_nstrhits += nfakes_strhits
        tot_fakes_npixhits_true += nfakes_pixhits_true
        tot_fakes_nstrhits_true += nfakes_strhits_true
        tot_fakes_npixhits_tps += nfakes_pixhits_tps
        tot_duplicates += ndups

        # seeds
        if hasattr(tree, "see_simIdx"):
            nseeds = tree.see_simIdx.size()
            tot_seeds += nseeds

            # finding seeds of a particular iteration
            for ioffset, offset in enumerate(tree.see_offset):
                if tree.see_algo[offset] == 5: # = lowPtTripletStep
                    next_offset = tree.see_offset[ioffset+1] if ioffset < tree.see_offset.size() else tree.see_algo.size()
                    tot_seeds_lowPtTriplet += next_offset - offset
                    break

            # links from seeds to TrackingParticles
            ntrue = 0
            for iseed in xrange(nseeds):
                if tree.see_simIdx[iseed].size() >= 1:
                    ntrue += 1
            tot_seeds_true = ntrue

            # links from tracks to seeds
            ntracktrue = 0
            for itrack in xrange(ntracks):
                iseed = tree.trk_seedIdx[itrack]
                if tree.see_simIdx[iseed].size() >= 1:
                    ntracktrue += 1
            tot_track_seeds_true += ntracktrue


    print "Processed %d events" % tot_nevents
    print "On average %f tracks from PV" % (float(tot_pv_ntracks)/tot_nevents)
    print "On average %f TrackingParticles" % (float(tot_tps)/tot_nevents)
    print " with %f %% reconstructed" % (float(tot_recoed)/tot_tps * 100)
    print "  of which %f %% were reconstructed at least twice" % (float(tot_tp_dups)/tot_recoed * 100)
    print "On average %f tracks" % (float(tot_ntracks)/tot_nevents)
    print " with %f %% being high purity" % (float(tot_hptracks)/tot_ntracks * 100)
    print " with fake rate %f %%" % (float(tot_fakes)/tot_ntracks * 100)
    if tot_fakes_npixhits > 0:
        print "  on average %f %% of pixel hits are true" % (float(tot_fakes_npixhits_true)/tot_fakes_npixhits * 100)
        print "   pixel hits from %f TrackingParticles/track" % (float(tot_fakes_npixhits_tps)/tot_fakes)
        print "  on average %f %% of strip hits are true" % (float(tot_fakes_nstrhits_true)/tot_fakes_nstrhits * 100)
        print "  on average %f %% of hits are invalid" % (float(tot_fakes_ninvalidhits)/(tot_fakes_npixhits+tot_fakes_nstrhits) * 100)
    print " with duplicate rate %f %%" % (float(tot_duplicates)/tot_ntracks * 100)
    if tot_seeds > 0:
        print " of which %f %% had a true seed" % (float(tot_track_seeds_true)/tot_ntracks * 100)
        print "On average %f seeds" % (float(tot_seeds)/tot_nevents)
        print " of which %f were from lowPtTripletStep" % (float(tot_seeds_lowPtTriplet)/tot_nevents)
        print " of which %f %% were true" % (float(tot_seeds_true)/tot_seeds * 100)


if __name__ == "__main__":
    main()
