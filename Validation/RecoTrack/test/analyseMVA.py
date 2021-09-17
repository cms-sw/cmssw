#!/usr/bin/env python3

from builtins import range
import array
import collections
import itertools

import ROOT

from Validation.RecoTrack.plotting.ntuple import *
from Validation.RecoTrack.plotting.ntuplePlotting import *

# This file is a sketch of a simple analysis for MVA debugging

def selectMVA(track):
    mva = track.mva()
    return mva > 0.35 and mva < 0.6

def selectTrue(track):
    return track.nMatchedTrackingParticles() > 0

def selectFake(track):
    return track.nMatchedTrackingParticles() == 0

def main():
    ROOT.TH1.AddDirectory(False)

    ntuple_new = TrackingNtuple("trackingNtuple.root")
    ntuple_old = TrackingNtuple("trackingNtuple_oldMVA.root")

    common = dict(
        #ratio=True,
    )
    common_ylog = dict(
        ylog=True,
        ymin=0.5,
    )
    common_ylog.update(common)

    opts = dict(
        mva = dict(xtitle="MVA", **common_ylog),
        pt  = dict(xtitle="p_{T}", xlog=True, **common_ylog),
        eta = dict(xtitle="#eta", **common),
        relpterr = dict(xtitle="p_{T} error / p_{T}", **common_ylog),
        absdxy = dict(xtitle="|d_{xy}(BS)|", **common_ylog),
        absdz = dict(xtitle="|d_{z}(BS)|", **common_ylog),
        absdxypv = dict(xtitle="|d_{xy}(closest PV)|", **common_ylog),
        absdzpv = dict(xtitle="|d_{z}(closest PV)|", **common_ylog),
        nhits = dict(xtitle="hits", **common),
        nlayers = dict(xtitle="layers", **common),
        nlayers3D = dict(xtitle="3D layers", **common),
        nlayersLost = dict(xtitle="lost layers", **common),
        minlost = dict(xtitle="min(inner, outer) lost layers", **common_ylog),
        lostmidfrac = dict(xtitle="(lost hits) / (lost + valid hits)", **common),
        ndof = dict(xtitle="ndof", **common),
        chi2 = dict(xtitle="chi2/ndof", **common_ylog),
        chi2_1Dmod = dict(xtitle="chi2/ndof with 1D modification", **common_ylog),
    )

    if True:
        histos_new = histos(ntuple_new)
        histos_old = histos(ntuple_old)
        drawMany("newMVA_vs_oldMVA", [histos_old, histos_new], opts=opts)

    if True:
        histos_new = histos(ntuple_new, selector=selectMVA)
        histos_old = histos(ntuple_old, selector=selectMVA)
        drawMany("newMVA_vs_oldMVA_mvaselected", [histos_old, histos_new], opts=opts)

    if True:
        histos_new = histos(ntuple_new, selector=selectTrue)
        histos_old = histos(ntuple_old, selector=selectTrue)
        drawMany("newMVA_vs_oldMVA_true", [histos_old, histos_new], opts=opts)

    if True:
        histos_new = histos(ntuple_new, selector=lambda t: selectTrue(t) and selectMVA(t))
        histos_old = histos(ntuple_old, selector=lambda t: selectTrue(t) and selectMVA(t))
        drawMany("newMVA_vs_oldMVA_true_mvaselected", [histos_old, histos_new], opts=opts)

    if True:
        histos_new = histos(ntuple_new, selector=selectFake)
        histos_old = histos(ntuple_old, selector=selectFake)
        drawMany("newMVA_vs_oldMVA_fake", [histos_old, histos_new], opts=opts)

    if True:
        histos_new = histos(ntuple_new, selector=lambda t: selectFake(t) and selectMVA(t))
        histos_old = histos(ntuple_old, selector=lambda t: selectFake(t) and selectMVA(t))
        drawMany("newMVA_vs_oldMVA_fake_mvaselected", [histos_old, histos_new], opts=opts)

    if True:
        (histos_old, histos_new) = histos2(ntuple_old, ntuple_new, selectMVA)
        drawMany("newMVA_vs_oldMVA_mvaSelectedNew", [histos_old, histos_new], opts=opts)

    if True:
        (histos_old, histos_new) = histos2(ntuple_old, ntuple_new, lambda t: selectTrue(t) and selectMVA(t))
        drawMany("newMVA_vs_oldMVA_true_mvaSelectedNew", [histos_old, histos_new], opts=opts)

    if True:
        (histos_old, histos_new) = histos2(ntuple_old, ntuple_new, lambda t: selectFake(t) and selectMVA(t))
        drawMany("newMVA_vs_oldMVA_fake_mvaSelectedNew", [histos_old, histos_new], opts=opts)


def makeHistos():
    h = collections.OrderedDict()
    def addTH(name, *args, **kwargs):
        _h = ROOT.TH1F(name, name, *args)
        if kwargs.get("xlog", False):
            axis = _h.GetXaxis()
            bins = axis.GetNbins()
            minLog10 = math.log10(axis.GetXmin())
            maxLog10 = math.log10(axis.GetXmax())
            width = (maxLog10-minLog10)/bins
            new_bins = array.array("d", [0]*(bins+1))
            new_bins[0] = 10**minLog10
            mult = 10**width
            for i in range(1, bins+1):
                new_bins[i] = new_bins[i-1]*mult
            axis.Set(bins, new_bins)
        h[name] = _h

    addTH("mva", 80, -1, 1)
    addTH("pt", 40, 0.1, 1000, xlog=True)
    addTH("eta", 60, -3, 3)
    addTH("relpterr", 20, 0, 1)

    addTH("absdxy", 50, 0, 1)
    addTH("absdz", 30, 0, 15)
    addTH("absdxypv", 50, 0., 0.5)
    addTH("absdzpv", 20, 0, 1)

    addTH("nhits", 41, -0.5, 40.5)
    addTH("nlayers", 26, -0.5, 25.5)
    addTH("nlayers3D", 26, -0.5, 25.5)
    addTH("nlayersLost", 6, -0.5, 5.5)
    addTH("minlost", 6, -0.5, 5.5)
    addTH("lostmidfrac", 20, 0, 1)

    addTH("ndof", 20, 0, 20)
    addTH("chi2", 40, 0, 20)
    addTH("chi2_1Dmod", 40, 0, 20)

    return h

def fillHistos(h, track):
    h["mva"].Fill(track.mva())
    h["pt"].Fill(track.pt())
    h["eta"].Fill(track.eta())
    h["ndof"].Fill(track.ndof())
    h["nlayers"].Fill(track.nPixelLay()+track.nStripLay())
    h["nlayers3D"].Fill(track.n3DLay())
    h["nlayersLost"].Fill(track.nLostLay())
    h["chi2"].Fill(track.nChi2())
    h["chi2_1Dmod"].Fill(track.nChi2_1Dmod())
    h["relpterr"].Fill(track.ptErr()/track.pt())
    h["nhits"].Fill(track.nValid())
    h["minlost"].Fill(min(track.nInnerLost(), track.nOuterLost()))
    h["lostmidfrac"].Fill(track.nInvalid() / (track.nValid() + track.nInvalid()))

    h["absdxy"].Fill(abs(track.dxy()))
    h["absdz"].Fill(abs(track.dz()))
    h["absdxypv"].Fill(abs(track.dxyClosestPV()))
    h["absdzpv"].Fill(abs(track.dzClosestPV()))

def histos(ntuple, selector=None):
    h = makeHistos()

    for event in ntuple:
        for track in event.tracks():
            if selector is not None and not selector(track):
                continue
            fillHistos(h, track)

    return h

def histos2(ntuple1, ntuple2, selector2=None):
    # assume the two ntuples have the very same tracks except possibly
    # for their parameters
    h1 = makeHistos()
    h2 = makeHistos()

    for (event1, event2) in itertools.izip(ntuple1, ntuple2):
        #print event1.eventIdStr(), event2.eventIdStr()
        if event1.eventId() != event2.eventId():
            raise Exception("Inconsistent events %s != %s" % (event1.eventIdStr(), event2.eventIdStr()))

        for (track1, track2) in itertools.izip(event1.tracks(), event2.tracks()):

            if selector2 is not None and not selector2(track2):
                continue

            fillHistos(h1, track1)
            fillHistos(h2, track2)
    return (h1, h2)


if __name__ == "__main__":
    main()
