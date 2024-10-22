from __future__ import print_function
# This customise file provides an example (in the form of holes in
# BPix L1-L2 and L3-L3) on how to select a subset of generalTracks
# (e.g. by phi and eta) and setup various MTV instances for those
# (selected tracks, built tracks, and seeds in this case). The naming
# of DQM folders is consistent with an example in trackingCompare.py
import FWCore.ParameterSet.Config as cms

def customiseMTVForBPix123Holes(process):
    from Validation.RecoTrack.cutsRecoTracks_cfi import cutsRecoTracks as _cutsRecoTracks
    import math
    _minPhi = process.trackValidatorTrackingOnly.histoProducerAlgoBlock.minPhi.value()
    _maxPhi = process.trackValidatorTrackingOnly.histoProducerAlgoBlock.maxPhi.value()
    _nPhi = process.trackValidatorTrackingOnly.histoProducerAlgoBlock.nintPhi.value()
    _binPhi = (_maxPhi - _minPhi) / _nPhi
    process.generalTracksL1L2 = _cutsRecoTracks.clone(
        minLayer = 0,
        quality = [],
        minRapidity = -1.0, # also eta < -1 is affected, but let's start with this
        minPhi=_minPhi+_binPhi*14, maxPhi=_minPhi+_binPhi*19) # ~0.7 .. ~0.2
    process.generalTracksL2L3 = process.generalTracksL1L2.clone(
        minRapidity = -0.9, maxRapidity = 2,
        minPhi=_minPhi+_binPhi*33, maxPhi=_minPhi+_binPhi + 2*math.pi) # ~2.6 .. ~3.3
    
    print("L1L2 %f %f" % (process.generalTracksL1L2.minPhi.value(), process.generalTracksL1L2.maxPhi.value()))
    print("L2L3 %f %f" % (process.generalTracksL2L3.minPhi.value(), process.generalTracksL2L3.maxPhi.value()))
    
    from CommonTools.RecoAlgos.trackingParticleRefSelector_cfi import trackingParticleRefSelector as _trackingParticleRefSelector
    process.trackingParticlesL1L2 = _trackingParticleRefSelector.clone(
        signalOnly = False,
        chargedOnly = False,
        tip = 1e5,
        lip = 1e5,
        minRapidity = process.generalTracksL1L2.minRapidity.value(),
        maxRapidity = process.generalTracksL1L2.maxRapidity.value(),
        ptMin = 0,
        minPhi = process.generalTracksL1L2.minPhi.value(),
        maxPhi = process.generalTracksL1L2.maxPhi.value(),
    )
    process.trackingParticlesL2L3 = process.trackingParticlesL1L2.clone(
        minRapidity = process.generalTracksL2L3.minRapidity.value(),
        maxRapidity = process.generalTracksL2L3.maxRapidity.value(),
        minPhi = process.generalTracksL2L3.minPhi.value(),
        maxPhi = process.generalTracksL2L3.maxPhi.value(),
    )
    process.tracksPreValidationTrackingOnly += (
        process.trackingParticlesL1L2 +
        process.trackingParticlesL2L3 +
        process.generalTracksL1L2 +
        process.generalTracksL2L3
    )

    process.trackValidatorTrackingOnlyL1L2 = process.trackValidatorTrackingOnly.clone(
        dirName = process.trackValidatorTrackingOnly.dirName.value().replace("Track/", "TrackL1L2/"),
        label_tp_effic = "trackingParticlesL1L2",
        label_tp_effic_refvector = True,
        label = ["generalTracksL1L2"],
    )
    process.trackValidatorTrackingOnlyL2L3 = process.trackValidatorTrackingOnlyL1L2.clone(
        dirName = process.trackValidatorTrackingOnlyL1L2.dirName.value().replace("L1L2", "L2L3"),
        label_tp_effic = "trackingParticlesL2L3",
        label = ["generalTracksL2L3"],
    )
    process.trackValidatorsTrackingOnly += (
        process.trackValidatorTrackingOnlyL1L2 +
        process.trackValidatorTrackingOnlyL2L3
    )
    for trkColl in process.trackValidatorTrackingOnly.label:
        if "ByAlgoMask" in trkColl: continue
        if "Pt09" in trkColl and not trkColl in ["generalTracksPt09", "cutsRecoTracksPt09Hp"]: continue
        if trkColl != "generalTracks":
            selL1L2 = getattr(process, trkColl).clone(src="generalTracksL1L2")
            selL2L3 = getattr(process, trkColl).clone(src="generalTracksL2L3")
            if "Pt09" in trkColl:
                selL1L2Name = trkColl.replace("Pt09", "Pt09L1L2")
                selL2L3Name = trkColl.replace("Pt09", "Pt09L2L3")
            else:
                selL1L2Name = trkColl.replace("cutsRecoTracks", "cutsRecoTracksL1L2")
                selL2L3Name = trkColl.replace("cutsRecoTracks", "cutsRecoTracksL2L3")
            setattr(process, selL1L2Name, selL1L2)
            setattr(process, selL2L3Name, selL2L3)
            process.tracksPreValidationTrackingOnly += (selL1L2+selL2L3)
            process.trackValidatorTrackingOnlyL1L2.label.append(selL1L2Name)
            process.trackValidatorTrackingOnlyL2L3.label.append(selL2L3Name)
            
    for midfix in ["Building", "Seeding"]:
        label = "trackValidator%sTrackingOnly" % midfix
        mtv = getattr(process, label)
        mtvL1L2 = mtv.clone(
            dirName = mtv.dirName.value()[:-1] + "L1L2/",
            label_tp_effic = "trackingParticlesL1L2",
            label_tp_effic_refvector = True,
            label = [],
            mvaLabels = cms.PSet(),
            doMVAPlots = False,
        )
        mtvL2L3 = mtvL1L2.clone(
            dirName = mtvL1L2.dirName.value().replace("L1L2", "L2L3"),
            label_tp_effic = "trackingParticlesL2L3",
        )
        setattr(process, label+"L1L2", mtvL1L2)
        setattr(process, label+"L2L3", mtvL2L3)
        process.trackValidatorsTrackingOnly += (
            mtvL1L2 +
            mtvL2L3
        )
        for trkColl in mtv.label:
            selL1L2 = process.generalTracksL1L2.clone(src=trkColl)
            selL2L3 = process.generalTracksL2L3.clone(src=trkColl)
            selL1L2Name = trkColl+"L1L2"
            selL2L3Name = trkColl+"L2L3"
            setattr(process, selL1L2Name, selL1L2)
            setattr(process, selL2L3Name, selL2L3)
            process.tracksPreValidationTrackingOnly += (selL1L2+selL2L3)
    
            mtvL1L2.label.append(selL1L2Name)
            mtvL2L3.label.append(selL2L3Name)
    
    return process
