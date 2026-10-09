import FWCore.ParameterSet.Config as cms


def customiseHGCalOnly(process):
    ## avoid crash due to missing track collection in HGCalOnly workflows
    process.trackingParticleGsfTrackAssociation.ignoremissingtrackcollection = cms.untracked.bool(True)
    return process
