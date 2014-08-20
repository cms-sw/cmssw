import FWCore.ParameterSet.Config as cms

# make ntuples from RAW (ie. remove RECO)

from L1TriggerDPG.L1Ntuples.l1Ntuple_cfg import *

process.p.remove(process.l1RecoTreeProducer)
process.p.remove(process.l1MuonRecoTreeProducer)

# edit here
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

process.GlobalTag.globaltag = ''

readFiles.extend( [

    ] )
