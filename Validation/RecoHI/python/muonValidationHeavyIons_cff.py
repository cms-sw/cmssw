import FWCore.ParameterSet.Config as cms

from Validation.RecoHI.track_selectors_cff import *
from Validation.RecoMuon.muonValidation_cff import *

# MuonAssociation labels; hit-by-hit matching only,MuonAssociator
#
import SimMuon.MCTruth.MuonAssociatorByHits_cfi
hiMABH = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone(
# DEFAULTS ###################################
#    EfficiencyCut_track = cms.double(0.),
#    PurityCut_track = cms.double(0.),
#    EfficiencyCut_muon = cms.double(0.),
#    PurityCut_muon = cms.double(0.),
#    includeZeroHitMuons = cms.bool(True),
#    acceptOneStubMatchings = cms.bool(False),
##############################################
    tpTag = 'cutsTpMuons',
    #acceptOneStubMatchings = True # this was the OLD setting
    PurityCut_track = 0.75, 
    PurityCut_muon = 0.75
    #EfficiencyCut_track = 0.5 # maybe this could be added
    #EfficiencyCut_muon = 0.5 # maybe this could be added
    #includeZeroHitMuons = False # maybe this could be added
)
################################################

# sim to tracker tracks, 
tpToTkMuonAssociationHI = hiMABH.clone(
    tracksTag = 'cutsRecoTrkMuons',
    UseTracker = True,
    UseMuon = False
)
# sim to sta, and sta:updatedAtVtx
tpToStaMuonAssociationHI = hiMABH.clone(
    tracksTag = 'standAloneMuons',
    UseTracker = False,
    UseMuon = True
)
tpToStaUpdMuonAssociationHI = hiMABH.clone(
    tracksTag = 'standAloneMuons:UpdatedAtVtx',
    UseTracker = False,
    UseMuon = True
)
# sim to glb track 
tpToGlbMuonAssociationHI = hiMABH.clone(
    tracksTag = 'globalMuons',
    UseTracker = True,
    UseMuon = True
)

# Muon association sequences
# (some are commented out until timing is addressed)
hiMuonAssociation_seq = cms.Sequence(
    tpToTkMuonAssociationHI+
    tpToStaMuonAssociationHI+
    tpToStaUpdMuonAssociationHI+
    tpToGlbMuonAssociationHI
    )

#----------------------------------------

from Validation.RecoMuon.histoParameters_cff import *

import Validation.RecoMuon.MuonTrackValidator_cfi
MTVhi = Validation.RecoMuon.MuonTrackValidator_cfi.muonTrackValidator.clone()
MTVhi.label_tp_effic = cms.InputTag("cutsTpMuons")
MTVhi.label_tp_fake = cms.InputTag("cutsTpMuons")
MTVhi.maxPt = cms.double(100)

# MuonTrackValidator parameters
trkMuonTrackVMuonAssocHI = MTVhi.clone(
    associatormap  = 'tpToTkMuonAssociationHI',
    label          = ['cutsRecoTrkMuons'],
    muonHistoParameters = trkMuonHistoParameters
)
glbMuonTrackVMuonAssocHI = MTVhi.clone(
    associatormap = 'tpToGlbMuonAssociationHI',
    label           = ['globalMuons'],
    muonHistoParameters = glbMuonHistoParameters
)
staMuonTrackVMuonAssocHI = MTVhi.clone(
    associatormap = 'tpToStaMuonAssociationHI',
    label = ('standAloneMuons',),
    muonHistoParameters = staMuonHistoParameters
)
staUpdMuonTrackVMuonAssocHI = MTVhi.clone(
    associatormap = 'tpToStaUpdMuonAssociationHI',
    label = ('standAloneMuons:UpdatedAtVtx',),
    muonHistoParameters = staUpdMuonHistoParameters
)

# Muon validation sequences
hiMuonValidation_seq = cms.Sequence(
    trkMuonTrackVMuonAssocHI+
    staMuonTrackVMuonAssocHI+
    staUpdMuonTrackVMuonAssocHI+
    glbMuonTrackVMuonAssocHI
    )

# HI muon prevalidation
hiRecoMuonPrevalidation = cms.Sequence(
    cutsRecoTrkMuons
  * cutsTpMuons
  * hiMuonAssociation_seq
)

# HI muon validation sequence
hiRecoMuonValidation = cms.Sequence( hiMuonValidation_seq )    
