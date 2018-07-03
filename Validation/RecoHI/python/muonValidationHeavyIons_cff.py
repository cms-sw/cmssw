import FWCore.ParameterSet.Config as cms

from Validation.RecoHI.track_selectors_cff import *
from Validation.RecoMuon.muonValidation_cff import *

# MuonAssociation labels; hit-by-hit matching only,MuonAssociator
#
import SimMuon.MCTruth.MuonAssociatorByHits_cfi
hiMABH = SimMuon.MCTruth.MuonAssociatorByHits_cfi.muonAssociatorByHits.clone()
# DEFAULTS ###################################
#    EfficiencyCut_track = cms.double(0.),
#    PurityCut_track = cms.double(0.),
#    EfficiencyCut_muon = cms.double(0.),
#    PurityCut_muon = cms.double(0.),
#    includeZeroHitMuons = cms.bool(True),
#    acceptOneStubMatchings = cms.bool(False),
##############################################
hiMABH.tpTag = 'cutsTpMuons'
#hiMABH.acceptOneStubMatchings = cms.bool(True) # this was the OLD setting
hiMABH.PurityCut_track = 0.75
hiMABH.PurityCut_muon = 0.75
#hiMABH.EfficiencyCut_track = 0.5 # maybe this could be added
#hiMABH.EfficiencyCut_muon = 0.5 # maybe this could be added
#hiMABH.includeZeroHitMuons = False # maybe this could be added
################################################

# sim to tracker tracks, 
tpToTkMuonAssociationHI = hiMABH.clone()
tpToTkMuonAssociationHI.tracksTag = 'cutsRecoTrkMuons'
tpToTkMuonAssociationHI.UseTracker = True
tpToTkMuonAssociationHI.UseMuon = False

# sim to sta, and sta:updatedAtVtx
tpToStaMuonAssociationHI = hiMABH.clone()
tpToStaMuonAssociationHI.tracksTag = 'standAloneMuons'
tpToStaMuonAssociationHI.UseTracker = False
tpToStaMuonAssociationHI.UseMuon = True

tpToStaUpdMuonAssociationHI = hiMABH.clone()
tpToStaUpdMuonAssociationHI.tracksTag = 'standAloneMuons:UpdatedAtVtx'
tpToStaUpdMuonAssociationHI.UseTracker = False
tpToStaUpdMuonAssociationHI.UseMuon = True

# sim to glb track 
tpToGlbMuonAssociationHI = hiMABH.clone()
tpToGlbMuonAssociationHI.tracksTag = 'globalMuons'
tpToGlbMuonAssociationHI.UseTracker = True
tpToGlbMuonAssociationHI.UseMuon = True


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
trkMuonTrackVMuonAssocHI = MTVhi.clone()
trkMuonTrackVMuonAssocHI.associatormap  = 'tpToTkMuonAssociationHI'
trkMuonTrackVMuonAssocHI.label          = ['cutsRecoTrkMuons']
trkMuonTrackVMuonAssocHI.muonHistoParameters = trkMuonHistoParameters

glbMuonTrackVMuonAssocHI = MTVhi.clone()
glbMuonTrackVMuonAssocHI.associatormap = 'tpToGlbMuonAssociationHI'
glbMuonTrackVMuonAssocHI.label           = ['globalMuons']
glbMuonTrackVMuonAssocHI.muonHistoParameters = glbMuonHistoParameters

staMuonTrackVMuonAssocHI = MTVhi.clone()
staMuonTrackVMuonAssocHI.associatormap = 'tpToStaMuonAssociationHI'
staMuonTrackVMuonAssocHI.label = ('standAloneMuons',)
staMuonTrackVMuonAssocHI.muonHistoParameters = staMuonHistoParameters

staUpdMuonTrackVMuonAssocHI = MTVhi.clone()
staUpdMuonTrackVMuonAssocHI.associatormap = 'tpToStaUpdMuonAssociationHI'
staUpdMuonTrackVMuonAssocHI.label = ('standAloneMuons:UpdatedAtVtx',)
staUpdMuonTrackVMuonAssocHI.muonHistoParameters = staUpdMuonHistoParameters


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
