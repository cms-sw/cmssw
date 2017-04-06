import FWCore.ParameterSet.Config as cms

from Validation.RecoHI.track_selectors_cff import *
from Validation.RecoMuon.NewMuonValidation_cff import *

# MuonAssociation labels; hit-by-hit matching only,MuonAssociator
#
import SimMuon.MCTruth.NewMuonAssociatorByHits_cfi
hiMABH = SimMuon.MCTruth.NewMuonAssociatorByHits_cfi.NewMuonAssociatorByHits.clone()
# DEFAULTS ###################################
#    EfficiencyCut_track = cms.double(0.),
#    PurityCut_track = cms.double(0.),
#    EfficiencyCut_muon = cms.double(0.),
#    PurityCut_muon = cms.double(0.),
#    includeZeroHitMuons = cms.bool(True),
#    acceptOneStubMatchings = cms.bool(False),
##############################################
hiMABH.tpTag = 'NEWcutsTpMuons'
#hiMABH.acceptOneStubMatchings = cms.bool(True) # this was the OLD setting
hiMABH.PurityCut_track = 0.75
hiMABH.PurityCut_muon = 0.75
#hiMABH.EfficiencyCut_track = 0.5 # maybe this could be added
#hiMABH.includeZeroHitMuons = False # maybe this could be added
################################################

# sim to tracker tracks, 
NEWtpToTkMuonAssociationHI = hiMABH.clone()
NEWtpToTkMuonAssociationHI.tracksTag = 'NEWcutsRecoTrkMuons'
NEWtpToTkMuonAssociationHI.UseTracker = True
NEWtpToTkMuonAssociationHI.UseMuon = False

# sim to sta, and sta:updatedAtVtx
NEWtpToStaMuonAssociationHI = hiMABH.clone()
NEWtpToStaMuonAssociationHI.tracksTag = 'standAloneMuons'
NEWtpToStaMuonAssociationHI.UseTracker = False
NEWtpToStaMuonAssociationHI.UseMuon = True

NEWtpToStaUpdMuonAssociationHI = hiMABH.clone()
NEWtpToStaUpdMuonAssociationHI.tracksTag = 'standAloneMuons:UpdatedAtVtx'
NEWtpToStaUpdMuonAssociationHI.UseTracker = False
NEWtpToStaUpdMuonAssociationHI.UseMuon = True

# sim to glb track 
NEWtpToGlbMuonAssociationHI = hiMABH.clone()
NEWtpToGlbMuonAssociationHI.tracksTag = 'globalMuons'
NEWtpToGlbMuonAssociationHI.UseTracker = True
NEWtpToGlbMuonAssociationHI.UseMuon = True


# Muon association sequences
# (some are commented out until timing is addressed)
NEWhiMuonAssociation_seq = cms.Sequence(
    NEWtpToTkMuonAssociationHI+
    NEWtpToStaMuonAssociationHI+
    NEWtpToStaUpdMuonAssociationHI+
    NEWtpToGlbMuonAssociationHI
    )

#----------------------------------------

from Validation.RecoMuon.histoParameters_cff import *

import Validation.RecoMuon.NewMuonTrackValidator_cfi
MTVhi = Validation.RecoMuon.NewMuonTrackValidator_cfi.NewMuonTrackValidator.clone()
MTVhi.label_tp_effic = cms.InputTag("NEWcutsTpMuons")
MTVhi.label_tp_fake = cms.InputTag("NEWcutsTpMuons")
MTVhi.maxPt = cms.double(100)

# MuonTrackValidator parameters
NEWtrkMuonTrackVMuonAssocHI = MTVhi.clone()
NEWtrkMuonTrackVMuonAssocHI.associatormap  = 'NEWtpToTkMuonAssociationHI'
NEWtrkMuonTrackVMuonAssocHI.label          = ['NEWcutsRecoTrkMuons']
NEWtrkMuonTrackVMuonAssocHI.muonHistoParameters = trkMuonHistoParameters

NEWglbMuonTrackVMuonAssocHI = MTVhi.clone()
NEWglbMuonTrackVMuonAssocHI.associatormap = 'NEWtpToGlbMuonAssociationHI'
NEWglbMuonTrackVMuonAssocHI.label           = ['globalMuons']
NEWglbMuonTrackVMuonAssocHI.muonHistoParameters = glbMuonHistoParameters

NEWstaMuonTrackVMuonAssocHI = MTVhi.clone()
NEWstaMuonTrackVMuonAssocHI.associatormap = 'NEWtpToStaMuonAssociationHI'
NEWstaMuonTrackVMuonAssocHI.label = ('standAloneMuons',)
NEWstaMuonTrackVMuonAssocHI.muonHistoParameters = staMuonHistoParameters

NEWstaUpdMuonTrackVMuonAssocHI = MTVhi.clone()
NEWstaUpdMuonTrackVMuonAssocHI.associatormap = 'NEWtpToStaUpdMuonAssociationHI'
NEWstaUpdMuonTrackVMuonAssocHI.label = ('standAloneMuons:UpdatedAtVtx',)
NEWstaUpdMuonTrackVMuonAssocHI.muonHistoParameters = staUpdMuonHistoParameters


# Muon validation sequences
NEWhiMuonValidation_seq = cms.Sequence(
    NEWtrkMuonTrackVMuonAssocHI+
    NEWstaMuonTrackVMuonAssocHI+
    NEWstaUpdMuonTrackVMuonAssocHI+
    NEWglbMuonTrackVMuonAssocHI
    )

# HI muon prevalidation
NEWhiRecoMuonPrevalidation = cms.Sequence(
    NEWcutsRecoTrkMuons
  * NEWcutsTpMuons
  * NEWhiMuonAssociation_seq
)

# HI muon validation sequence
NEWhiRecoMuonValidation = cms.Sequence( NEWhiMuonValidation_seq )    
