import FWCore.ParameterSet.Config as cms

# Start by reading in all the fake conditions that are common to all Phase1 trackers
# Then do the 1 or two that are specific the R34F16
from SLHCUpgradeSimulations.Geometry.fakeConditions_Phase1_cff import *
siPixelFakeLorentzAngleESSource = cms.ESSource("SiPixelFakeLorentzAngleESSource",
        file = cms.FileInPath('SLHCUpgradeSimulations/Geometry/data/PhaseI/R34F16/PixelSkimmedGeometry_phase1.txt')
        )
es_prefer_fake_lorentz = cms.ESPrefer("SiPixelFakeLorentzAngleESSource","siPixelFakeLorentzAngleESSource")

# HCal
from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import *
from Configuration.StandardSequences.SimIdeal_cff import *
from Configuration.StandardSequences.Generator_cff import *
# use hardcoded values
es_hardcode.toGet.extend(['Gains', 'Pedestals', 'PedestalWidths', 'QIEData', 
                          'ElectronicsMap','ChannelQuality','RespCorrs',
                          'ZSThresholds',
                          'LutMetadata',
                          'L1TriggerObjects','TimeCorrs','PFCorrs','LUTCorrs',
                          'RecoParams'])
es_hardcode.H2Mode = cms.untracked.bool(False)
es_hardcode.SLHCMode = cms.untracked.bool(True)
es_prefer_hcalHardcode = cms.ESPrefer("HcalHardcodeCalibrations", "es_hardcode")

# Keep more stuff
#myOutputCommands.extend([
#    'keep *_hcalDigis_*_*', 'keep *_simHcalUnsuppressedDigis_*_*',
#    'keep *_towerMakerWithHO_*_*'
#    ])

# turn on test numbering
g4SimHits.Physics.type = 'SimG4Core/Physics/QGSP_BERT_EMV'
g4SimHits.HCalSD.TestNumberingScheme = True

# turn on SLHC topology
#HcalTopologyIdealEP.SLHCMode = cms.untracked.bool(True) -- In the Geometry file

#pgen.remove(genJetMET)

