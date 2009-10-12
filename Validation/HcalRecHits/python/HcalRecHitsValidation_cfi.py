import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff import *
from Configuration.StandardSequences.Simulation_cff import *
from Configuration.StandardSequences.MixingNoPileUp_cff import *
from Configuration.StandardSequences.Reconstruction_cff import *
from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import *
GlobalTag.globaltag = 'IDEAL_31X::All'


from DQMServices.Core.DQM_cfg import *

maxEvents = cms.untracked.PSet(
   input = cms.untracked.int32(-1)
)
source = cms.Source("PoolSource",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring('file:/afs/cern.ch/cms/data/CMSSW/Validation/HcalDigis/data/1_8_0/simhits_pi+30_hcalonly.root')
)

DQM.collectorHost = ''

#--- DigiToRaw
from Configuration.StandardSequences.DigiToRaw_cff import *
ecalPacker.Label = 'simEcalDigis'
ecalPacker.InstanceEB = 'ebDigis'
ecalPacker.InstanceEE = 'eeDigis'
ecalPacker.labelEBSRFlags = "simEcalDigis:ebSrFlags"
ecalPacker.labelEESRFlags = "simEcalDigis:eeSrFlags"
#--- RawToDigi
from Configuration.StandardSequences.RawToDigi_cff  import *
hcalDigis.InputLabel = 'hcalRawData'
ecalDigis.InputLabel = 'ecalPacker'
