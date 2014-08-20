import FWCore.ParameterSet.Config as cms

from L1TriggerDPG.L1Ntuples.l1Ntuple_cfg import *

# global tag
process.GlobalTag.globaltag = 'GR10_P_V6::All'

# load standard RECO for MET cleaning
process.load('Configuration.StandardSequences.Reconstruction_cff')

##################################good collisions############################################
# This filter select ~73% of events in MinBias PD
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskAlgoTrigConfig_cff')
process.load('L1TriggerConfig.L1GtConfigProducers.L1GtTriggerMaskTechTrigConfig_cff')
process.load('HLTrigger/HLTfilters/hltLevel1GTSeed_cfi')

process.L1T1coll=process.hltLevel1GTSeed.clone()
process.L1T1coll.L1TechTriggerSeeding = cms.bool(True)
process.L1T1coll.L1SeedsLogicalExpression = cms.string('0 AND (40 OR 41) AND NOT (36 OR 37 OR 38 OR 39) AND NOT ((42 AND NOT 43) OR (43 AND NOT 42))')

process.primaryVertexFilter = cms.EDFilter("VertexSelector",
   src = cms.InputTag("offlinePrimaryVertices"),
   cut = cms.string("!isFake && ndof > 4 && abs(z) <= 15 && position.Rho <= 2"), # tracksSize() > 3 for the older cut
   filter = cms.bool(True),   # otherwise it won't filter the events, just produce an empty vertex collection.
)


process.noscraping = cms.EDFilter("FilterOutScraping",
  applyfilter = cms.untracked.bool(True),
  debugOn = cms.untracked.bool(False),
  numtrack = cms.untracked.uint32(10),
  thresh = cms.untracked.double(0.25)
)


######################################################
####### MET CLEANING RECOMMENDATION STARTS HERE#######
######################################################

#is it MC or DATA
#WARNING: FOR MC turn isMC = True, otherwise the v4 of HF cleaning will be used, which includes timing cut. Timing is not modeled well in MC
isMC = False
useHBHEcleaning = True
useHBHEfilter = True

HFPMTcleaningversion = 4   # version 1 = (loose), version 2 = (medium), version 3 = (tight)
# VERSION 4 is the currently recommended version, as of 28 May 2010.

if useHBHEfilter == True:
    process.load('CommonTools/RecoAlgos/HBHENoiseFilter_cfi')
    process.hbhefilter = cms.Path(process.HBHENoiseFilter)
    
# New SeverityLevelComputer that forces RecHits with UserDefinedBit0 set to be excluded from new rechit collection
import JetMETAnalysis.HcalReflagging.RemoveAddSevLevel as RemoveAddSevLevel
process.hcalRecAlgos=RemoveAddSevLevel.RemoveFlag(process.hcalRecAlgos,"HFLongShort")

# UserDefinedBit0 is used by both the HF and HBHE reflaggers
process.hcalRecAlgos=RemoveAddSevLevel.AddFlag(process.hcalRecAlgos,"UserDefinedBit0",10)

# HF RecHit reflagger
process.load("JetMETAnalysis/HcalReflagging/HFrechitreflaggerJETMET_cff")
if HFPMTcleaningversion==1:
    process.hfrecoReflagged = process.HFrechitreflaggerJETMETv1.clone()
elif HFPMTcleaningversion==2:
    process.hfrecoReflagged = process.HFrechitreflaggerJETMETv2.clone()
elif HFPMTcleaningversion==3:
    process.hfrecoReflagged = process.HFrechitreflaggerJETMETv3.clone()
elif HFPMTcleaningversion==4:
    if (isMC==False):
        process.hfrecoReflagged = process.HFrechitreflaggerJETMETv4.clone()
    else:
        process.hfrecoReflagged = process.HFrechitreflaggerJETMETv2.clone()
elif HFPMTcleaningversion==5:
    if (isMC==False):
        process.hfrecoReflagged = process.HFrechitreflaggerJETMETv5.clone()
    else:
        process.hfrecoReflagged = process.HFrechitreflaggerJETMETv3.clone()


# HBHE RecHit reflagger
process.load("JetMETAnalysis/HcalReflagging/hbherechitreflaggerJETMET_cfi")
process.hbherecoReflagged = process.hbherechitreflaggerJETMET.clone()
process.hbherecoReflagged.debug=0

# Use the reflagged HF RecHits to make the CaloTowers
process.towerMaker.hfInput = "hfrecoReflagged"
process.towerMakerWithHO.hfInput = "hfrecoReflagged"

# Path and EndPath definitions

if (useHBHEcleaning==False):
    process.reflagging_step = cms.Path(process.hfrecoReflagged)
else:
    process.reflagging_step = cms.Path(process.hfrecoReflagged+process.hbherecoReflagged)
    # Need to specify that new HBHE collection should be fed to calotower maker
    process.towerMaker.hbheInput = "hbherecoReflagged"
    process.towerMakerWithHO.hbheInput = "hbherecoReflagged"

# Instead of rejecting the event, add a flag indicating the HBHE noise 
process.load('CommonTools/RecoAlgos/HBHENoiseFilterResultProducer_cfi')
process.hbheflag = cms.Path(process.HBHENoiseFilterResultProducer)

######################################################
####### MET CLEANING RECOMMENDATION ENDS HERE#######
######################################################


# L1 ntuples
process.l1NtupleProducer.generatorSource      = cms.InputTag("none")
process.l1NtupleProducer.simulationSource     = cms.InputTag("none")
process.l1NtupleProducer.gmtSource            = cms.InputTag("gtDigis")
process.l1NtupleProducer.gtEvmSource          = cms.InputTag("none")
process.l1NtupleProducer.gtSource             = cms.InputTag("gtDigis")
process.l1NtupleProducer.gctIsoEmSource       = cms.InputTag("gctDigis","isoEm")
process.l1NtupleProducer.gctNonIsoEmSource    = cms.InputTag("gctDigis","nonIsoEm")
process.l1NtupleProducer.gctCentralJetsSource = cms.InputTag("gctDigis","cenJets")
process.l1NtupleProducer.gctTauJetsSource     = cms.InputTag("gctDigis","tauJets")
process.l1NtupleProducer.gctForwardJetsSource = cms.InputTag("gctDigis","forJets")
process.l1NtupleProducer.gctEnergySumsSource  = cms.InputTag("gctDigis")
process.l1NtupleProducer.rctSource            = cms.InputTag("gctDigis")
process.l1NtupleProducer.dttfSource           = cms.InputTag("none")
process.l1NtupleProducer.csctfTrkSource       = cms.InputTag("none")  
process.l1NtupleProducer.csctfLCTSource       = cms.InputTag("none")
process.l1NtupleProducer.csctfStatusSource    = cms.InputTag("none")

process.l1RecoTreeProducer.jetptThreshold = cms.double(5)

process.gctDigis.numberOfGctSamplesToUnpack = cms.uint32(5)
process.l1extraParticles.centralBxOnly = cms.bool(False)

#process.p.remove(process.dttfDigis)
#process.p.remove(process.csctfDigis)
#process.p.remove(process.l1extraParticles)
#process.p.remove(process.l1ExtraTreeProducer)
#process.p.remove(process.l1MuonRecoTreeProducer)


process.ntuple = cms.Path(
#    process.L1T1coll
    process.primaryVertexFilter
    +process.noscraping
    +process.HBHENoiseFilter
    +process.l1NtupleProducer
    +process.l1ExtraTreeProducer
    +process.l1RecoTreeProducer
)

# re-reco jets and MET
process.rereco_step = cms.Path(
    process.caloTowersRec
    *(process.recoJets*process.recoJetIds+process.recoTrackJets)
    *process.recoJetAssociations
    *process.metreco
)

process.schedule = cms.Schedule(process.reflagging_step,
                                process.rereco_step,
                                process.ntuple)


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )


readFiles.extend( [
    '/store/data/Run2010A/JetMETTau/RECO/Jun14thReReco_v2/0000/FE90D67B-047A-DF11-ACE6-001A64789DC8.root',
    '/store/data/Run2010A/JetMETTau/RECO/Jun14thReReco_v2/0000/FE3E05A0-037A-DF11-967C-003048635CE2.root',
    '/store/data/Run2010A/JetMETTau/RECO/Jun14thReReco_v2/0000/FE285150-047A-DF11-9AD1-003048D47A36.root',
    '/store/data/Run2010A/JetMETTau/RECO/Jun14thReReco_v2/0000/FCEBE72D-067A-DF11-81FE-003048D45FAA.root',
    '/store/data/Run2010A/JetMETTau/RECO/Jun14thReReco_v2/0000/FCD95397-067A-DF11-8372-003048D47778.root'
#    '/store/data/Commissioning10/MinimumBias/RECO/May27thReReco_PreProduction_v1/0166/8258DC0D-AF69-DF11-882A-0026189437F8.root',
#    '/store/data/Commissioning10/MinimumBias/RECO/May27thReReco_PreProduction_v1/0166/06E76E18-AF69-DF11-8DA7-003048D42DC8.root',
#    '/store/data/Commissioning10/MinimumBias/RECO/May27thReReco_PreProduction_v1/0165/FAE07A9A-7969-DF11-92ED-002354EF3BDB.root',
#    '/store/data/Commissioning10/MinimumBias/RECO/May27thReReco_PreProduction_v1/0165/FA4D4132-7769-DF11-BEE5-00261894383B.root',
#    '/store/data/Commissioning10/MinimumBias/RECO/May27thReReco_PreProduction_v1/0165/F69C71AC-7769-DF11-B4BC-002618943960.root',
#    '/store/data/Commissioning10/MinimumBias/RECO/May27thReReco_PreProduction_v1/0165/F64EB00B-7869-DF11-B7D1-002618943902.root'
] )
