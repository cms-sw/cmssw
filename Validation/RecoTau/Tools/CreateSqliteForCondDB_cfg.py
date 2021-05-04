import FWCore.ParameterSet.Config as cms

process = cms.Process( "CREATE" )

Triggers = cms.VPSet(
    cms.PSet(
        listName = cms.string( 'TauTriggerForMuDataset' ),
        hltPaths = cms.vstring('HLT_IsoMu24_eta2p1_v*'),
        dataTypeToInclude = cms.vstring('RealData','RealMuonsData'),
        ),    
    ## cms.PSet(
    ##     listName = cms.string( 'TauTriggerForSingleMuDataset' ),
    ##     hltPaths = cms.vstring('HLT_IsoMu24_eta2p1_v*'),
    ##     dataTypeToInclude = cms.vstring('RealData','RealMuonsData'),
    ##     ),    
    cms.PSet(
        listName = cms.string( 'TauTriggerForJetDataset' ),
        hltPaths = cms.vstring('HLT_Jet30_L1FastJet_v*'),
        dataTypeToInclude = cms.vstring('RealData'),
        ),    
    cms.PSet(
        listName = cms.string( 'TauTriggerForMultiJetDataset' ),
        hltPaths = cms.vstring('HLT_Jet30_L1FastJet_v*'),
        dataTypeToInclude = cms.vstring('RealData'),
        ),    
    cms.PSet(
        listName = cms.string( 'TauTriggerForDoubleElectronDataset' ),
        hltPaths = cms.vstring('HLT_Ele20_CaloIdVT_CaloIsoVT_TrkIdT_TrkIsoVT_SC4_Mass50_v*'),
        dataTypeToInclude = cms.vstring('RealElectronsData'),
        ),    
    cms.PSet(
        listName = cms.string( 'TauTriggerForTauPlusXDataset' ),
        hltPaths = cms.vstring('HLT_Ele20_CaloIdVT_CaloIsoRhoT_TrkIdT_TrkIsoT_LooseIsoPFTau20_v*'),
        dataTypeToInclude = cms.vstring('RealElectronsData'),
        ),    
    )

usedQCDTriggers = []
usedMuTriggers = []
usedEleTriggers = []
for trigger in Triggers:
    if 'RealData' in trigger.dataTypeToInclude.value():
        usedQCDTriggers.extend(trigger.hltPaths.value())
    if 'RealMuonsData' in trigger.dataTypeToInclude.value():
        usedMuTriggers.extend(trigger.hltPaths.value())
    if 'RealElectronsData' in trigger.dataTypeToInclude.value():
        usedEleTriggers.extend(trigger.hltPaths.value())

Triggers.append(
    cms.PSet(
        listName = cms.string( 'TauTriggerForALLQCDDataset' ),
        hltPaths = cms.vstring(list(set(usedQCDTriggers))),
        )
    )

Triggers.append(
    cms.PSet(
        listName = cms.string( 'TauTriggerForALLMuDataset' ),
        hltPaths = cms.vstring(list(set(usedMuTriggers))),
        )
    )

Triggers.append(
    cms.PSet(
        listName = cms.string( 'TauTriggerForALLEleDataset' ),
        hltPaths = cms.vstring(list(set(usedEleTriggers))),
        )
    )

process.load( "FWCore.MessageLogger.MessageLogger_cfi" )
process.MessageLogger.cerr.enable = False
process.MessageLogger.cout = cms.untracked.PSet(
  enable = cms.untracked.bool(True),
  INFO = cms.untracked.PSet(
    reportEvery = cms.untracked.int32( 1 )
  )
)

process.source = cms.Source( "EmptySource")
process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32( 1 )
)

process.AlCaRecoTriggerBitsRcdCreate = cms.EDAnalyzer(
    "AlCaRecoTriggerBitsRcdUpdate",
    firstRunIOV = cms.uint32( 1 ),
    lastRunIOV  = cms.int32( -1 ),
    startEmpty  = cms.bool( True ),
    listNamesRemove = cms.vstring(),
    # parameter sets to define lists of logical expressions
    triggerListsAdd = Triggers,
)

import CondCore.DBCommon.CondDBSetup_cfi
process.PoolDBOutputService = cms.Service(
    "PoolDBOutputService",
    CondCore.DBCommon.CondDBSetup_cfi.CondDBSetup,
    timetype = cms.untracked.string( 'runnumber' ),
    connect  = cms.string( 'sqlite_file:GenericTriggerEventFlag_AlCaRecoTriggerBits.db' ),
    toPut    = cms.VPSet(
        cms.PSet(
            record = cms.string( 'AlCaRecoTriggerBitsRcd' ),
            tag    = cms.string( 'PFTauDQMTrigger_v0' ),
            label = cms.untracked.string("PFTauDQMTrigger")
            ),
            ),
    )

process.p = cms.Path(
  process.AlCaRecoTriggerBitsRcdCreate
)
