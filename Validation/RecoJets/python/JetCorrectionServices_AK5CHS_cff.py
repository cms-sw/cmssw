import FWCore.ParameterSet.Config as cms

#http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/JetMETCorrections/Configuration/python/JetCorrectionServices_cff.py?revision=1.21&view=markup
ak5PFchsL1Offset = cms.ESProducer(
    'L1OffsetCorrectionESProducer',
    level = cms.string('L1Offset'),
    algorithm = cms.string('AK5PFchs'),
    vertexCollection = cms.string('offlinePrimaryVertices'),
    minVtxNdof = cms.int32(4)
    )
ak5PFchsL1Fastjet = cms.ESProducer(
    'L1FastjetCorrectionESProducer',
    level       = cms.string('L1FastJet'),
    algorithm   = cms.string('AK5PFchs'),
    srcRho      = cms.InputTag('kt6PFJets','rho')
    )
ak5PFchsL2Relative = ak5CaloL2Relative = cms.ESProducer(
    'LXXXCorrectionESProducer',
    level     = cms.string('L2Relative'),
    algorithm = cms.string('AK5PFchs')
    )
ak5PFchsL3Absolute = ak5CaloL3Absolute = cms.ESProducer(
    'LXXXCorrectionESProducer',
    level     = cms.string('L3Absolute'),
    algorithm = cms.string('AK5PFchs')
    )

ak5PFchsResidual = cms.ESProducer(
    'LXXXCorrectionESProducer',
    level     = cms.string('L2L3Residual'),
    algorithm = cms.string('AK5PFchs')
    )
ak5PFchsL2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak5PFchsL2Relative','ak5PFchsL3Absolute')
    )
ak5PFchsL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak5PFchsL2Relative','ak5PFchsL3Absolute','ak5PFchsResidual')
    )
ak5PFchsL1L2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak5PFchsL1Offset','ak5PFchsL2Relative','ak5PFchsL3Absolute')
    )
ak5PFchsL1L2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak5PFchsL1Offset','ak5PFchsL2Relative','ak5PFchsL3Absolute','ak5PFchsResidual')
    )
ak5PFchsL1FastL2L3 = ak5PFchsL2L3.clone()
ak5PFchsL1FastL2L3.correctors.insert(0,'ak5PFchsL1Fastjet')
ak5PFchsL1FastL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak5PFchsL1Fastjet','ak5PFchsL2Relative','ak5PFchsL3Absolute','ak5PFchsResidual')
    )

