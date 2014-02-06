import FWCore.ParameterSet.Config as cms

#http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/CMSSW/JetMETCorrections/Configuration/python/JetCorrectionServices_cff.py?revision=1.21&view=markup
ak4PFchsL1Offset = cms.ESProducer(
    'L1OffsetCorrectionESProducer',
    level = cms.string('L1Offset'),
    algorithm = cms.string('AK5PFchs'),
    vertexCollection = cms.string('offlinePrimaryVertices'),
    minVtxNdof = cms.int32(4)
    )
ak4PFchsL1Fastjet = cms.ESProducer(
    'L1FastjetCorrectionESProducer',
    level       = cms.string('L1FastJet'),
    algorithm   = cms.string('AK5PFchs'),
    srcRho      = cms.InputTag('fixedGridRhoFastjetAll')
    )
ak4PFchsL2Relative = ak4CaloL2Relative = cms.ESProducer(
    'LXXXCorrectionESProducer',
    level     = cms.string('L2Relative'),
    algorithm = cms.string('AK5PFchs')
    )
ak4PFchsL3Absolute = ak4CaloL3Absolute = cms.ESProducer(
    'LXXXCorrectionESProducer',
    level     = cms.string('L3Absolute'),
    algorithm = cms.string('AK5PFchs')
    )

ak4PFchsResidual = cms.ESProducer(
    'LXXXCorrectionESProducer',
    level     = cms.string('L2L3Residual'),
    algorithm = cms.string('AK5PFchs')
    )
ak4PFchsL2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4PFchsL2Relative','ak4PFchsL3Absolute')
    )
ak4PFchsL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4PFchsL2Relative','ak4PFchsL3Absolute','ak4PFchsResidual')
    )
ak4PFchsL1L2L3 = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4PFchsL1Offset','ak4PFchsL2Relative','ak4PFchsL3Absolute')
    )
ak4PFchsL1L2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4PFchsL1Offset','ak4PFchsL2Relative','ak4PFchsL3Absolute','ak4PFchsResidual')
    )
ak4PFchsL1FastL2L3 = ak4PFchsL2L3.clone()
ak4PFchsL1FastL2L3.correctors.insert(0,'ak4PFchsL1Fastjet')
ak4PFchsL1FastL2L3Residual = cms.ESProducer(
    'JetCorrectionESChain',
    correctors = cms.vstring('ak4PFchsL1Fastjet','ak4PFchsL2Relative','ak4PFchsL3Absolute','ak4PFchsResidual')
    )

