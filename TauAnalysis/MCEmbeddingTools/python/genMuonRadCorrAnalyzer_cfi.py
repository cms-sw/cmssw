import FWCore.ParameterSet.Config as cms

genMuonRadCorrAnalyzer = cms.EDAnalyzer("GenMuonRadCorrAnalyzer",
    srcSelectedMuons = cms.InputTag('goldenZmumuCandidatesGe0IsoMuons'),
    srcGenParticles = cms.InputTag('genParticles'),
    srcWeights = cms.VInputTag(),                                     
    binningMuonEn = cms.vdouble(0., 30., 40., 50., 200., -1.), # CV: -1 represents "infinity"
    muonRadiationAlgo = cms.string(''),
    beamEnergy = cms.double(4000.),                                        
    numBinsRadDivMuonEn = cms.uint32(21),
    minRadDivMuonEn = cms.double(-0.025),
    maxRadDivMuonEn = cms.double(+1.025),
    directory = cms.string('genMuonRadCorrAnalyzer')                                        
)

genMuonRadCorrAnalyzerPYTHIA = genMuonRadCorrAnalyzer.clone(
    muonRadiationAlgo = cms.string('pythia'),
    directory = cms.string('genMuonRadCorrAnalyzerPYTHIA')  
)

genMuonRadCorrAnalyzerPHOTOS = genMuonRadCorrAnalyzer.clone(
    muonRadiationAlgo = cms.string('photos'),
    directory = cms.string('genMuonRadCorrAnalyzerPHOTOS')  
)
