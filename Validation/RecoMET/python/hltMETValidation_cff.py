from PhysicsTools.PatAlgos.slimming.genParticles_cff import *
from PhysicsTools.NanoAOD.genparticles_cff import *
from RecoJets.Configuration.GenJetParticles_cff import *
from RecoJets.Configuration.RecoGenJets_cff import *

hltMetPreValidSeq = cms.Sequence(
)

# metPreValidSeqTask = cms.Task(ak4PFCHSL1FastjetCorrector,
#                               ak4PFCHSL2RelativeCorrector,
#                               ak4PFCHSL3AbsoluteCorrector,
#                               ak4PFCHSResidualCorrector
# )
# metPreValidSeq = cms.Sequence(metPreValidSeqTask)

# jetPreValidSeqTask = cms.Task(ak4CaloL2RelativeCorrector,
#                               ak4CaloL3AbsoluteCorrector,
#                               ak4PFL1FastjetCorrector,
#                               ak4PFL2RelativeCorrector,
#                               ak4PFL3AbsoluteCorrector,
#                               ak4PFCHSL1FastjetCorrector,
#                               ak4PFCHSL2RelativeCorrector,
#                               ak4PFCHSL3AbsoluteCorrector
# )
# jetPreValidSeq=cms.Sequence(jetPreValidSeqTask)


from Validation.RecoMET.metTester_cfi import metTester as _metTester
_hltMetTester = _metTester.clone(
    primaryVertices = cms.InputTag("hltPixelVertices"), # hltGoodOfflinePrimaryVertices, hltOfflinePrimaryVertices
    genMetTrue = cms.InputTag("genMetTrue"),
    genMetCalo = cms.InputTag("genMetCalo"),
)

hltMetAnalyzerPF = _hltMetTester.clone(
    inputMETLabel = cms.InputTag('hltPFMET'), 
    METType = cms.untracked.string('pf'), 
)

hltMetAnalyzerPFPuppi = _hltMetTester.clone(
    inputMETLabel = cms.InputTag('hltPFPuppiMET'),
    METType = cms.untracked.string('pf'),
)

hltMetAnalyzerPFPuppi = _hltMetTester.clone(
    inputMETLabel = cms.InputTag('hltPFPuppiMET'),
    METType = cms.untracked.string('pf'),
)

hltMetTypeOneAnalyzerPFPuppi = _hltMetTester.clone(
    inputMETLabel = cms.InputTag('hltPFPuppiMETTypeOne'),
    METType = cms.untracked.string('pf'),
)

hltMetAnalyzerPFCalo = _hltMetTester.clone(
    inputMETLabel = cms.InputTag('hltCaloMET'),
    METType = cms.untracked.string('calo'),
)

