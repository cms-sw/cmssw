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


from Validation.RecoJets.metTester_cfi import metTester as _metTester
_hltmetTester = _metTester.clone(
    primVertex = cms.InputTag("hltPixelVertices"), # hltGoodOfflinePrimaryVertices, hltOfflinePrimaryVertices
    genMetTrue = cms.InputTag("genMetTrue"),
    genMetCalo = cms.InputTag("genMetCalo"),
)

hltMetAnalyzerPFPuppi = _hltMetTester.clone(
    InputMETLabel = cms.InputTag('hltPFPuppiMET'),
    METType = cms.untracked.string('pf'),
)

hltMetAnalyzerPF = _hltMetTester.clone(
    InputMETLabel = cms.InputTag('hltPFMET'), 
    METType = cms.untracked.string('pf'), 
)

hltMetAnalyzerPFCalo = _hltMetTester.clone(
    InputMETLabel = cms.InputTag('hltCaloMET'),
    METType = cms.untracked.string('calo'),
)

