import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
TTbarAnalyzeSpinCorr = DQMEDAnalyzer('TTbarSpinCorrHepMCAnalyzer',
                                      genEventInfoProductTag = cms.InputTag("generator"),
                                      genParticlesTag = cms.InputTag("genParticles")
                                      )

from GeneratorInterface.LHEInterface.lheCOMWeightProducer import *
lheCOMWeightProducer.NewECMS = cms.double(8000)


## get lorentzvectors
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
analyzeTopKinematics = DQMEDAnalyzer('TTbar_Kinematics',
                                      SaveTree = cms.untracked.bool(False),
                                      hepmcCollection = cms.InputTag("generatorSmeared"),
                                      genEventInfoProductTag = cms.InputTag("generator")
                                      )

## analyze genjets
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
analyzeGenJets = DQMEDAnalyzer('TTbar_GenJetAnalyzer',
                                jets = cms.InputTag('ak4GenJets' ),
                                genEventInfoProductTag = cms.InputTag("generator")
                                )

# --- Create list of interesting genParticles ---
from SimGeneral.HepPDTESSource.pythiapdt_cfi import *

# --- ShortList is to prevent running multiple times over full particle list ---
#see https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideGenParticlePruner

genParticlesShortList = cms.EDProducer("GenParticlePruner",
                                       src = cms.InputTag("genParticles",""),
                                       select = cms.vstring("drop  *  ",
                                                            "keep pdgId = {mu+}      & status = 1",
                                                            "keep pdgId = {mu-}      & status = 1",
                                                            "keep pdgId = {e+}       & status = 1",
                                                            "keep pdgId = {e-}       & status = 1",
                                                            "keep pdgId = {nu_e}     & status = 1",
                                                            "keep pdgId = {nu_ebar}  & status = 1",
                                                            "keep pdgId = {nu_mu}    & status = 1",
                                                            "keep pdgId = {nu_mubar} & status = 1"
                                                            )
                                       )

#see https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideGenParticlePruner
genParticlesMuons     = cms.EDProducer("GenParticlePruner",
                                       src = cms.InputTag("genParticlesShortList"),
                                       select = cms.vstring("drop  *  ",
                                                            "keep pdgId = {mu+} & status = 1 & pt > 30 & abs(eta) < 2.5",
                                                            "keep pdgId = {mu-} & status = 1 & pt > 30 & abs(eta) < 2.5"
                                                            )
                                       )

#see https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideGenParticlePruner
genParticlesElectrons = cms.EDProducer("GenParticlePruner",
                                       src = cms.InputTag("genParticlesShortList"),
                                       select = cms.vstring("drop  *  ",
                                                            "keep pdgId = {e+} & status = 1 & pt > 30 & abs(eta) < 2.5",
                                                            "keep pdgId = {e-} & status = 1 & pt > 30 & abs(eta) < 2.5"
                                                            )
                                       )

#see https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideGenParticlePruner
genParticlesNeutrinos = cms.EDProducer("GenParticlePruner",
                                       src = cms.InputTag("genParticlesShortList"),
                                       select = cms.vstring("drop  *  ",
                                                            "keep pdgId = {nu_e}     & status = 1",
                                                            "keep pdgId = {nu_ebar}  & status = 1",
                                                            "keep pdgId = {nu_mu}    & status = 1",
                                                            "keep pdgId = {nu_mubar} & status = 1"
                                                            )
                                       )

## analyze gen leptons
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
analyzeGenMuons = DQMEDAnalyzer('TTbar_GenLepAnalyzer', leptons = cms.InputTag('genParticlesMuons' ))
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
analyzeGenElecs = DQMEDAnalyzer('TTbar_GenLepAnalyzer', leptons = cms.InputTag('genParticlesElectrons' ))
from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
analyzeGenNtrns = DQMEDAnalyzer('TTbar_GenLepAnalyzer', leptons = cms.InputTag('genParticlesNeutrinos' ))



