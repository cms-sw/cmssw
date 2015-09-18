import FWCore.ParameterSet.Config as cms

TTbarAnalyzeSpinCorr = cms.EDAnalyzer("TTbarSpinCorrHepMCAnalyzer",
                                      genEventInfoProductTag = cms.InputTag("generator"),
                                      genParticlesTag = cms.InputTag("genParticles")
                                      )

from GeneratorInterface.LHEInterface.lheCOMWeightProducer import *
lheCOMWeightProducer.NewECMS = cms.double(8000)


## get lorentzvectors
analyzeTopKinematics = cms.EDAnalyzer('TTbar_Kinematics',
                                      SaveTree = cms.untracked.bool(False),
                                      hepmcCollection = cms.InputTag("generator",""),
                                      genEventInfoProductTag = cms.InputTag("generator")
                                      )

## analyze genjets
analyzeGenJets = cms.EDAnalyzer("TTbar_GenJetAnalyzer",
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
                                                            "keep obj.pdgId() == {mu+}      && obj.status() == 1",
                                                            "keep obj.pdgId() == {mu-}      && obj.status() == 1",
                                                            "keep obj.pdgId() == {e+}       && obj.status() == 1",
                                                            "keep obj.pdgId() == {e-}       && obj.status() == 1",
                                                            "keep obj.pdgId() == {nu_e}     && obj.status() == 1",
                                                            "keep obj.pdgId() == {nu_ebar}  && obj.status() == 1",
                                                            "keep obj.pdgId() == {nu_mu}    && obj.status() == 1",
                                                            "keep obj.pdgId() == {nu_mubar} && obj.status() == 1"
                                                            )
                                       )

#see https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideGenParticlePruner
genParticlesMuons     = cms.EDProducer("GenParticlePruner",
                                       src = cms.InputTag("genParticlesShortList"),
                                       select = cms.vstring("drop  *  ",
                                                            "keep obj.pdgId() == {mu+} && obj.status() == 1 && obj.pt() > 30 && std::abs(obj.eta()) < 2.5",
                                                            "keep obj.pdgId() == {mu-} && obj.status() == 1 && obj.pt() > 30 && std::abs(obj.eta()) < 2.5"
                                                            )
                                       )

#see https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideGenParticlePruner
genParticlesElectrons = cms.EDProducer("GenParticlePruner",
                                       src = cms.InputTag("genParticlesShortList"),
                                       select = cms.vstring("drop  *  ",
                                                            "keep obj.pdgId() == {e+} && obj.status() == 1 && obj.pt() > 30 && std::abs(obj.eta()) < 2.5",
                                                            "keep obj.pdgId() == {e-} && obj.status() == 1 && obj.pt() > 30 && std::abs(obj.eta()) < 2.5"
                                                            )
                                       )

#see https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideGenParticlePruner
genParticlesNeutrinos = cms.EDProducer("GenParticlePruner",
                                       src = cms.InputTag("genParticlesShortList"),
                                       select = cms.vstring("drop  *  ",
                                                            "keep obj.pdgId() == {nu_e}     && obj.status() == 1",
                                                            "keep obj.pdgId() == {nu_ebar}  && obj.status() == 1",
                                                            "keep obj.pdgId() == {nu_mu}    && obj.status() == 1",
                                                            "keep obj.pdgId() == {nu_mubar} && obj.status() == 1"
                                                            )
                                       )

## analyze gen leptons
analyzeGenMuons = cms.EDAnalyzer("TTbar_GenLepAnalyzer", leptons = cms.InputTag('genParticlesMuons' ))
analyzeGenElecs = cms.EDAnalyzer("TTbar_GenLepAnalyzer", leptons = cms.InputTag('genParticlesElectrons' ))
analyzeGenNtrns = cms.EDAnalyzer("TTbar_GenLepAnalyzer", leptons = cms.InputTag('genParticlesNeutrinos' ))



