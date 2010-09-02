import FWCore.ParameterSet.Config as cms

mcElectrons = cms.EDProducer(
    "GenParticlePruner",
    src = cms.InputTag("genParticles"),
    select = cms.vstring(
    "drop  *  ", # this is the default
    "keep++ pdgId = 23 | pdgId = 24",
    "drop abs(pdgId) != 11",
    "drop status != 1 "
    )
    )


mcPhotons = cms.EDProducer(
    "GenParticlePruner",
    src = cms.InputTag("genParticles"),
    select = cms.vstring(
    "drop  *  ", # this is the default
    "keep++ pdgId = 25",
    "drop pdgId != 22",
    "drop status != 1"
    )
    )

tauGenJets = cms.EDProducer(
    "TauGenJetProducer",
    GenParticles =  cms.InputTag("genParticles"),
    includeNeutrinos = cms.bool( False ),
    verbose = cms.untracked.bool( False )
    )

tauGenJetsSelectorAllHadrons = cms.EDFilter("TauGenJetDecayModeSelector",
                                            src = cms.InputTag("tauGenJets"),
                                            select = cms.vstring('oneProng0Pi0',
                                                                 'oneProng1Pi0',
                                                                 'oneProng2Pi0',
                                                                 'oneProngOther',
                                                                 'threeProng0Pi0',
                                                                 'threeProng1Pi0',
                                                                 'threeProngOther',
                                                                 'rare'),
                                            filter = cms.bool(False)
                                            )

SLHCTaus = cms.EDAnalyzer('CaloTriggerAnalyzer',
                          src    = cms.InputTag("SLHCL1ExtraParticles","Taus"),
                          ref    = cms.InputTag("tauGenJetsSelectorAllHadrons"),
                          deltaR = cms.double(0.5),
                          threshold = cms.double(30)
                          )

SLHCisoTaus = cms.EDAnalyzer('CaloTriggerAnalyzer',
                             src    = cms.InputTag("SLHCL1ExtraParticles","IsoTaus"),
                             ref    = cms.InputTag("tauGenJetsSelectorAllHadrons"),
                             deltaR = cms.double(0.5),
                             threshold = cms.double(30)
                             )

LHCTaus = cms.EDAnalyzer('CaloTriggerAnalyzer',
                         src    = cms.InputTag("l1extraParticlesCalibrated","Taus"),
                         ref    = cms.InputTag("tauGenJetsSelectorAllHadrons"),
                         deltaR = cms.double(0.5),
                         threshold = cms.double(30)
                         )

mcSequence = cms.Sequence(mcElectrons*
                          mcPhotons*
                          tauGenJets*
                          tauGenJetsSelectorAllHadrons
                          )

SLHCelectrons = cms.EDAnalyzer('CaloTriggerAnalyzer',
                               src    = cms.InputTag("SLHCL1ExtraParticles","EGamma"),
                               ref    = cms.InputTag("mcElectrons"),
                               deltaR = cms.double(0.3),
                               threshold = cms.double(30.)
                               )

LHCelectrons = cms.EDAnalyzer('CaloTriggerAnalyzer',
                              src    = cms.InputTag("l1extraParticlesCalibrated","EGamma"),
                              ref    = cms.InputTag("mcElectrons"),
                              deltaR = cms.double(0.3),
                              threshold = cms.double(30.)
                              )

SLHCisoElectrons = cms.EDAnalyzer('CaloTriggerAnalyzer',
                                  src    = cms.InputTag("SLHCL1ExtraParticles","IsoEGamma"),
                                  ref    = cms.InputTag("mcElectrons"),
                                  deltaR = cms.double(0.3),
                                  threshold = cms.double(30.)
                                  )

LHCisoElectrons = cms.EDAnalyzer('CaloTriggerAnalyzer',
                                 src    = cms.InputTag("l1extraParticlesCalibrated","IsoEGamma"),
                                 ref    = cms.InputTag("mcElectrons"),
                                 deltaR = cms.double(0.3),
                                 threshold = cms.double(30.)
                                 )

LHCallE = cms.EDProducer("CandViewMerger",
                         src=cms.VInputTag(cms.InputTag("l1extraParticlesCalibrated","EGamma"), cms.InputTag("l1extraParticlesCalibrated","IsoEGamma") )
                         )

LHCallElectrons = cms.EDAnalyzer('CaloTriggerAnalyzer',
                                 src    = cms.InputTag("LHCallE"),
                                 ref    = cms.InputTag("mcElectrons"),
                                 deltaR = cms.double(0.3),
                                 threshold = cms.double(30.)
                                 )


SLHCphotons = cms.EDAnalyzer('CaloTriggerAnalyzer',
                             src    = cms.InputTag("SLHCL1ExtraParticles","EGamma"),
                             ref    = cms.InputTag("mcPhotons"),
                             deltaR = cms.double(0.3),
                             threshold = cms.double(30.)
                             )

LHCphotons = cms.EDAnalyzer('CaloTriggerAnalyzer',
                            src    = cms.InputTag("l1extraParticlesCalibrated","EGamma"),
                            ref    = cms.InputTag("mcPhotons"),
                            deltaR = cms.double(0.3),
                            threshold = cms.double(30.)
                            )

SLHCisoPhotons = cms.EDAnalyzer('CaloTriggerAnalyzer',
                                src    = cms.InputTag("SLHCL1ExtraParticles","IsoEGamma"),
                                ref    = cms.InputTag("mcPhotons"),
                                deltaR = cms.double(0.3),
                                threshold = cms.double(30.)
                                )

LHCisoPhotons = cms.EDAnalyzer('CaloTriggerAnalyzer',
                               src    = cms.InputTag("l1extraParticlesCalibrated","IsoEGamma"),
                               ref    = cms.InputTag("mcPhotons"),
                               deltaR = cms.double(0.3),
                               threshold = cms.double(30.)
                               )

LHCallG = cms.EDProducer("CandViewMerger",
                         src=cms.VInputTag(cms.InputTag("l1extraParticlesCalibrated","EGamma"), cms.InputTag("l1extraParticlesCalibrated","IsoEGamma"))
                         )

LHCallPhotons = cms.EDAnalyzer('CaloTriggerAnalyzer',
                               src    = cms.InputTag("LHCallG"),
                               ref    = cms.InputTag("mcPhotons"),
                               deltaR = cms.double(0.3),
                               threshold = cms.double(30.)
                               )


SLHCjets = cms.EDAnalyzer('CaloTriggerAnalyzer',
                          src    = cms.InputTag("SLHCL1ExtraParticles","Jets"),
                          ref    = cms.InputTag("ak5CaloJets"),
                          deltaR = cms.double(0.5),
                          threshold = cms.double(30.)
                          )

analysisSequenceCalibrated = cms.Sequence(SLHCelectrons*
                                          LHCelectrons*
                                          SLHCisoElectrons*
                                          LHCisoElectrons*
                                          LHCallE*
                                          LHCallElectrons*
                                          SLHCphotons*
                                          LHCphotons*
                                          SLHCisoPhotons*
                                          LHCisoPhotons*
                                          LHCallG*
                                          LHCallPhotons*
                                          SLHCTaus*
                                          SLHCisoTaus*
                                          LHCTaus*
                                          SLHCjets
                                          )
