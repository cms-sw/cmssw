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
	threshold = cms.double(5)
)

SLHCisoTaus = cms.EDAnalyzer(
    'CaloTriggerAnalyzer',
    src    = cms.InputTag("SLHCL1ExtraParticles","IsoTaus"),
    ref    = cms.InputTag("tauGenJetsSelectorAllHadrons"),
    deltaR = cms.double(0.5),
    threshold = cms.double(5)
)


LHCTaus = cms.EDAnalyzer(
    'CaloTriggerAnalyzer',
    src    = cms.InputTag("l1extraParticles","Tau"),
    ref    = cms.InputTag("tauGenJetsSelectorAllHadrons"),
    deltaR = cms.double(0.5),
    threshold = cms.double(5)
)

mcSequence = cms.Sequence(
    mcElectrons*
    mcPhotons*
    tauGenJets*
    tauGenJetsSelectorAllHadrons
)

SLHCelectrons = cms.EDAnalyzer('CaloTriggerAnalyzer',
                           src    = cms.InputTag("SLHCL1ExtraParticles","EGamma"),
                           ref    = cms.InputTag("mcElectrons"),
                           deltaR = cms.double(0.3),
                           threshold = cms.double(5.)
)

LHCelectrons = cms.EDAnalyzer('CaloTriggerAnalyzer',
                           src    = cms.InputTag("l1extraParticles","NonIsolated"),
                           ref    = cms.InputTag("mcElectrons"),
                           deltaR = cms.double(0.3),
                           threshold = cms.double(5.)
)

SLHCisoElectrons = cms.EDAnalyzer('CaloTriggerAnalyzer',
                           src    = cms.InputTag("SLHCL1ExtraParticles","IsoEGamma"),
                           ref    = cms.InputTag("mcElectrons"),
                           deltaR = cms.double(0.3),
                           threshold = cms.double(5.)
)

LHCisoElectrons = cms.EDAnalyzer('CaloTriggerAnalyzer',
                           src    = cms.InputTag("l1extraParticles","Isolated"),
                           ref    = cms.InputTag("mcElectrons"),
                           deltaR = cms.double(0.3),
                           threshold = cms.double(5.)
)


SLHCphotons = cms.EDAnalyzer('CaloTriggerAnalyzer',
                           src    = cms.InputTag("SLHCL1ExtraParticles","EGamma"),
                           ref    = cms.InputTag("mcPhotons"),
                           deltaR = cms.double(0.3),
                           threshold = cms.double(10.)
)

LHCphotons = cms.EDAnalyzer('CaloTriggerAnalyzer',
                           src    = cms.InputTag("l1extraParticles","NonIsolated"),
                           ref    = cms.InputTag("mcPhotons"),
                           deltaR = cms.double(0.3),
                           threshold = cms.double(10.)
)


SLHCisoPhotons = cms.EDAnalyzer('CaloTriggerAnalyzer',
                           src    = cms.InputTag("SLHCL1ExtraParticles","IsoEGamma"),
                           ref    = cms.InputTag("mcPhotons"),
                           deltaR = cms.double(0.3),
                           threshold = cms.double(5.)
)

LHCisoPhotons = cms.EDAnalyzer('CaloTriggerAnalyzer',
                           src    = cms.InputTag("l1extraParticles","Isolated"),
                           ref    = cms.InputTag("mcPhotons"),
                           deltaR = cms.double(0.3),
                           threshold = cms.double(5.)
)

SLHCjets = cms.EDAnalyzer('CaloTriggerAnalyzer',
                           src    = cms.InputTag("SLHCL1ExtraParticles","Jets"),
                           ref    = cms.InputTag("ak5CaloJets"),
                           deltaR = cms.double(0.5),
                           threshold = cms.double(30.)
)

analysisSequence = cms.Sequence(
    SLHCelectrons*
    LHCelectrons*
    SLHCisoElectrons*
    LHCisoElectrons*
    SLHCphotons*
    LHCphotons*
    SLHCisoPhotons*
    LHCisoPhotons*
    SLHCTaus*
    SLHCisoTaus*
    LHCTaus*
    SLHCjets
)
