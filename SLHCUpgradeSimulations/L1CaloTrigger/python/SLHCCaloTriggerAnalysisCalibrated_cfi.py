import FWCore.ParameterSet.Config as cms

mcElectrons = cms.EDProducer(
    "GenParticlePruner",
    src = cms.InputTag("genParticles"),
    select = cms.vstring(
    "drop  *  ", # this is the default
	"keep++ pdgId = 23 | pdgId = 24 || pdgId = 25",
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

mcEG = cms.EDProducer("CandViewMerger",
		src=cms.VInputTag(cms.InputTag("mcElectrons"), cms.InputTag("mcPhotons"))
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
		threshold = cms.double(60)
		)

SLHCisoTaus = cms.EDAnalyzer('CaloTriggerAnalyzer',
		src    = cms.InputTag("SLHCL1ExtraParticles","IsoTaus"),
		ref    = cms.InputTag("tauGenJetsSelectorAllHadrons"),
		deltaR = cms.double(0.5),
		threshold = cms.double(60.0)
		)

LHCTaus = cms.EDAnalyzer('CaloTriggerAnalyzer',
		src    = cms.InputTag("l1extraParticlesCalibrated","Taus"),
		ref    = cms.InputTag("tauGenJetsSelectorAllHadrons"),
		deltaR = cms.double(0.5),
		threshold = cms.double(60.0)
		)

mcSequence = cms.Sequence(mcElectrons*
		mcPhotons*
		mcEG*
		tauGenJets*
		tauGenJetsSelectorAllHadrons
		)

SLHCelectrons = cms.EDAnalyzer('CaloTriggerAnalyzer',
		src    = cms.InputTag("SLHCL1ExtraParticles","EGamma"),
		ref    = cms.InputTag("mcElectrons"),
		deltaR = cms.double(0.3),
		threshold = cms.double(30.0)
		)

LHCelectrons = cms.EDAnalyzer('CaloTriggerAnalyzer',
		src    = cms.InputTag("l1extraParticlesCalibrated","EGamma"),
		ref    = cms.InputTag("mcElectrons"),
		deltaR = cms.double(0.3),
		threshold = cms.double(30.0)
		)

SLHCisoElectrons = cms.EDAnalyzer('CaloTriggerAnalyzer',
		src    = cms.InputTag("SLHCL1ExtraParticles","IsoEGamma"),
		ref    = cms.InputTag("mcElectrons"),
		deltaR = cms.double(0.3),
		threshold = cms.double(30.0)
		)

LHCisoElectrons = cms.EDAnalyzer('CaloTriggerAnalyzer',
		src    = cms.InputTag("l1extraParticlesCalibrated","IsoEGamma"),
		ref    = cms.InputTag("mcElectrons"),
		deltaR = cms.double(0.3),
		threshold = cms.double(30.0)
		)

LHCallE = cms.EDProducer("CandViewMerger",
		src=cms.VInputTag(cms.InputTag("l1extraParticlesCalibrated","EGamma"), cms.InputTag("l1extraParticlesCalibrated","IsoEGamma") )
		)

LHCallElectrons = cms.EDAnalyzer('CaloTriggerAnalyzer',
		src    = cms.InputTag("LHCallE"),
		ref    = cms.InputTag("mcElectrons"),
		deltaR = cms.double(0.3),
		threshold = cms.double(30.0)
		)


SLHCphotons = cms.EDAnalyzer('CaloTriggerAnalyzer',
		src    = cms.InputTag("SLHCL1ExtraParticles","EGamma"),
		ref    = cms.InputTag("mcPhotons"),
		deltaR = cms.double(0.3),
		threshold = cms.double(30.0)
		)

LHCphotons = cms.EDAnalyzer('CaloTriggerAnalyzer',
		src    = cms.InputTag("l1extraParticlesCalibrated","EGamma"),
		ref    = cms.InputTag("mcPhotons"),
		deltaR = cms.double(0.3),
		threshold = cms.double(30.0)
		)

SLHCisoPhotons = cms.EDAnalyzer('CaloTriggerAnalyzer',
		src    = cms.InputTag("SLHCL1ExtraParticles","IsoEGamma"),
		ref    = cms.InputTag("mcPhotons"),
		deltaR = cms.double(0.3),
		threshold = cms.double(30.0)
		)

LHCisoPhotons = cms.EDAnalyzer('CaloTriggerAnalyzer',
		src    = cms.InputTag("l1extraParticlesCalibrated","IsoEGamma"),
		ref    = cms.InputTag("mcPhotons"),
		deltaR = cms.double(0.3),
		threshold = cms.double(30.0)
		)

LHCallG = cms.EDProducer("CandViewMerger",
		src=cms.VInputTag(cms.InputTag("l1extraParticlesCalibrated","EGamma"), cms.InputTag("l1extraParticlesCalibrated","IsoEGamma"))
		)

LHCallPhotons = cms.EDAnalyzer('CaloTriggerAnalyzer',
		src    = cms.InputTag("LHCallG"),
		ref    = cms.InputTag("mcPhotons"),
		deltaR = cms.double(0.3),
		threshold = cms.double(30.0)
		)


SLHCjets = cms.EDAnalyzer('CaloTriggerAnalyzer',
		src    = cms.InputTag("SLHCL1ExtraParticles","Jets"),
		ref    = cms.InputTag("ak5CaloJets"),
		deltaR = cms.double(0.5),
		threshold = cms.double(30.0)
		)

isoEG = cms.EDAnalyzer('CaloTriggerAnalyzer2',
		LHCsrc = cms.InputTag("l1extraParticlesCalibrated","IsoEGamma"),
		SLHCsrc = cms.InputTag("SLHCL1ExtraParticles","IsoEGamma"),
		ref= cms.InputTag("mcEG"),
		deltaR=cms.double(0.3),
		threshold=cms.double(30.0),
		)
EG = cms.EDAnalyzer('CaloTriggerAnalyzer2',
		LHCsrc = cms.InputTag("LHCallE"),
		SLHCsrc = cms.InputTag("SLHCL1ExtraParticles","EGamma"),
		ref= cms.InputTag("mcEG"),
		deltaR=cms.double(0.3),
		threshold=cms.double(30.0),
		)

isoTau = cms.EDAnalyzer('CaloTriggerAnalyzer2',
		LHCsrc = cms.InputTag("l1extraParticlesCalibrated","Taus"),
		SLHCsrc    = cms.InputTag("SLHCL1ExtraParticles","IsoTaus"),
		ref    = cms.InputTag("tauGenJetsSelectorAllHadrons"),
		deltaR = cms.double(0.5),
		threshold = cms.double(60.0)
		)


analysisSequenceCalibrated = cms.Sequence(#SLHCelectrons*
		#                                          LHCelectrons*
		#SLHCisoElectrons*
		#LHCisoElectrons*
		#LHCallE*
		#                                          LHCallElectrons*
		#                                          SLHCphotons*
		#                                          LHCphotons*
		#                                          SLHCisoPhotons*
		#                                          LHCisoPhotons*
		#                                          LHCallG*
		#                                          LHCallPhotons*
		#                                          SLHCTaus*
		#                                          SLHCisoTaus*
		#                                          LHCTaus*
		#                                          SLHCjets*
		isoEG*
		isoTau
		)

