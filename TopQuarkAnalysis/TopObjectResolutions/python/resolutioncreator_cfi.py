import FWCore.ParameterSet.Config as cms

Resolutions_electrons = cms.EDAnalyzer("ResolutionChecker",
	object    	= cms.string('electron'), 
	label  		= cms.string('selectedPatElectrons'),
	minMatchingDR	= cms.double(0.1),
	etabinValues 	= cms.vdouble(0,0.17,0.35,0.5,0.7,0.9,1.15,1.3,1.6,1.9,2.5),
	pTbinValues  	= cms.vdouble(15,22,28,35,41,49,57,68,81,104,200)
)
   
Resolutions_muons  = cms.EDAnalyzer("ResolutionChecker",
	object    	= cms.string('muon'),
	label  		= cms.string('selectedPatMuons'),
	minMatchingDR	= cms.double(0.1),
	etabinValues 	= cms.vdouble(0,0.17,0.35,0.5,0.7,0.9,1.15,1.3,1.6,1.9,2.5),
	pTbinValues  	= cms.vdouble(15,22,28,35,41,49,57,68,81,104,200)
)
     
Resolutions_lJets  = cms.EDAnalyzer("ResolutionChecker",
	object    	= cms.string('lJets'), 
	label  		= cms.string('selectedPatJets'),
	minMatchingDR	= cms.double(0.3),
	etabinValues 	= cms.vdouble(0, 0.17, 0.35, 0.5, 0.7, 0.9, 1.15, 1.4, 1.7, 2.1, 2.5),
	pTbinValues  	= cms.vdouble(20,30,40,50,60,70,80,90,110,130,200)
)
 
Resolutions_bJets  = cms.EDAnalyzer("ResolutionChecker",
	object    	= cms.string('bJets'),
	label  		= cms.string('selectedPatJets'),
	minMatchingDR	= cms.double(0.3),
	etabinValues 	= cms.vdouble(0, 0.17, 0.35, 0.5, 0.7, 0.9, 1.15, 1.4, 1.7, 2.1, 2.5),
	pTbinValues  	= cms.vdouble(20,30,40,50,60,70,80,90,110,130,200)
)

Resolutions_met  = cms.EDAnalyzer("ResolutionChecker",
	object    	= cms.string('met'),
	label  		= cms.string('patMETs'),
	minMatchingDR	= cms.double(1000.),
	pTbinValues  	= cms.vdouble(20,29,37,44,51,59,69,80,96,122,200)	
) 
	
