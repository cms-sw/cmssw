import FWCore.ParameterSet.Config as cms

def switchInputTags(process):
    for iModule in [ process.patCandHLT1ElectronStartup, 
	             process.patHLT1Photon, 
	             process.patHLT1PhotonRelaxed, 
	             process.patHLT2Photon, 
		     process.patHLT2PhotonRelaxed, 
		     process.patHLT1Electron, 
	             process.patHLT1ElectronRelaxed, 
		     process.patHLT2Electron, 
		     process.patHLT2ElectronRelaxed, 
		     process.patHLT1MuonIso, 
		     process.patHLT1MuonNonIso, 
		     process.patHLT2MuonNonIso, 
	             process.patHLT1Tau, 
	             process.patHLT2TauPixel, 
		     process.patHLT2jet, 
		     process.patHLT3jet, 
		     process.patHLT4jet, 
		     process.patHLT1MET65, 
	             process.patHLTIsoMu11, 
		     process.patHLTMu11, 
		     process.patHLTDoubleIsoMu3, 
		     process.patHLTDoubleMu3, 
		     process.patHLTIsoEle15LWL1I, 
		     process.patHLTEle15LWL1R, 
		     process.patHLTDoubleIsoEle10LWL1I, 
		     process.patHLTDoubleEle5SWL1R, 
		     process.patHLTLooseIsoTauMET30L1MET, 
	             process.patHLTDoubleIsoTauTrk3 ]:
	iModule.filterName.processName = "SECONDHLT"
	iModule.triggerEvent.processName = "SECONDHLT"

    process.cfgTrigger.src.processName = "SECONDHLT"
    process.muTauEventDump.l1GtReadoutRecordSource.processName = "SECONDHLT"
    process.muTauEventDump.l1GtObjectMapRecordSource.processName = "SECONDHLT"
    process.muTauEventDump.hltResultsSource.processName = "SECONDHLT"

    process.patTrigger.processName = "SECONDHLT"
    process.patTriggerEvent.processName = "SECONDHLT"
    process.triggerHistManager.hltResultsSource.processName = "SECONDHLT"

