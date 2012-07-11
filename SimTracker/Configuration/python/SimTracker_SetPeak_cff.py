import FWCore.ParameterSet.Config as cms

def customise(process):
	# Signal in Deconvolution Mode
	process.simSiStripDigis.APVpeakmode = cms.bool(True)
	process.simSiStripDigis.electronPerAdc = cms.double(262.0) #this is the value measured in peak... should we add 15%?
	return(process)

