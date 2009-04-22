import FWCore.ParameterSet.Config as cms

def customise(process):
	# Signal in Deconvolution Mode
	process.simSiStripDigis.APVpeakmode = cms.bool(False)
	return(process)

