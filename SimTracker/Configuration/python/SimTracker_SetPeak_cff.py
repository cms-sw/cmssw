import FWCore.ParameterSet.Config as cms

def customise(process):
	# Signal in Deconvolution Mode
        for ps in process.mix.digitizers:
            if ps.accumulatorType == cms.string('SiStripDigitizer'):
	        ps.APVpeakmode = cms.bool(True)
	        ps.electronPerAdc = cms.double(262.0) #this is the value measured in peak... should we add 15%?
	        return(process)

