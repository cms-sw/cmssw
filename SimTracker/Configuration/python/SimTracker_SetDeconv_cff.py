import FWCore.ParameterSet.Config as cms

def customise(process):
	# Signal in Deconvolution Mode
        for ps in process.mix.digitizers:
            if ps.accumulatorType == cms.string('SiStripDigitizer'):
	        ps.APVpeakmode = cms.bool(False)
	        return(process)

