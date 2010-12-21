INPUT_MINBIAS = '/build/RAWReference/MinBias_RAW_320_STARTUP.root'
INPUT_TTBAR = '/build/RAWReference/TTbar_RAW_320_STARTUP.root'

puSTARTUP_TTBAR = '/build/RAWReference/TTbar_Tauola_PileUp_RAW_320_STARTUP.root'

relval = {
       'step1': {	'step': 'GEN-HLT',
			'timesize': (100, ['MinBias','TTbar']),
			'igprof': (50, ['TTbar']),
			'memcheck': (5, ['TTbar']),
			'pileup': ['TTbar'],
#???			'pileupInput': '',
			'cmsdriver': '--eventcontent RAWSIM --conditions auto:mc' },

	'step2': {	'step': 'RAW2DIGI-RECO',
			'timesize': (8000, ['MinBias','TTbar']),
	 		'igprof': (200, ['TTbar']),
			'memcheck': (5, ['TTbar']),
			'pileup': ['TTbar'],
			'pileupInput': puSTARTUP_TTBAR,
			'fileInput': [INPUT_MINBIAS,INPUT_TTBAR],
			'cmsdriver': '--eventcontent RECOSIM --conditions auto:startup' },

	'GENSIMDIGI': {	'step': 'GEN-SIM,DIGI',
			'timesize': (100, ['MinBias','SingleElectronE1000','SingleMuMinusPt10','SinglePiMinusE1000','TTbar']),
			'igprof': (5, ['TTbar']),
			'memcheck': (5, ['TTbar']),
			'pileup': ['TTbar'],
#???			'pileupInput': '',
			'fileInput': '',
			'cmsdriver': '--eventcontent FEVTDEBUG --conditions auto:mc' },

	'HLT': {        'step': 'HLT',
			'timesize': (8000, ['MinBias','TTbar']),
			'igprof': (500, ['TTbar']),
			'memcheck': (5, ['TTbar']),
			'pileup': ['TTbar'],
			'pileupInput': puSTARTUP_TTBAR,
			'fileInput': [INPUT_MINBIAS,INPUT_TTBAR],
			'cmsdriver': '--eventcontent RAWSIM --conditions auto:startup --processName HLTFROMRAW' },

	'FASTSIM': {	'step': 'GEN-FASTSIM',
			'timesize': (8000, ['MinBias','TTbar']),
			'igprof': (500, ['TTbar']),
			'memcheck': (5, ['TTbar']),
			'pileup': ['TTbar'],
			'cmsdriver': '--eventcontent RECOSIM --conditions auto:mc' }
}
