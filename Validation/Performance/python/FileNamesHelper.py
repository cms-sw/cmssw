#!/usr/bin/env python
import re, os
import parsingRulesHelper

""" a lambda fucntion which checks only two first parts of tuple: candle and step of the JobID"""
f_candle_and_step_inJobID = lambda candle, step, x: x[0] == candle and x[1] == step


"""
Includes general functions to work with fileNames and related operations:
* getting candle, step etc - JobID from fileName and vice-versa
  - includes conditions, pileup_type, event_content <-- read this from Simulationcandles [TODO: we have it in this module for simplicity, might be moved later]
* root file size from candle,step
* reads simulation candles to get release version

"""

universal_candle_step_regs = {}
test_root_file = re.compile(".root$", re.IGNORECASE)


""" 
We have Simulation candles lines in format like: 

cmsDriver.py TTbar_Tauola.cfi -n 100 --step=DIGI --filein file:TTBAR__GEN,SIM_PILEUP.root --fileout=TTBAR__DIGI_PILEUP.root --customise=Validation/Performance/MixingModule.py --conditions FrontierConditions_GlobalTag,MC_31X_V3::All --eventcontent FEVTDEBUG  --pileup=LowLumiPileUp @@@ Timing_Parser @@@ TTBAR__DIGI_PILEUP_TimingReport @@@ reuse

"""
simCandlesRules =  (

			#e.g.: --conditions FrontierConditions_GlobalTag,MC_31X_V4::All --eventcontent RECOSIM
			(("cms_driver_options", ), r"""^cmsDriver.py(.+)$"""),
			#Changing the following to allow for new cmsDriver.py --conditions option (that can optionally drop the FrontierConditions_GlobalTag,)
			(("", "conditions", ""), r"""^cmsDriver.py(.*)--conditions ([^\s]+)(.*)$""", "req"),
			(("",  "pileup_type", ""), r"""^cmsDriver.py(.*)--pileup=([^\s]+)(.*)$"""),
			(("",  "step", ""), r"""^cmsDriver.py(.*)--step=([^\s]+)(.*)$""", "req"),
			#not shure if event content is required
			(("",  "event_content", ""), r"""^cmsDriver.py(.*)--eventcontent ([^\s]+)(.*)$""", "req"),
			(("",  "num_events", ""), r"""^cmsDriver.py(.*)-n ([^\s]+)(.*)$""", "req"),
  
			#TODO: after changeing the splitter to "taskset -c ..." this is no longer included into the part of correct job
			#(("input_user_root_file", ), r"""^For these tests will use user input file (.+)$"""),
)
simCandlesRules = map(parsingRulesHelper.rulesRegexpCompileFunction, simCandlesRules)
        
def read_ConfigurationFromSimulationCandles(path, step, is_pileup):
	# Here we parse SimulationCandles_<version: e.g. CMSSW_3_2_0>.txt which contains
	# release:TODO, release_base [path] - we can put it to release [but it's of different granularity]
	# how to reproduce stuff: TODO

	try:
		""" get the acual file """
		SimulationCandles_file = [os.path.join(path, f) for f in os.listdir(path)
					 if os.path.isfile(os.path.join(path, f)) and f.startswith("SimulationCandles_")][0]
	except IndexError:
		return None

	""" read and parse it;  format: #Version     : CMSSW_3_2_0 """
	f = open(SimulationCandles_file, 'r')	

	lines =  [s.strip() for s in f.readlines()]
	f.close()



	""" we call a shared helper to parse the file """

	for line in lines:
		#print line
		#print simCandlesRules[2][1].match(line) and simCandlesRules[2][1].match(line).groups() or ""

		info, missing_fields = parsingRulesHelper.rulesParser(simCandlesRules, [line], compileRules = False)
		#print info
		#Massaging the info dictionary conditions entry to allow for new cmsDriver.py --conditions option:
		if 'auto:' in info['conditions']:
			from Configuration.AlCa.autoCond import autoCond
			info['conditions'] = autoCond[ info['conditions'].split(':')[1] ].split("::")[0] 
		else:
			if 'FrontierConditions_GlobalTag' in info['conditions']:
				info['conditions']=info['conditions'].split(",")[1]
		#print (info, missing_fields)
		#if we successfully parsed the line of simulation candles:
		if not missing_fields:
			#we have to match only step and 
			if info["step"].strip() == step.strip() and ((not is_pileup and not info["pileup_type"]) or (is_pileup and info["pileup_type"])):
				# if it's pile up or not:
				#print "Info for <<%s, %s>>: %s" % (str(step), str(is_pileup), str(info))
				return info
				




def getJobID_fromFileName(logfile_name, suffix, givenPath =""):
	#TODO: join together with the one from parseTimingReport.py
	""" 
	Returns the JobID (candle, step, pileup_type, conditions, event_content) out of filename
	-- if no pile up returns empty string for pileup type
	
	* the candle might include one optional underscore:
	>>> getJobID_fromFileName("PI-_1000_GEN,SIM.root", "\.root")
	('PI-_1000', 'GEN,SIM', '', '')
	
	* otherwise after candle we have two underscores:
	>>> getJobID_fromFileName("MINBIAS__GEN,FASTSIM.root", "\.root")
	('MINBIAS', 'GEN,FASTSIM', '', '')
	
	* and lastly we have the PILEUP possibility:
	>>> getJobID_fromFileName("TTBAR__DIGI_PILEUP.root", "\.root")
	('TTBAR', 'DIGI', 'PILEUP', '')
	"""
	import os
	
	# get the actual filename (no path
	(path, filename) = os.path.split(logfile_name)
	if givenPath:
		path = givenPath
	
	if not universal_candle_step_regs.has_key(suffix):
		#create and cache a regexp
		universal_candle_step_regs[suffix] = re.compile( \
			r"""
			#candle1_[opt:candle2]_		
			^([^_]+_[^_]*)_

			# step
			([^_]+)(_PILEUP)?%s$
		""" % suffix , re.VERBOSE)

	

	#print logfile_name
	result = universal_candle_step_regs[suffix].search(filename)
	if result:
		#print result.groups()
		#print "result: %s" % str(result.groups())
		candle = result.groups()[0]
		step = result.groups()[1].replace('-', ',')
		is_pileup = result.groups()[2]
		if is_pileup:
			is_pileup = "PILEUP"
		else:
			is_pileup = ""
		
		""" if we had the candle without underscore inside (like TTBAR but not E_1000) 
		on the end of result and underscore which needs to be removed """
		
		if (candle[-1] == '_'):
			candle = candle[0:-1]

		""" try to fetch the conditions and real pileup type if the SimulationCandles.txt is existing """
		conditions = ''
		event_content = ''
		try:
			conf = read_ConfigurationFromSimulationCandles(path = path, step = step, is_pileup= is_pileup)
			if conf:
				is_pileup = conf["pileup_type"]
				conditions = conf["conditions"]
				event_content = conf["event_content"]
		except OSError, e:
			pass

		return (candle, step, is_pileup, conditions, event_content)
	else:
		return (None, None, None, None, None)


def getJobID_fromRootFileName(logfile_name):
	""" 
	Returns the candle and STEP out of filename:
	
	* the candle might include one optional underscore:
	>>> getJobID_fromRootFileName("PI-_1000_GEN,SIM.root")
	('PI-_1000', 'GEN,SIM', '', '')
	
	* otherwise after candle we have two underscores:
	>>> getJobID_fromRootFileName("MINBIAS__GEN,FASTSIM.root")
	('MINBIAS', 'GEN,FASTSIM', '', '')
	
	* and lastly we have the PILEUP possibility:
	>>> getJobID_fromRootFileName("TTBAR__DIGI_PILEUP.root")
	('TTBAR', 'DIGI', 'PILEUP', '')
	"""
	return getJobID_fromFileName(logfile_name, "\\.root")

def getJobID_fromEdmSizeFileName(logfile_name):
	""" 
	Returns the candle and STEP out of filename:
	
	* the candle might include one optional underscore:
	>>> getJobID_fromEdmSizeFileName("E_1000_GEN,SIM_EdmSize")
	('E_1000', 'GEN,SIM', '', '')
	
	* otherwise after candle we have two underscores:
	>>> getJobID_fromEdmSizeFileName("TTBAR__RAW2DIGI,RECO_EdmSize")
	('TTBAR', 'RAW2DIGI,RECO', '', '')
	
	* and lastly we have the PILEUP possibility:
	>>> getJobID_fromEdmSizeFileName("TTBAR__GEN,SIM_PILEUP_EdmSize")
	('TTBAR', 'GEN,SIM', 'PILEUP', '')
	"""
	return getJobID_fromFileName(logfile_name, "_EdmSize")

def getJobID_fromTimeReportLogName(logfile_name):
	""" 
	Returns the candle and STEP out of filename:
	
	* the candle might include one optional underscore:
	>>> getJobID_fromTimeReportLogName("E_1000_GEN,SIM_TimingReport.log")
	('E_1000', 'GEN,SIM', '', '')
	
	* otherwise after candle we have two underscores:
	>>> getJobID_fromTimeReportLogName("test_data/TTBAR__RAW2DIGI,RECO_TimingReport.log")
	('TTBAR', 'RAW2DIGI,RECO', '', '')
	
	* and lastly we have the PILEUP possibility:
	>>> getJobID_fromTimeReportLogName("TTBAR__DIGI_PILEUP_TimingReport.log")
	('TTBAR', 'DIGI', 'PILEUP', '')
	"""
	return getJobID_fromFileName(logfile_name, "_TimingReport.log")

def getJobID_fromMemcheckLogName(logfile_name):
	""" 
	Returns the candle and STEP out of filename:
	
	* otherwise after candle we have two underscores:
	>>> getJobID_fromTimeReportLogName("test_data/TTBAR__RAW2DIGI,RECO_memcheck_vlgd.xml")
	('TTBAR', 'RAW2DIGI,RECO', '', '')
	
	* and lastly we have the PILEUP possibility:
	>>> getJobID_fromTimeReportLogName("TTBAR__DIGI_PILEUP_memcheck_vlgd.xml")
	('TTBAR', 'DIGI', 'PILEUP', '')
	"""
	return getJobID_fromFileName(logfile_name, "_memcheck_vlgd.xml")	

def getJobID_fromIgProfLogName(logfile_name):
	""" 
	Returns the candle and STEP out of .sql3 filename:

	everything is given, just have to split it...
	like:
	TTbar___GEN,FASTSIM___LowLumiPileUp___MC_37Y_V5___RAWSIM___MEM_LIVE___1.sql3
	and correct the conditions!
	
	"""

	(path, filename) = os.path.split(logfile_name)

	params = filename.split("___")
	candle = params[0].upper()
	step = params[1]
	pileup_type = params[2]
	if pileup_type == "NOPILEUP":
		pileup_type = ""
	elif pileup_type == "LowLumiPileUp":
		pileup_type = "PILEUP"
	#conditions = params[3] + "::All"
	#event_content = params[4]
	
	#get the conditions from the SimulationCandles!!
	conf = read_ConfigurationFromSimulationCandles(path = path, step = step, is_pileup= pileup_type)
	if conf:
		is_pileup = conf["pileup_type"]
		conditions = conf["conditions"]
		event_content = conf["event_content"]
		return (candle, step, is_pileup, conditions, event_content)
	else:
		return (None, None, None, None, None)			

""" Get the root file size for the candle, step in current dir """
def getRootFileSize(path, candle, step):
	files = os.listdir(path)
	root_files = [os.path.join(path, f) for f in files
				 if test_root_file.search(f) 
					and os.path.isfile(os.path.join(path, f)) ]

	""" get the size of file if it is the root file for current candle and step """
	try:
		size = [os.stat(f).st_size for f in root_files
			 if f_candle_and_step_inJobID(candle, step, getJobID_fromRootFileName(f))][0]
	except Exception, e:
		print e
		return 0
	return size

def read_SimulationCandles(path):
	# Here we parse SimulationCandles_<version: e.g. CMSSW_3_2_0>.txt which contains
	# release:TODO, release_base [path] - we can put it to release [but it's of different granularity]
	# how to reproduce stuff: TODO

	""" get the acual file """
	SimulationCandles_file = [os.path.join(path, f) for f in os.listdir(path)
				 if os.path.isfile(os.path.join(path, f)) and f.startswith("SimulationCandles_")][0]

	""" read and parse it;  format: #Version     : CMSSW_3_2_0 """
	f = open(SimulationCandles_file, 'r')	
	lines = f.readlines()
	f.close()

	release_version =[[a.strip() for a in line.split(":")] for line in lines if line.startswith("#Version")][0][1]
	return release_version


if __name__ == "__main__":
	import doctest
	doctest.testmod()
	path = path = "/home/vidma/Desktop/CERN_code/cmssw/data/CMSSW_3_2_0_--usersteps=GEN-SIM,DIGI_lxbuild106.cern.ch_relval/relval/CMSSW_3_2_0/workGENSIMDIGI/TTbar_PU_TimeSize"
	print "Job ID: " + str(getJobID_fromTimeReportLogName(os.path.join(path, "TTBAR__DIGI_PILEUP_TimingReport.log")))

	#read_ConfigurationFromSimulationCandles(, step = "DIGI", is_pileup= "PILEUP")



