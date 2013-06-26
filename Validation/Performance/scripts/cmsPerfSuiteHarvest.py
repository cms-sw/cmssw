#!/usr/bin/env python
#A script to "harverst" PerfSuite work directories, producing an xml file with all the data ready to be uploaded to the PerfSuiteDB DB.
import sys, os, re
import getopt
from Validation.Performance.parserTimingReport import *
from Validation.Performance.cmssw_exportdb_xml import *
import Validation.Performance.cmssw_exportdb_xml as cmssw_exportdb_xml
from Validation.Performance.parserPerfsuiteMetadata import parserPerfsuiteMetadata

from Validation.Performance.FileNamesHelper import *
import Validation.Performance.parserEdmSize as parserEdmSize

import glob
from commands import getstatusoutput

""" indicates whether the CMSSW is [use False] available or not. on our testing machine it's not [use True] """
_TEST_RUN = False

""" global variables """
test_timing_report_log = re.compile("TimingReport.log$", re.IGNORECASE)
test_igprof_report_log = re.compile("^(.*)(IgProfMem|IgProfPerf)\.gz", re.IGNORECASE)
test_memcheck_report_log = re.compile("^(.*)memcheck_vlgd.xml", re.IGNORECASE)


xmldoc = minidom.Document()
release = None
steps = {}
candles = {}
pileups = {}

def usage(argv):
    script = argv[0]
    return """
    Usage: %(script)s [-v cmssw_version] [--version=cmssw_version]
    
    if the cmssw version is in the system's environment (after running cmsenv):
    $ %(script)s 
    
    otherwise one must specify the cmssw version:
    $ %(script)s --version=CMSSW_3_2_0
    $ %(script)s -v CMSSW_3_2_0    
    
    """ % locals()

def get_params(argv):
    """ 
    Returns the version of CMSSW to be used which it is taken from:
    * command line parameter or 
    * environment variable 
    in case of error returns None

	And also the directory to put the xml files to: if none --> returns ""
    """
    
    """ try to get the version for command line argument """
    #print argv
    #FIXME: this should be rewritten using getopt properly
    version = None
    #xml_dir = "cmsperfvm:/data/projects/conf/PerfSuiteDB/xml_dropbox" #Set this as default (assume change in write_xml to write to remote machines)
    #NB write_xml is in Validation/Performance/python/cmssw_exportdb_xml.py
    #Setting the default to write to a local directory:
    xml_dir="PerfSuiteDBData"
    try:                              
        opts, args = getopt.getopt(argv[1:], "v:", ["version=", "outdir="])
    except getopt.GetoptError, e:  
        print e
    for opt, arg in opts:
        if opt in ("-v", "--version"):
            version = arg
	if opt == "--outdir":
	     xml_dir = arg
    
    """ if not get it from environment string """
    if not version:
        try:
            version = os.environ["CMSSW_VERSION"]
        except KeyError:
            pass
    
    return (version, xml_dir)
        
def _eventContent_DEBUG(edm_report):
	# for testing / information
	EC_count = {}
	if not _TEST_RUN:
		# count the products in event-content's
		for prod in edm_report:
			ecs = parseEventContent.List_ECs_forProduct(prod)
			for ec in ecs:
				if not EC_count.has_key(ec):
					EC_count[ec] = []	
				EC_count[ec].append(prod)
		#print out the statistics
		for (ec, prods) in EC_count.items():
			print "==== %s EVENT CONTENT: have %d items, the listing is: ===" % (ec, len(prods))
			# list of products
			print "\n *".join(["%(cpp_type)s_%(module_name)s_%(module_label)s" % prod for prod in prods])


def assign_event_content_for_product(product):
	""" returns modified product by adding the event content relationship """

	if not _TEST_RUN:
		product["event_content"] = ",".join(parseEventContent.List_ECs_forProduct(product))
	return product


def get_modules_sequences_relationships():
	(sequenceWithModules, sequenceWithModulesString) =ModuleToSequenceAssign.assignModulesToSeqs()
	return [{"name": seq, "modules": ",".join(modules)} for (seq, modules) in sequenceWithModulesString.items()]


def exportIgProfReport(path, igProfReport, igProfType, runinfo):
    jobID = igProfReport["jobID"]
    #print jobID
    candleLong = os.path.split(path)[1].replace("_IgProf_Perf", "").replace("_IgProf_Mem", "").replace("_PU", "")
    found = False
    #print igProfType
    if runinfo['TestResults'].has_key(igProfType):
        for result in runinfo['TestResults'][igProfType]:
            if candleLong == result["candle"] and jobID["pileup_type"] == result['pileup_type'] and jobID["conditions"] == result['conditions'] and jobID["event_content"] == result['event_content']:
                jobID["candle"] = jobID["candle"].upper()
                if not result.has_key("jobs"):
                    result['jobs'] = []
                result['jobs'].append(igProfReport)
                found = True
                break
		
    if not found:
        print "============ (almost) ERROR: NOT FOUND THE ENTRY in cmsPerfSuite.log, exporting as separate entry ======== "
        print "JOB ID: %s " % str(jobID)
        print " ====================== "
        runinfo['unrecognized_jobs'].append(igProfReport)
        #export_xml(xml_doc = xmldoc, **igProfReport)	
        

def exportTimeSizeJob(path, timeSizeReport,  runinfo):
		candleLong = os.path.split(path)[1].replace("_TimeSize", "").replace("_PU", "")
		jobID = timeSizeReport["jobID"]

		#search for a run Test to which could belong our JOB
		found = False
		if runinfo['TestResults'].has_key('TimeSize'):
			for result in runinfo['TestResults']['TimeSize']:
				#print result
				""" If this is the testResult which fits TimeSize job """
				#TODO: we do not check teh step when assigning because of the different names, check if this is really OK. make a decission which step name to use later, long or short one
				#and jobID["step"] in result['steps'].split(parserPerfsuiteMetadata._LINE_SEPARATOR)
				if result['candle'] == candleLong  and jobID["pileup_type"] == result['pileup_type'] and jobID["conditions"] == result['conditions'] and jobID["event_content"] == result['event_content']:
					#print result
					if not result.has_key("jobs"):
						result['jobs'] = []
					result['jobs'].append(timeSizeReport)
					found = True
					break
		
		if not found:
			print "============ (almost) ERROR: NOT FOUND THE ENTRY in cmsPerfSuite.log, exporting as separate entry ======== "
			print "JOB ID: %s " % str(jobID)
			print " ====================== "
			runinfo['unrecognized_jobs'].append(timeSizeReport)
			#export_xml(xml_doc = xmldoc, **timeSizeReport)	
			
def exportMemcheckReport(path, MemcheckReport, runinfo):
		candleLong = os.path.split(path)[1].replace("_Memcheck", "").replace("_PU", "")
		jobID = MemcheckReport["jobID"]

		#search for a run Test to which could belong our JOB
		found = False
		if runinfo['TestResults'].has_key('Memcheck'):
			for result in runinfo['TestResults']['Memcheck']:
				#print result
                                #print jobID
				""" If this is the testResult which fits Memcheck job """
				#TODO: we do not check teh step when assigning because of the different names, check if this is really OK. make a decission which step name to use later, long or short one
				#and jobID["step"] in result['steps'].split(parserPerfsuiteMetadata._LINE_SEPARATOR)
				if result['candle'] == candleLong  and jobID["pileup_type"] == result['pileup_type'] and jobID["conditions"] == result['conditions'] and jobID["event_content"] == result['event_content']:
					#print result
					if not result.has_key("jobs"):
						result['jobs'] = []
					result['jobs'].append(MemcheckReport)
					found = True
					break
		
		if not found:
			print "============ (almost) ERROR: NOT FOUND THE ENTRY in cmsPerfSuite.log, exporting as separate entry ======== "
			print "JOB ID: %s " % str(jobID)
			print " ====================== "
			runinfo['unrecognized_jobs'].append(MemcheckReport)

def process_timesize_dir(path, runinfo):
	global release,event_content,conditions
	""" if the release is not provided explicitly we take it from the Simulation candles file """
	if (not release):
		release_fromlogfile = read_SimulationCandles(path)
		release  = release_fromlogfile 
		print "release from simulation candles: %s" % release
	
	if (not release):
		# TODO: raise exception!
		raise Exception("the release was not found!")


	""" process the TimingReport log files """

        # get the file list 
	files = os.listdir(path)
	timing_report_files = [os.path.join(path, f) for f in files
				 if test_timing_report_log.search(f) 
					and os.path.isfile(os.path.join(path, f)) ]

	# print timing_report_files
	for timelog_f in timing_report_files:
		print "\nProcessing file: %s" % timelog_f
		print "------- "
		
		jobID = getJobID_fromTimeReportLogName(os.path.join(path, timelog_f))
		print "jobID: %s" % str(jobID)
		(candle, step, pileup_type, conditions, event_content) = jobID
		jobID = dict(zip(("candle", "step", "pileup_type", "conditions", "event_content"), jobID))
		print "Dictionary based jobID %s: " % str(jobID)
		
		#if any of jobID fields except (isPILEUP) is empty we discard the job as all those are the jobID keys and we must have them
		discard = len([key for key, value in jobID.items() if key != "pileup_type" and not value])
		if discard:
			print " ====================== The job HAS BEEN DISCARDED =============== "
			print " NOT ALL DATA WAS AVAILABLE "
			print " JOB ID = %s " % str(jobID)
			print " ======================= end ===================================== "
			continue

		# TODO: automaticaly detect type of report file!!!
		(mod_timelog, evt_timelog, rss_data, vsize_data) =loadTimeLog(timelog_f)
	
		mod_timelog= processModuleTimeLogData(mod_timelog, groupBy = "module_name")
		print "Number of modules grouped by (module_label+module_name): %s" % len(mod_timelog)

		# add to the list to generate the readable filename :)
		steps[step] = 1
		candles[candle] = 1
                if pileup_type=="":
                    pileups["NoPileUp"]=1
                else:
                    pileups[pileup_type] = 1
	
		# root file size (number)
                root_file_size = getRootFileSize(path = path, candle = candle, step = step.replace(':', '='))
                # number of events
                num_events = read_ConfigurationFromSimulationCandles(path = path, step = step, is_pileup = pileup_type)["num_events"]

		#EdmSize
		edm_report = parserEdmSize.getEdmReport(path = path, candle = candle, step = step)
		if edm_report != False:
			try:
				# add event content data
				edm_report  = map(assign_event_content_for_product, edm_report)
				# for testing / imformation
				_eventContent_DEBUG(edm_report)
			except Exception, e:
				print e

		timeSizeReport = {
				"jobID":jobID,
				"release": release,
                                "timelog_result": (mod_timelog, evt_timelog, rss_data, vsize_data), 
				"metadata": {"testname": "TimeSize", "root_file_size": root_file_size, "num_events": num_events}, 
				"edmSize_result": edm_report 
		}
		
		# export to xml: actualy exporting gets suspended and put into runinfo
		exportTimeSizeJob(path, timeSizeReport,  runinfo)

def process_memcheck_dir(path, runinfo):
	global release,event_content,conditions
	""" if the release is not provided explicitly we take it from the Simulation candles file """
	if (not release):
		release_fromlogfile = read_SimulationCandles(path)
		release  = release_fromlogfile 
		print "release from simulation candles: %s" % release
	
	if (not release):
		# TODO: raise exception!
		raise Exception("the release was not found!")

	""" process the vlgd files """

        # get the file list 
	files = os.listdir(path)
	memcheck_files = [os.path.join(path, f) for f in files
				 if test_memcheck_report_log.search(f) 
					and os.path.isfile(os.path.join(path, f)) ]

        if len(memcheck_files) == 0: # Fast protection for old runs, where the _vlgd files is not created...
            print "No _vlgd files found!"
        else:
            for file in memcheck_files:
                jobID = getJobID_fromMemcheckLogName(os.path.join(path, file))

                (candle, step, pileup_type, conditions, event_content) = jobID
                
                print "jobID: %s" % str(jobID)
                jobID = dict(zip(("candle", "step", "pileup_type", "conditions", "event_content"), jobID))

                print "Dictionary based jobID %s: " % str(jobID)
            
                #if any of jobID fields except (isPILEUP) is empty we discard the job as all those are the jobID keys and we must have them
                discard = len([key for key, value in jobID.items() if key != "pileup_type" and not value])
                if discard:
                    print " ====================== The job HAS BEEN DISCARDED =============== "
                    print " NOT ALL DATA WAS AVAILABLE "
                    print " JOB ID = %s " % str(jobID)
                    print " ======================= end ===================================== "
                    continue
            
                # add to the list to generate the readable filename :)
                steps[step] = 1
                candles[candle.upper()] = 1
                if pileup_type=="":
                    pileups["NoPileUp"]=1
                else:
                    pileups[pileup_type] = 1
                
                memerror = getMemcheckError(path)

                MemcheckReport = {
                    "jobID": jobID,
                    "release": release,
                    "memcheck_errors": {"error_num": memerror},
                    "metadata": {"testname": "Memcheck"},
                    }

                # export to xml: actualy exporting gets suspended and put into runinfo
                exportMemcheckReport(path, MemcheckReport, runinfo)

def getMemcheckError(path):
    globbed = glob.glob(os.path.join(path, "*memcheck_vlgd.xml"))

    errnum = 0

    for f in globbed:
        #print f
        cmd = "grep '<error>' "+f+ " | wc -l "
        p = os.popen(cmd, 'r')
        errnum += int(p.readlines()[0])
                        
    return errnum
    

def process_igprof_dir(path, runinfo):
	global release,event_content,conditions
	""" if the release is not provided explicitly we take it from the Simulation candles file """
	if (not release):
		release_fromlogfile = read_SimulationCandles(path)
		release  = release_fromlogfile 
		print "release from simulation candles: %s" % release
	
	if (not release):
		# TODO: raise exception!
		raise Exception("the release was not found!")

	""" process the IgProf sql3 files """

        # get the file list 
	files = os.listdir(path)
	igprof_files = [os.path.join(path, f) for f in files
				 if test_igprof_report_log.search(f) 
					and os.path.isfile(os.path.join(path, f)) ]

        if len(igprof_files) == 0: # No files...
            print "No igprof files found!"
        else:
            for file in igprof_files:
                jobID = getJobID_fromIgProfLogName(file)

                (candle, step, pileup_type, conditions, event_content) = jobID

                print "jobID: %s" % str(jobID)
                jobID = dict(zip(("candle", "step", "pileup_type", "conditions", "event_content"), jobID))
                
                print "Dictionary based jobID %s: " % str(jobID)
                
                igProfType = path.split("/")[-1].replace("TTbar_", "").replace("MinBias_", "").replace("PU_", "")
                
	        #if any of jobID fields except (isPILEUP) is empty we discard the job as all those are the jobID keys and we must have them
                discard = len([key for key, value in jobID.items() if key != "pileup_type" and not value])
                if discard:
                    print " ====================== The job HAS BEEN DISCARDED =============== "
                    print " NOT ALL DATA WAS AVAILABLE "
                    print " JOB ID = %s " % str(jobID)
                    print " ======================= end ===================================== "
                    continue
        
                # add to the list to generate the readable filename :)
                steps[step] = 1
                candles[candle.upper()] = 1
                if pileup_type=="":
                    pileups["NoPileUp"]=1
                else:
                    pileups[pileup_type] = 1
            
                igs = getIgSummary(path)
                #print igs

                igProfReport = {
                    "jobID": jobID,
                    "release": release, 
                    "igprof_result": igs,
                    "metadata": {"testname": igProfType},
                    }

                # print igProfReport
                # export to xml: actualy exporting gets suspended and put into runinfo
                exportIgProfReport(path, igProfReport, igProfType, runinfo)      

#get IgProf summary information from the sql3 files
def getIgSummary(path):
    igresult = []
    globbed = glob.glob(os.path.join(path, "*.sql3"))
    
    for f in globbed:
        #print f
        profileInfo = getSummaryInfo(f)
        if not profileInfo:
            continue
        cumCounts, cumCalls = profileInfo
        dump, architecture, release, rest = f.rsplit("/", 3)
        candle, sequence, pileup, conditions, process, counterType, events = rest.split("___")
        events = events.replace(".sql3", "")
        igresult.append({"counter_type": counterType, "event": events, "cumcounts": cumCounts, "cumcalls": cumCalls})

    #fail-safe(nasty) fix for the diff (even if it gets fixed in the sqls, won't screw this up again...)
    for ig in igresult:
        if 'diff' in ig['event']:
            eventLast,eventOne = ig['event'].split('_diff_')
            for part in igresult:
                if part['counter_type'] == ig['counter_type'] and part['event'] == eventOne:
                    cumcountsOne = part['cumcounts']
                    cumcallsOne = part['cumcalls']
                if part['counter_type'] == ig['counter_type'] and part['event'] == eventLast:
                    cumcountsLast = part['cumcounts']
                    cumcallsLast = part['cumcalls']
            ig['cumcounts'] = cumcountsLast - cumcountsOne
            ig['cumcalls'] = cumcallsLast - cumcallsOne

    return igresult
    
def getSummaryInfo(database):
    summary_query="""SELECT counter, total_count, total_freq, tick_period
                     FROM summary;"""
    error, output = doQuery(summary_query, database)
    if error or not output or output.count("\n") > 1:
        return None
    counter, total_count, total_freq, tick_period = output.split("@@@")
    if counter == "PERF_TICKS":
        return float(tick_period) * float(total_count), int(total_freq)
    else:
        return int(total_count), int(total_freq)
    
def doQuery(query, database):
    if os.path.exists("/usr/bin/sqlite3"):
        sqlite="/usr/bin/sqlite3"
    else:
        sqlite="/afs/cern.ch/user/e/eulisse/www/bin/sqlite"
    return getstatusoutput("echo '%s' | %s -separator @@@ %s" % (query, sqlite, database))

#TimeSize
def searchTimeSizeFiles(runinfo):
	""" so far we will use the current dir to search in """
	path = os.getcwd()
	#print path
	print 'full path =', os.path.abspath(path)

	files = os.listdir(path)
	
	test_timeSizeDirs = re.compile("_TimeSize$", re.IGNORECASE)          
	timesize_dirs = [os.path.join(path, f) for f in files if test_timeSizeDirs.search(f) and os.path.isdir(os.path.join(path, f))]
	
	for timesize_dir in timesize_dirs:
		# print timesize_dir
		process_timesize_dir(timesize_dir, runinfo)

#Memcheck
def searchMemcheckFiles(runinfo):
	""" so far we will use the current dir to search in """
	path = os.getcwd()
	#print path
	print 'full path =', os.path.abspath(path)

	files = os.listdir(path)
	
	test_MemcheckDirs = re.compile("_Memcheck(.*)$", re.IGNORECASE)          
	memcheck_dirs = [os.path.join(path, f) for f in files if test_MemcheckDirs.search(f) and os.path.isdir(os.path.join(path, f))]
	
	for memcheck_dir in memcheck_dirs:
		print memcheck_dir
		process_memcheck_dir(memcheck_dir, runinfo)

#IgProf
def searchIgProfFiles(runinfo):
	""" so far we will use the current dir to search in """
	path = os.getcwd()
	#print path
	print 'full path =', os.path.abspath(path)

	files = os.listdir(path)
	
	test_IgProfDirs = re.compile("_IgProf(.*)$", re.IGNORECASE)          
	igprof_dirs = [os.path.join(path, f) for f in files if test_IgProfDirs.search(f) and os.path.isdir(os.path.join(path, f))]
	
	for igprof_dir in igprof_dirs:
		print igprof_dir
		process_igprof_dir(igprof_dir, runinfo)

def exportSequences():
    """ Exports the sequences to XML Doc """
    try:
    	env_cmssw_version = os.environ["CMSSW_VERSION"]
    except KeyError:
    	print "<<<<<  ====== Error: cannot get CMSSW version [just integrity check for sequences]. \
					 Is the CMSSW environment initialized? (use cmsenv) ==== >>>>"
	env_cmssw_version = None

    print " ==== exporting the sequences. loading files for currently loaded CMSSW version: %s, while the CMSSW we are currently harversting is %s ===" %(env_cmssw_version, release)
    xml_export_Sequences(xml_doc = xmldoc, sequences = get_modules_sequences_relationships(), release=release)



if __name__ == "__main__":
	#searchFiles()
    #TO DO:
    #Use option parser! This is messy.
    
    (release, output_dir) = get_params(sys.argv)

    if not release:
        """ print usage(sys.argv)
        sys.exit(2) """
	print "The version was not provided explicitly, will try to get one from SimulationCandles file """


    # Export the metadata from cmsPerfSuite.log (in current working directory!)
    print "Parsing cmsPerfSuite.log: getting all the metadata concerning the run"
    p = parserPerfsuiteMetadata(os.getcwd())
    run_info = p.parseAll()

    print "Loading Sequences and Event-Content(s). Please wait..."

    Sequences_OK = False
    EventContents_OK = False

    if not _TEST_RUN:
	 try:
		 import Validation.Performance.ModuleToSequenceAssign as ModuleToSequenceAssign
		 Sequences_OK = True
	 except Exception, e:
		print e
	 try:
    	 	import Validation.Performance.parseEventContent as parseEventContent
		EventContents_OK = True
	 except Exception, e:
		print e	

    print "Parsing TimeSize report"
    # Search for TimeSize files: EdmSize, TimingReport
    searchTimeSizeFiles(run_info)
    print "Parsing IgProf report"
    # Search for IgProf files
    searchIgProfFiles(run_info)
    print "Parsing Memcheck report"
    # Search for Memcheck files
    searchMemcheckFiles(run_info)
    #print run_info

    print "Exporting sequences and event-content rules"
    if not _TEST_RUN:
	    """ for testing on laptom we have no CMSSW """
	    # export sequences (for currently loaded CMSSW)
	    if Sequences_OK:
	    	exportSequences()

            if EventContents_OK:
		    # export event's content rules
		    eventContentRules = parseEventContent.getTxtEventContentRules()
		    cmssw_exportdb_xml.exportECRules(xmldoc, eventContentRules)
		    

    cmssw_exportdb_xml.exportRunInfo(xmldoc, run_info, release = release)
    #save the XML file, TODO: change fileName after introducting the JobID
    import datetime
    now = datetime.datetime.now()
    #Changing slightly the XML filename format
    #FIXME: review this convention and archive the xml in a separate CASTOR xml directory for quick recovery of DB...
    file_name = "%s___%s___%s___%s___%s___%s___%s.xml" % (release, "_".join(steps.keys()), "_".join(candles.keys()), "_".join(pileups.keys()),event_content,conditions,now.isoformat())
    print "Writing the output to: %s " % file_name

    write_xml(xmldoc, output_dir, file_name) #change this function to be able to handle directories in remote machines (via tar pipes for now could always revert to rsync later).
    #NB write_xml is in Validation/Performance/python/cmssw_exportdb_xml.py 
