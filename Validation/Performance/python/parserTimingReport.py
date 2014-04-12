#!/usr/bin/env python
import sys
import math
import re
from cmssw_exportdb_xml import *
from FileNamesHelper import *

"""
Performance profiling:
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000   71.618   71.618 <stdin>:1(foo)
        1    0.315    0.315   71.933   71.933 <string>:1(<module>)
        1   47.561   47.561   71.618   71.618 parserTimingReport.py:27(loadTimeLog)
     8000    0.191    0.000    0.343    0.000 parserTimingReport.py:8(extractRSS_VSIZE)
        1    0.000    0.000    0.000    0.000 {len}
  2384000    3.239    0.000    3.239    0.000 {method 'append' of 'list' objects}
        1    0.000    0.000    0.000    0.000 {method 'close' of 'file' objects}
        1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
    24000    0.041    0.000    0.041    0.000 {method 'partition' of 'str' objects}
  2392000    5.804    0.000    5.804    0.000 {method 'split' of 'str' objects}
 10791332   14.782    0.000   14.782    0.000 {method 'strip' of 'str' objects}
        1    0.000    0.000    0.000    0.000 {method 'xreadlines' of 'file' objects}
        1    0.000    0.000    0.000    0.000 {open}

"""


""" given two lines returns the VSIZE and RSS values along with event number """
def extractRSS_VSIZE(line1, line2, record_number):
	""" 
	>>> extractRSS_VSIZE("%MSG-w MemoryCheck:  PostModule 19-Jun-2009 13:06:08 CEST Run: 1 Event: 1", \
			     "MemoryCheck: event : VSIZE 923.07 0 RSS 760.25 0")
	(('1', '760.25'), ('1', '923.07'))
	"""

	if ("Run" in line1) and  ("Event" in line1): # the first line
		event_number = line1.split('Event:')[1].strip()	
	else: return False

	""" it's first or second MemoryCheck line """
	if ("VSIZE" in line2) and ("RSS" in line2): # the second line
		RSS = line2.split("RSS")[1].strip().split(" ")[0].strip() #changed partition into split for backward compatability with py2.3
		VSIZE = line2.split("RSS")[0].strip().split("VSIZE")[1].strip().split(" ")[0].strip()
		#Hack to return the record number instea of event number for now... can always switch back of add event number on top
		#return ((event_number, RSS), (event_number, VSIZE))
		return ((record_number, RSS), (record_number, VSIZE))
	else: return False


def loadTimeLog(log_filename, maxsize_rad = 0): #TODO: remove maxsize to read, used for debugging
	""" gets the timing data from the logfile
	 returns 4 lists:

		* ModuleTime data (event_number, module_label, module_name, seconds) and
		* EventTime data
		 	- with granularity of event (initial - not processed data)
		* RSS per event
		* VSIZE per event
	 """
	# ----- format of logfile ----
	# Report columns headings for modules: eventnum runnum modulelabel modulename timetakeni"
	# e.g. TimeModule> 1 1 csctfDigis CSCTFUnpacker 0.0624561

	mod_data = []
	evt_data = []
	rss_data = []
	vsize_data = []
	# open file and read it and fill the structure!
	logfile = open(log_filename, 'r')
	
	# get only the lines which have time report data
	#TODO: reading and processing line by line might speed up the process!
				
	memcheck_line1 = False

	record_number=0
	last_record=0
	last_event=0
	for line in logfile.xreadlines():
		if 'TimeModule>' in line.strip():
			line = line.strip()
			line_content_list = line.split(' ')[0:]
			#Hack to avoid issues with the non-consecutive run numbers:
			event_number = int(line_content_list[1])
			if event_number != last_event:
				record_number=record_number+1
				last_event=event_number
				# module label and name were mixed up in the original doc
			module_label = str(line_content_list[4])
			module_name = str(line_content_list[3])
			seconds = float(line_content_list[5])
			#For now let's try to switch to the record_number... if we need to also have the event_number we can always add it back.
			#mod_data.append((event_number, module_label, module_name, seconds))
			mod_data.append((record_number, module_label, module_name, seconds))
		if 'TimeEvent>' in line.strip():
			line = line.strip()
			line_content_list = line.split(' ')[0:]
			#Hack to avoid issues with the non-consecutive run numbers:
			event_number = int(line_content_list[1])
			if event_number != last_event:
				record_number=record_number+1
				last_event=event_number
				# module label and name were mixed up in the original doc
			time_seconds = str(line_content_list[3])
			
			#TODO: what are the other [last two] numbers? Real time? smf else? TimeEvent> 1 1 15.3982 13.451 13.451
			#For now let's try to switch to the record_number... if we need to also have the event_number we can always add it back.
			#evt_data.append((event_number, time_seconds))
			evt_data.append((record_number, time_seconds))
		""" 
			%MSG-w MemoryCheck:  PostModule 19-Jun-2009 13:06:08 CEST Run: 1 Event: 1
			MemoryCheck: event : VSIZE 923.07 0 RSS 760.25 0
		"""
		if 'MemoryCheck:' in line.strip():
			# this is the first line out of two
			if (not memcheck_line1):
				memcheck_line1 = line.strip()
			else:
				#FIXME (eventually)
				#Hacking in the record_number extracted from the TimeEvent and TimeModule parsing... NOT ROBUST...
				(rss, vsize) = extractRSS_VSIZE(memcheck_line1, line.strip(), record_number)
				rss_data.append(rss)
				vsize_data.append(vsize)
		else: 
			memcheck_line1 = False 																																				

	logfile.close()
	
	return (mod_data, evt_data, rss_data, vsize_data)




def calcRMS(items,avg):
	""" returns RootMeanSquare  of items in a list """ 
	# sqrt(sum(x^2))
	# Not statistics RMS... "physics" RMS, i.e. standard deviation: sqrt(sum((x-avg)**2)/N)
	# return math.sqrt(reduce(lambda x: (x - avg)**2, items) / len(items))	
	return math.sqrt(sum([(x-avg)**2 for x in items])/len(items))

def calc_MinMaxAvgRMS(items, remove_first = True, f_time = lambda x: x[0], f_evt_num = lambda x: x[1],):
	""" returns a dict of avg, min, max, rms """
	# save the cpu time of first event before removing the first result!
	cpu_time_first = f_time(items[0])

	if len(items) > 1 and remove_first == True:
		items.remove(items[0]) #TODO: if there is only one event - we have a problem -> do we eliminate the whole module?
		# TODO: it removes it completely from all the data because we do not save/ do not copy it

	items_time = map(f_time, items)
	min_value = min(items_time)
	max_value = max(items_time)
	max_index = items_time.index(max_value)
	avg_value = float(sum(items_time)) / float(len(items_time))
	rms_value = calcRMS(items_time,avg_value)

	return {"min":  min_value, "max": max_value, "cputime_first": cpu_time_first,
			"rms": rms_value, "avg": avg_value,
			"event_number_of_max": f_evt_num(items[max_index])}
	
	
def processModuleTimeLogData(modules_timelog, groupBy = "module_name"):
	""" Processes the timelog data grouping events by module and calculates min, max, avg, rms 
	Returns data as a list of dicts like: !
	
	 {
	 	<module_name>: 
	 		{name:, label:, 
	 			stats: {num_events, avg, min, max, rms} 
	 } 
	 
	 """
	# group by module_name, we save a list for each module name
	times_bymod = {}
	
	# print "Num of useful TimeLog lines: %s" % len(modules_timelog)
	
	for time_data in modules_timelog:
		(event_number, module_label, module_name, seconds) = time_data
		
		# group times of modules By label or name, TODO: maybe both 
		if groupBy == "module_label":
			key = module_label
		else:
			if groupBy =="name+label": 
				key = module_name + "_" + module_label
			else:
				key = module_name

			
		try:
			# is the list for current module initialized?
			times_bymod[key]
		except KeyError:
			#Changing this from a list to a dict (see comments below):
			#times_bymod[key] = []
			times_bymod[key] = {}
		#Running out of memory!
		#times_bymod[key].append({"label": module_label, "name": module_name, "time": seconds, "event_number": event_number})
		#Let's do it right:
		#Instead of times_bymod[key]=[{"label": module_label, "name": module_name, "time": seconds, "event_number": event_number}]
		#let's do times_bymod[key]={"module_label":{"module_name":[(seconds,event_number)]}} so we do not repeat label and name and especially they are not a pair of key/value
		#During the first event all the keys will be initialized, then from event 2 on it will be just appending the (seconds,event_number) tuple to the list with the appropriate keys:

		#Check/Set up the module label dict:
		try:
			times_bymod[key][module_label]
		except KeyError:
			times_bymod[key].update({module_label:{}})
		
		#Check/Set up the module name dict:
		try:
			times_bymod[key][module_label][module_name]
		except KeyError:
			times_bymod[key][module_label].update({module_name:[]})
		
		#We're now ready to add the info as a tuple in the list!
		times_bymod[key][module_label][module_name].append((seconds,event_number))
		
		
	# calculate Min,Max, Avg, RMS for each module and in this way get the final data to be imported
	##for mod_name in times_bymod.keys():
	##	#copy needed data
	##	#mod_data = {"label": times_bymod[mod_name][0]["label"], "name": times_bymod[mod_name][0]["name"]}
	##	#New data structure:
	##	mod_data = {"label":times_bymod[mod_name].keys()[0],"name":times_bymod[mod_name][times_bymod[mod_name].keys()[0]].keys()[0]}
	##	# add statistical data
	##
	##	mod_data["stats"] =calc_MinMaxAvgRMS(f_time = lambda x: x["time"], f_evt_num = lambda x: x["event_number"], items = times_bymod[mod_name])
	##
	##	mod_data["stats"]["num_events"] = len(times_bymod[mod_name])
	##	
	##	times_bymod[mod_name] = mod_data
	#Let's rewrite this using the dictionary we now have without any logical change (could do with some...):
	for key in times_bymod.keys():
		for label in times_bymod[key].keys():
			mod_data={'label':label}
			for name in times_bymod[key][label].keys():
				mod_data.update({'name':name})
				mod_data['stats']= calc_MinMaxAvgRMS(f_time= lambda x:x[0],f_evt_num=lambda x:x[1],items=times_bymod[key][label][name])
				mod_data['stats']['num_events']=len(times_bymod[key][label][name])
		times_bymod[key]=mod_data
	return times_bymod

def manual_run():
	timelog_f = "TTBAR__RAW2DIGI,RECO_TimingReport.log"
	timelog_f = "TTBAR__GEN,SIM,DIGI,L1,DIGI2RAW,HLT_TimingReport.log"	
	#TODO: get STEP name from filename
	release_files = {
			 
			 "CMSSW_3_1_0_pre9": 
			 (
			 "CMSSW_3_1_0_pre9/MINBIAS__RAW2DIGI,RECO_TimingReport.log", 
			  "CMSSW_3_1_0_pre9/TTBAR__RAW2DIGI,RECO_TimingReport.log")
			 ## "CMSSW_3_1_0_pre10": 
			 }
	for release, files in release_files.items():
		print "Processing release: %s" % release
		for timelog_f in files:
			print "Processing file: %s" % timelog_f
			
			# TODO: automaticaly detect type of report file!!!
			(mod_timelog, evt_timelog, rss_data, vsize_data) =loadTimeLog(timelog_f)
			
			mod_timelog= processModuleTimeLogData(mod_timelog, groupBy = "module_label")
			print "Number of modules grouped by (module_label): %s" % len(mod_timelog)

			(candle, step, pileup_type, conditions, event_content) = getJobID_fromTimeReportLogName(timelog_f)
			
			""" We could get release from the path but that's quite ugly! """
			export_xml(jobID = jobID, release=release, timelog_result=(mod_timelog, evt_timelog, rss_data, vsize_data))	

""" use to run performance profiling """
def perf_profile():
	timelog_f = "test_data/TTBAR__RAW2DIGI,RECO_TimingReport.log"
	(modules_timelog, evt_timelog, rss_data, vsize_data) = loadTimeLog(timelog_f)   

	mod_timelog= processModuleTimeLogData(modules_timelog, groupBy = "module_label")

	(candle, step, pileup_type, conditions, event_content) = getJobID_fromTimeReportLogName(timelog_f)
	
	xmldoc = minidom.Document()
	export_xml(step = step, candle = candle, release="test", timelog_result=(mod_timelog, evt_timelog, rss_data, vsize_data), xml_doc = xmldoc)	
	write_xml(xmldoc, "test_xml_output.xml")
        
if (__name__ == "__main__"):
	perf_profile()
