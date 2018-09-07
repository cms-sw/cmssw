from __future__ import print_function
import re
import os, sys
import time
import parsingRulesHelper
import glob
from commands import getstatusoutput

class parserPerfsuiteMetadata:
    """ 
            The whole parsing works as following. We split the file into 3 parts (we keep 3 variables of line lists:self.lines_general, self.lines_timesize, self.lines_other ):

                    * General info
            As most of the info are simple one line strings, we define some regular expressions defining and matching each of those lines. The regular expressions are associated with data which we can get from them. e.g. ^Suite started at (.+) on (.+) by user (.+)$ would match only the line defining the time suite started and on which machine. It's associated with tuple of field names for general info which will be filled in. in this way we get info = {'start_time': start-taken-from-regexp, 'host': host, 'user': user}. This is done by calling simple function _applyParsingRules which checks each lines with each if one passes another, if it does fills in the result dictionary with the result.
            Additionaly we get the cpu and memmory info from /proc/cpuinfo /proc/meminfo

                    * TimeSize test
            We use the same technique a little bit also. But at first we divide the timesize lines by job (individual run of cmssw - per candle, and pileup/not). Then for each of the jobs we apply our parsing rules, also we find the starting and ending times (i.e. We know that start timestamp is somethere after certain line containing "Written out cmsRelvalreport.py input file at:")

                    * All other tests
            We find the stating that the test is being launched (containing the test name, core and num events). Above we have the thread number, and below the starting time.
            The ending time can be ONLY connected with the starting time by the Thread-ID. The problem is that the file names different the same test instance like <Launching "PILE UP Memcheck"> and <"Memcheck" stopped>.
    """
    _LINE_SEPARATOR = "|"
    def validateSteps(self, steps):
        """ Simple function for error detection. TODO: we could use a list of possible steps also """
        return not (not steps or len(steps) > self._MAX_STEPS)

    def __init__(self, path):

        self._MAX_STEPS  = 5 # MAXIMUM NUMBER OF STEPS PER RUN (taskset relvalreport.py...)
        self._DEBUG = False


        self._path = path

        """ some initialisation to speedup the other functions """
        #for cmsscimark
        self.reCmsScimarkTest = re.compile(r"""^Composite Score:(\s*)([^\s]+)$""")

        #TimeSize
        """ the separator for beginning of timeSize / end of general statistics """
        self._timeSizeStart = re.compile(r"""^Launching the TimeSize tests \(TimingReport, TimeReport, SimpleMemoryCheck, EdmSize\) with (\d+) events each$""")
        """ (the first timestamp is the start of TimeSize) """


        """ the separator for end of timeSize / beginning of IgProf_Perf, IgProf_Mem,  Memcheck, Callgrind tests """
        self._timeSizeEnd = re.compile(r"""^Stopping all cmsScimark jobs now$""")

        #Other tests:
        self._otherStart = re.compile(r"^Preparing")

        """ 
		----- READ THE DATA -----
		"""
        lines = self.readInput(path)
        """ split the whole file  into parts """
        #Let's not assume there are ALWAYS TimeSize tests in the runs of the Performance Suite!:
        #Check first:  
        #FIXME: Vidmantas did not think to this case... will need to implement protectionb against it for all the IB tests...
        #To do as soon as possible...
        #Maybe revisit the strategy if it can be done quickly.
        timesize_end= [lines.index(line)  for line in lines if self._timeSizeEnd.match(line)]
        if timesize_end:
            timesize_end_index = timesize_end[0]
        else:
            timesize_end_index=0
        timesize_start=[lines.index(line) for line in lines if self._timeSizeStart.match(line)]
        general_stop=[lines.index(line) for line in lines if self._otherStart.match(line)]
        if timesize_start:
            timesize_start_index = timesize_start[0]
            general_stop_index = timesize_start_index
        elif general_stop:
            timesize_start_index=timesize_end_index+1
            general_stop_index=general_stop[0]
        else:
            timesize_start_index=0
            general_stop_index=-1

        """ we split the structure:
			* general
			* timesize
			* all others [igprof etc]
		"""

        """ we get the indexes of spliting """
        #Not OK to use timsize_start_index for the general lines... want to be general, also to cases of no TimeSize tests...
        #self.lines_general = lines[:timesize_start_index]
        self.lines_general = lines[:general_stop_index]
        self.lines_timesize = lines[timesize_start_index:timesize_end_index+1]
        self.lines_other = lines[timesize_end_index:]		

        """ a list of missing fields """
        self.missing_fields = []

    @staticmethod
    def isTimeStamp(line):
        """
        Returns whether the string is a timestamp (if not returns None)

        >>> parserPerfsuiteMetadata.isTimeStamp("Fri Aug 14 01:16:03 2009")
        True
        >>> parserPerfsuiteMetadata.isTimeStamp("Fri Augx 14 01:16:03 2009")

        """
        datetime_format = "%a %b %d %H:%M:%S %Y" # we use default date format
        try:
            time.strptime(line, datetime_format)
            return True
        except ValueError:
            return None

    @staticmethod
    def findFirstIndex_ofStartsWith(job_lines, start_of_line):
        return [job_lines.index(line) 
                for line in job_lines 
                if line.startswith(start_of_line)][0]

    def findLineBefore(self, line_index, lines, test_condition):
        """ finds a line satisfying the `test_condition` comming before the `line_index` """
        # we're going backwards the lines list
        for line_index in  xrange(line_index -1, -1, -1):
            line = lines[line_index]

            if test_condition(line):
                return line
        raise ValueError


    def findLineAfter(self, line_index, lines, test_condition, return_index = False):
        """ finds a line satisfying the `test_condition` comming after the `line_index` """
        # we're going forward the lines list
        for line_index in xrange(line_index + 1, len(lines)):
            line = lines[line_index]

            if test_condition(line):	
                if return_index:
                    return line_index
                return line

    def firstTimeStampBefore(self, line_index, lines):
        """ returns the first timestamp BEFORE the line with given index """

        return self.findLineBefore(line_index, lines, test_condition = self.isTimeStamp)

    def firstTimeStampAfter(self, line_index, lines):
        """ returns the first timestamp AFTER the line with given index """

        return self.findLineAfter(line_index, lines, test_condition = self.isTimeStamp)

    def handleParsingError(self, message):
        if self._DEBUG:
            raise ValueError(message)
        print(" ======== AND ERROR WHILE PARSING METADATA ====")
        print(message)
        print(" =============== end ========================= ")

    #IgProf_Perf, IgProf_Mem,  Memcheck, Callgrind
    #TODO: divide the input using separators

    """ reads the input cmsPerfsuite.log file  """
    def readInput(self, path, fileName = "cmsPerfSuite.log"):
        try:
            f = open(os.path.join(path, fileName), "r")
            lines =  [s.strip() for s in f.readlines()]
            f.close()
        except IOError:
            lines = []

        #print self._lines
        return lines




    def getMachineInfo(self):
        """ Returns the cpu and memory info  """

        """ cpu info """

        """
 		we assume that:
		 * num_cores = max(core id+1) [it's counted from 0]
		 * 'model name' is processor type [we will return only the first one - we assume others to be same!!??
		 * cpu MHz - is the speed of CPU
		"""
        #TODO: BUT cpu MHz show not the maximum speed but current, 
        """
		for 
			model name	: Intel(R) Core(TM)2 Duo CPU     L9400  @ 1.86GHz
			cpu MHz		: 800.000
			cache size	: 6144 KB
		"""
        cpu_result = {}
        try:
            f= open(os.path.join(self._path, "cpuinfo"), "r")

            #we split data into a list of tuples = [(attr_name, attr_value), ...]
            cpu_attributes = [l.strip().split(":") for l in f.readlines()]
            #print cpu_attributes
            f.close()
            cpu_result = {
                    "num_cores": max ([int(attr[1].strip())+1 for attr in cpu_attributes if attr[0].strip() == "processor"]), #Bug... Vidmantas used "core id"
                    "cpu_speed_MHZ": max ([attr[1].strip() for attr in cpu_attributes if attr[0].strip() == "cpu MHz"]),
                    "cpu_cache_size": [attr[1].strip() for attr in cpu_attributes if attr[0].strip() == "cache size"][0],
                    "cpu_model_name": [attr[1].strip() for attr in cpu_attributes if attr[0].strip() == "model name"][0]
            }
        except IOError as e:
            print(e)





        """ memory info """
        mem_result = {}

        try:
            f= open(os.path.join(self._path, "meminfo"), "r")

            #we split data into a list of tuples = [(attr_name, attr_value), ...]
            mem_attributes = [l.strip().split(":") for l in f.readlines()]

            mem_result = {
                    "memory_total_ram": [attr[1].strip() for attr in mem_attributes if attr[0].strip() == "MemTotal"][0]
            }

        except IOError as e:
            print(e)

        cpu_result.update(mem_result)
        return cpu_result



    def _applyParsingRules(self, parsing_rules, lines):
        """ 
                Applies the (provided) regular expression rules (=rule[1] for rule in parsing_rules)
                to each line and if it matches the line,
                puts the mached information to the dictionary as the specified keys (=rule[0]) which is later returned
                Rule[3] contains whether the field is required to be found. If so and it isn't found the exception would be raised.
                rules = [
                  ( (field_name_1_to_match, field_name_2), regular expression, /optionaly: is the field required? if so "req"/ )
                ]
         """
        """ we call a shared parsing helper """
        #parsing_rules = map(parsingRulesHelper.rulesRegexpCompileFunction, parsing_rules)
        #print parsing_rules
        (info, missing_fields) = parsingRulesHelper.rulesParser(parsing_rules, lines, compileRules = True)

        self.missing_fields.extend(missing_fields)

        return info


    def parseGeneralInfo(self):
        lines = self.lines_general
        """ we define a simple list (tuple) of rules for parsing, the first part tuple defines the parameters to be fetched from the
			regexp while the second one is the regexp itself """
        #TIP: don't forget that tuple of one ends with ,
        parsing_rules = (
                (("", "num_cores", "run_on_cpus"), r"""^This machine \((.+)\) is assumed to have (\d+) cores, and the suite will be run on cpu \[(.+)\]$"""),
                (("start_time", "host", "local_workdir", "user"), r"""^Performance Suite started running at (.+) on (.+) in directory (.+), run by user (.+)$""", "req"),
                (("architecture",) ,r"""^Current Architecture is (.+)$"""),
                (("test_release_based_on",), r"""^Test Release based on: (.+)$""", "req"),
                (("base_release_path",) , r"""^Base Release in: (.+)$"""),
                (("test_release_local_path",) , r"""^Your Test release in: (.+)$"""),

                (("castor_dir",) , r"""^The performance suite results tarball will be stored in CASTOR at (.+)$"""),

                (("TimeSize_events",) , r"""^(\d+) TimeSize events$"""),
                (("IgProf_events",) , r"""^(\d+) IgProf events$"""),
                (("CallGrind_events",) , r"""^(\d+) Callgrind events$"""),
                (("Memcheck_events",) , r"""^(\d+) Memcheck events$"""), 

                (("candles_TimeSize",) , r"""^TimeSizeCandles \[(.*)\]$"""),
                (("candles_TimeSizePU",) , r"""^TimeSizePUCandles \[(.*)\]$"""),

                (("candles_Memcheck",) , r"""^MemcheckCandles \[(.*)\]$"""),
                (("candles_MemcheckPU",) , r"""^MemcheckPUCandles \[(.*)\]$"""),

                (("candles_Callgrind",) , r"""^CallgrindCandles \[(.*)\]$"""),
                (("candles_CallgrindPU",) , r"""^CallgrindPUCandles \[(.*)\]$"""),

                (("candles_IgProfPU",) , r"""^IgProfPUCandles \[(.*)\]$"""),
                (("candles_IgProf",) , r"""^IgProfCandles \[(.*)\]$"""),


                (("cmsScimark_before",) , r"""^(\d+) cmsScimark benchmarks before starting the tests$"""),
                (("cmsScimark_after",) , r"""^(\d+) cmsScimarkLarge benchmarks before starting the tests$"""),
                (("cmsDriverOptions",) , r"""^Running cmsDriver.py with user defined options: --cmsdriver="(.+)"$"""),

                (("HEPSPEC06_SCORE",) ,r"""^This machine's HEPSPEC06 score is: (.+)$"""),


        )
        """ we apply the defined parsing rules to extract the required fields of information into the dictionary (as defined in parsing rules) """
        info = self._applyParsingRules(parsing_rules, lines)


        """ postprocess the candles list """
        candles = {}
        for field, value in info.items():
            if field.startswith("candles_"):
                test = field.replace("candles_", "")
                value = [v.strip(" '") for v in value.split(",")]
                #if value:
                candles[test]=value
                del info[field]
        #print candles
        info["candles"] = self._LINE_SEPARATOR.join([k+":"+",".join(v) for (k, v) in candles.items()])


        """ TAGS """
        """ 
		--- Tag ---    --- RelTag --- -------- Package --------                        
		HEAD           V05-03-06      IgTools/IgProf                                   
		V01-06-05      V01-06-04      Validation/Performance                           
		---------------------------------------
		total packages: 2 (2 displayed)
		"""
        tags_start_index = -1 # set some default
        try:
            tags_start_index = [i for i in xrange(0, len(lines)) if lines[i].startswith("--- Tag ---")][0]
        except:
            pass
        if tags_start_index > -1:
            tags_end_index = [i for i in xrange(tags_start_index + 1, len(lines)) if lines[i].startswith("---------------------------------------")][0]
            # print "tags start index: %s, end index: %s" % (tags_start_index, tags_end_index)
            tags = lines[tags_start_index:tags_end_index+2]
            # print [tag.split("  ") for tag in tags]
            # print "\n".join(tags)
        else: # no tags found, make an empty list ...
            tags = []
        """ we join the tags with separator to store as simple string """
        info["tags"] = self._LINE_SEPARATOR.join(tags)
        #FILES/PATHS


        """ get the command line """
        try:
            cmd_index = self.findFirstIndex_ofStartsWith(lines, "Performance suite invoked with command line:") + 1 #that's the next line
            info["command_line"] = 	lines[cmd_index]
        except IndexError as e:
            if self._DEBUG:
                print(e)
            info["command_line"] = 	""

        try:
            cmd_parsed_start = self.findFirstIndex_ofStartsWith(lines, "Initial PerfSuite Arguments:") + 1
            cmd_parsed_end = self.findFirstIndex_ofStartsWith(lines, "Running cmsDriver.py")
            info["command_line_parsed"] = self._LINE_SEPARATOR.join(lines[cmd_parsed_start:cmd_parsed_end])
        except IndexError as e:
            if self._DEBUG:
                print(e)
            info["command_line"] = 	""

        return  info


    def parseAllOtherTests(self):
        #make it general, for whatever test comes...
        test = {}

        parsing_rules = (
                (("", "candle", ), r"""^(Candle|ONLY) (.+) will be PROCESSED$""", "req"),
                #e.g.: --conditions FrontierConditions_GlobalTag,MC_31X_V4::All --eventcontent RECOSIM
                (("cms_driver_options", ), r"""^Using user-specified cmsDriver.py options: (.+)$"""),
                (("", "conditions", ""), r"""^Using user-specified cmsDriver.py options: (.*)--conditions ([^\s]+)(.*)$""", "req"),
                # for this we cannot guarrantee that it has been found, TODO: we might count the number of pileup candles and compare with arguments
                (("",  "pileup_type", ""), r"""^Using user-specified cmsDriver.py options:(.*)--pileup=([^\s]+)(.*)$"""),
                #not shure if event content is required
                (("",  "event_content", ""), r"""^Using user-specified cmsDriver.py options:(.*)--eventcontent ([^\s]+)(.*)$""", "req"),
                #TODO: after changeing the splitter to "taskset -c ..." this is no longer included into the part of correct job
                #(("input_user_root_file", ), r"""^For these tests will use user input file (.+)$"""),
        )


        lines = self.lines_other
        """

		for each of IgProf_Perf, IgProf_Mem,  Memcheck, Callgrind tests we have such a structure of input file:
		* beginning ->> and start timestamp- the firstone:
        		Launching the PILE UP IgProf_Mem tests on cpu 4 with 201 events each
			Adding thread <simpleGenReportThread(Thread-1, started -176235632)> to the list of active threads
		        Mon Jun 14 20:06:54 2010

			<... whatever might be here, might overlap with other test start/end messages ..>

			Mon Jun 14 21:59:33 2010
			IgProf_Mem test, in thread <simpleGenReportThread(Thread-1, stopped -176235632)> is done running on core 4

		* ending - the last timestamp "before is done running ...."
		"""
        # we take the first TimeStamp after the starting message and the first before the finishing message in 2 rounds..

        #TODO: if threads would be changed it would stop working!!!

        # i.e. Memcheck, cpu, events
        reSubmit = re.compile(r"""^Let's submit (.+) test on core (\d+)$""")

        reStart = re.compile(r"""^Launching the (PILE UP |)(.*) tests on cpu (\d+) with (\d+) events each$""")

        # i.e. Memcheck, thread name,id,core number
        reEnd = re.compile(r"""^(.*) test, in thread <simpleGenReportThread\((.+), stopped -(\d+)\)> is done running on core (\d+)$""")

        reAddThread =  re.compile(r"""^Adding thread <simpleGenReportThread\((.+), started -(\d+)\)> to the list of active threads$""")

        reWaiting = re.compile(r"""^Waiting for tests to be done...$""")

        reExitCode = re.compile(r"""Individual cmsRelvalreport.py ExitCode (\d+)""")
        """ we search for lines being either: (it's a little pascal'ish but we need the index!) """

        jobs = []

        #can split it into jobs ! just have to reparse it for the exit codes later....
        for line_index in xrange(0, len(lines)):
            line = lines[line_index]
            if reSubmit.match(line):
                end_index = self.findLineAfter(line_index, lines, test_condition=lambda l: reWaiting.match(l), return_index = True)
                jobs.append(lines[line_index:end_index])

        for job_lines in jobs:
            #print job_lines
            info = self._applyParsingRules(parsing_rules, job_lines)
            #Fixing here the compatibility with new cmsdriver.py --conditions option
            #(for which now we have autoconditions and FrontierConditions_GlobalTag is optional):
            if 'auto:' in info['conditions']:
                from Configuration.AlCa.autoCond import autoCond
                info['conditions'] = autoCond[ info['conditions'].split(':')[1] ].split("::")[0]
            else:
                if 'FrontierConditions_GlobalTag' in info['conditions']:
                    info['conditions']=info['conditions'].split(",")[1]

            steps_start = self.findFirstIndex_ofStartsWith(job_lines, "You defined your own steps to run:")
            steps_end = self.findFirstIndex_ofStartsWith(job_lines, "*Candle ")
            #probably it includes steps until we found *Candle... ?
            steps = job_lines[steps_start + 1:steps_end]
            if not self.validateSteps(steps):
                self.handleParsingError( "Steps were not found corrently: %s for current job: %s" % (str(steps), str(job_lines)))

                """ quite nasty - just a work around """
                print("Trying to recover from this error in case of old cmssw")

                """ we assume that steps are between the following sentance and a TimeStamp """
                steps_start = self.findFirstIndex_ofStartsWith(job_lines, "Steps passed to writeCommands")
                steps_end = self.findLineAfter(steps_start, job_lines, test_condition = self.isTimeStamp, return_index = True)

                steps = job_lines[steps_start + 1:steps_end]
                if not self.validateSteps(steps):
                    self.handleParsingError( "EVEN AFTER RECOVERY Steps were not found corrently! : %s for current job: %s" % (str(steps), str(job_lines)))
                else:
                    print("RECOVERY SEEMS to be successful: %s" % str(steps))

            info["steps"] = self._LINE_SEPARATOR.join(steps) #!!!! STEPS MIGHT CONTAIN COMMA: ","

            start_id_index = self.findLineAfter(0, job_lines, test_condition = reStart.match, return_index = True)
            pileUp, testName, testCore, testEventsNum = reStart.match(job_lines[start_id_index]).groups()			
            info["testname"] = testName

            thread_id_index = self.findLineAfter(0, job_lines, test_condition = reAddThread.match, return_index = True)
            info["start"] = self.firstTimeStampAfter(thread_id_index, job_lines)

            thread_id, thread_number = reAddThread.match(job_lines[thread_id_index]).groups()
            info["thread_id"] = thread_id

            if testName not in test:
                test[testName] = []
            test[testName].append(info)

        for line_index in xrange(0, len(lines)):
            line = lines[line_index]

            if reEnd.match(line):
                testName, thread_id, thread_num, testCore = reEnd.match(line).groups()
                time = self.firstTimeStampBefore(line_index, lines)
                try:
                    exit_code = ""
                    #we search for the exit code
                    line_exitcode = self.findLineBefore(line_index, lines, test_condition=lambda l: reExitCode.match(l))
                    exit_code, = reExitCode.match(line_exitcode).groups()
                except Exception as e:
                    print("Error while getting exit code (Other test): %s" + str(e))

                for key, thread in test.items():
                    for i in range(0, len(thread)):
                        if thread[i]["thread_id"] == thread_id:
                            thread[i].update({"end": time, "exit_code": exit_code})
                            break

        return test


    def parseTimeSize(self):
        """ parses the timeSize """
        timesize_result = []

        # TODO: we will use the first timestamp after the "or these tests will use user input file..."
        #TODO: do we have to save the name of input file somewhere?
        """
		the structure of input file:
		* beginning ->> and start timestamp- the firstone:		
			>>> [optional:For these tests will use user input file /build/RAWReference/MinBias_RAW_320_IDEAL.root]
			<...>
			Using user-specified cmsDriver.py options: --conditions FrontierConditions_GlobalTag,MC_31X_V4::All --eventcontent RECOSIM
			Candle MinBias will be PROCESSED
			You defined your own steps to run:
			RAW2DIGI-RECO
			*Candle MinBias
			Written out cmsRelvalreport.py input file at:
			/build/relval/CMSSW_3_2_4/workStep2/MinBias_TimeSize/SimulationCandles_CMSSW_3_2_4.txt
			Thu Aug 13 14:53:37 2009 [start]
			<....>
			Thu Aug 13 16:04:48 2009 [end]
			Individual cmsRelvalreport.py ExitCode 0
		* ending - the last timestamp "... ExitCode ...."
		"""
        #TODO: do we need the cmsDriver --conditions? I suppose it would the global per work directory = 1 perfsuite run (so samefor all candles in one work dir)
        # TODO: which candle definition to use?
        """ divide into separate jobs """
        lines = self.lines_timesize
        jobs = []
        start = False
        timesize_start_indicator = re.compile(r"""^taskset -c (\d+) cmsRelvalreportInput.py""")
        for line_index in xrange(0, len(lines)):
            line = lines[line_index]
            # search for start of each TimeSize job (with a certain candle and step)
            if timesize_start_indicator.match(line):
                if start:
                    jobs.append(lines[start:line_index])
                start = line_index
        #add the last one
        jobs.append(lines[start:len(lines)])
        #print "\n".join(str(i) for i in jobs)

        parsing_rules = (
                (("", "candle", ), r"""^(Candle|ONLY) (.+) will be PROCESSED$""", "req"),
                #e.g.: --conditions FrontierConditions_GlobalTag,MC_31X_V4::All --eventcontent RECOSIM
                (("cms_driver_options", ), r"""^Using user-specified cmsDriver.py options: (.+)$"""),
                (("", "conditions", ""), r"""^Using user-specified cmsDriver.py options: (.*)--conditions ([^\s]+)(.*)$""", "req"),
                # for this we cannot guarrantee that it has been found, TODO: we might count the number of pileup candles and compare with arguments
                (("",  "pileup_type", ""), r"""^Using user-specified cmsDriver.py options:(.*)--pileup=([^\s]+)(.*)$"""),
                #not shure if event content is required
                (("",  "event_content", ""), r"""^Using user-specified cmsDriver.py options:(.*)--eventcontent ([^\s]+)(.*)$""", "req"),
                #TODO: after changeing the splitter to "taskset -c ..." this is no longer included into the part of correct job
                #(("input_user_root_file", ), r"""^For these tests will use user input file (.+)$"""),
        )

        #parse each of the TimeSize jobs: find candles, etc and start-end times

        reExit_code = re.compile(r"""Individual ([^\s]+) ExitCode (\d+)""")

        if self._DEBUG:
            print("TimeSize (%d) jobs: %s" % (len(jobs), str(jobs)))

        for job_lines in jobs:
            """ we apply the defined parsing rules to extract the required fields of information into the dictionary (as defined in parsing rules) """
            info = self._applyParsingRules(parsing_rules, job_lines)
            #Fixing here the compatibility with new cmsdriver.py --conditions option (for which now we have autoconditions and FrontierConditions_GlobalTag is optional):
            if 'auto:' in info['conditions']:
                from Configuration.AlCa.autoCond import autoCond
                info['conditions'] = autoCond[ info['conditions'].split(':')[1] ].split("::")[0]
            else:
                if 'FrontierConditions_GlobalTag' in info['conditions']:
                    info['conditions']=info['conditions'].split(",")[1]

            #DEBUG:
            #print "CONDITIONS are: %s"%info['conditions']
            #start time - the index after which comes the time stamp
            """ the following is not available on one of the releases, instead
			use the first timestamp available on our job - that's the starting time :) """ 

            #start_time_after = self.findFirstIndex_ofStartsWith(job_lines, "Written out cmsRelvalreport.py input file at:")
            #print start_time_after
            info["start"] = self.firstTimeStampAfter(0, job_lines)

            #TODO: improve in future (in case of some changes) we could use findBefore instead which uses the regexp as parameter for searching 
            #end time - the index before which comes the time stamp

            # On older files we have - "Individual Relvalreport.py ExitCode 0" instead of "Individual cmsRelvalreport.py ExitCode"
            end_time_before = self.findLineAfter(0, job_lines, test_condition = reExit_code.match, return_index = True)

            # on the same line we have the exit Code - so let's get it
            nothing, exit_code = reExit_code.match(job_lines[end_time_before]).groups()

            info["end"] = self.firstTimeStampBefore(end_time_before, job_lines)
            info["exit_code"] = exit_code

            steps_start = self.findFirstIndex_ofStartsWith(job_lines, "You defined your own steps to run:")
            steps_end = self.findFirstIndex_ofStartsWith(job_lines, "*Candle ")
            #probably it includes steps until we found *Candle... ?
            steps = job_lines[steps_start + 1:steps_end]
            if not self.validateSteps(steps):
                self.handleParsingError( "Steps were not found corrently: %s for current job: %s" % (str(steps), str(job_lines)))

                """ quite nasty - just a work around """
                print("Trying to recover from this error in case of old cmssw")

                """ we assume that steps are between the following sentance and a TimeStamp """
                steps_start = self.findFirstIndex_ofStartsWith(job_lines, "Steps passed to writeCommands")
                steps_end = self.findLineAfter(steps_start, job_lines, test_condition = self.isTimeStamp, return_index = True)

                steps = job_lines[steps_start + 1:steps_end]
                if not self.validateSteps(steps):
                    self.handleParsingError( "EVEN AFTER RECOVERY Steps were not found corrently! : %s for current job: %s" % (str(steps), str(job_lines)))
                else:
                    print("RECOVERY SEEMS to be successful: %s" % str(steps))

            info["steps"] = self._LINE_SEPARATOR.join(steps) #!!!! STEPS MIGHT CONTAIN COMMA: ","


            timesize_result.append(info)
        return {"TimeSize": timesize_result}
    #TODO:



    def readCmsScimarkTest(self, testName, testType, core):
        lines  = self.readInput(self._path, fileName = testName + ".log")
        scores = [{"score": self.reCmsScimarkTest.match(line).groups()[1], "type": testType, "core": core}
                        for line in lines 
                        if self.reCmsScimarkTest.match(line)]
        #add the number of messurment
        i = 0
        for score in scores:
            i += 1
            score.update({"messurement_number": i})
        return scores

    def readCmsScimark(self, main_cores = [1]):
        main_core = main_cores[0]
        #TODO: WE DO NOT ALWAYS REALLY KNOW THE MAIN CORE NUMBER! but we don't care too much
        #we parse each of the SciMark files and the Composite scores
        csimark = []
        csimark.extend(self.readCmsScimarkTest(testName = "cmsScimark2", testType = "mainCore", core = main_core))
        csimark.extend(self.readCmsScimarkTest(testName = "cmsScimark2_large", testType = "mainCore_Large", core = main_core))


        #we not always know the number of cores available so we will just search the directory to find out core numbers
        reIsCsiMark_notusedcore = re.compile("^cmsScimark_(\d+).log$")
        scimark_files = [reIsCsiMark_notusedcore.match(f).groups()[0]
                        for f in os.listdir(self._path)
                         if reIsCsiMark_notusedcore.match(f) 
                                and os.path.isfile(os.path.join(self._path, f)) ]

        for core_number in scimark_files:
            try:
                csimark.extend(self.readCmsScimarkTest(testName = "cmsScimark_%s" % str(core_number), testType = "NotUsedCore_%s" %str(core_number), core = core_number))
            except IOError as e:
                if self._DEBUG:
                    print(e)
        return csimark
        #print csimark

    def parseTheCompletion(self):
        """
         checks if the suite has successfully finished  
                and if the tarball was successfully archived and uploaded to the castor """

        parsing_rules = (
                (("finishing_time", "", ""), r"""^Performance Suite finished running at (.+) on (.+) in directory (.+)$"""),
                (("castor_md5",) , r"""^The md5 checksum of the tarball: (.+)$"""),	
                (("successfully_archived_tarball", ), r"""^Successfully archived the tarball (.+) in CASTOR!$"""),
                #TODO: WE MUST HAVE THE CASTOR URL, but for some of files it's not included [probably crashed]
                (("castor_file_url",), r"""^The tarball can be found: (.+)$"""),			
                (("castor_logfile_url",), r"""^The logfile can be found: (.+)$"""),
        )


        """ we apply the defined parsing rules to extract the required fields of information into the dictionary (as defined in parsing rules) """
        info = self._applyParsingRules(parsing_rules, self.lines_other)

        """ did we detect any errors in log files ? """
        info["no_errors_detected"] = [line for line in self.lines_other if line == "There were no errors detected in any of the log files!"] and "1" or "0"
        if not info["successfully_archived_tarball"]:
            info["castor_file_url"] = ""

        if not info["castor_file_url"]:
            #TODO: get the castor file url or abort
            self.handleParsingError( "Castor tarball URL not found. Trying to get from environment")
            lmdb_castor_url_is_valid = lambda url: url.startswith("/castor/")

            url = ""
            try:
                #print "HERE!"
                url=self.get_tarball_fromlog()
                print("Extracted castor tarball full path by re-parsing cmsPerfSuite.log: %s"%url)

            except:
                if "PERFDB_CASTOR_FILE_URL" in os.environ:
                    url = os.environ["PERFDB_CASTOR_FILE_URL"]

                else: #FIXME: add the possibility to get it directly from the cmsPerfSuite.log file (make sure it is dumped there before doing the tarball itself...)
                    print("Failed to get the tarball location from environment variable PERFDB_CASTOR_FILE_URL") 
                    self.handleParsingError( "Castor tarball URL not found. Provide interactively")

            while True:

                if lmdb_castor_url_is_valid(url):
                    info["castor_file_url"] = url
                    break
                print("Please enter a valid CASTOR url: has to start with /castor/ and should point to the tarball")
                if os.isatty(0): url = sys.stdin.readline()
                else: raise IOError("stdin is closed.")


        return info
    def get_tarball_fromlog(self):
        '''Return the tarball castor location by parsing the cmsPerfSuite.log file'''
        print("Getting the url from the cmsPerfSuite.log")
        log=open("cmsPerfSuite.log","r")
        castor_dir="UNKNOWN_CASTOR_DIR"
        tarball="UNKNOWN_TARBALL"
        for line in log.readlines():
            if 'castordir' in line:
                castor_dir=line.split()[1]
            if 'tgz' in line and tarball=="UNKNOWN_TARBALL": #Pick the first line that contains the tar command...
                if 'tar' in line:
                    tarball=os.path.basename(line.split()[2])
        castor_tarball=os.path.join(castor_dir,tarball)
        return castor_tarball

    def parseAll(self):
        result = {"General": {}, "TestResults":{}, "cmsSciMark":{}, 'unrecognized_jobs': []}

        """ all the general info - start, arguments, host etc """
        result["General"].update(self.parseGeneralInfo())

        """ machine info - cpu, memmory """
        result["General"].update(self.getMachineInfo())

        """ we add info about how successfull was the run, when it finished and final castor url to the file! """
        result["General"].update(self.parseTheCompletion())

        print("Parsing TimeSize runs...")
        if len(self.lines_timesize) > 0:
            try:
                result["TestResults"].update(self.parseTimeSize())
            except Exception as e:
                print("BAD BAD BAD UNHANDLED ERROR in parseTimeSize: " + str(e))

        print("Parsing Other(IgProf, Memcheck, ...) runs...")
        try:
            result["TestResults"].update(self.parseAllOtherTests())
        except Exception as e:
            print("BAD BAD BAD UNHANDLED ERROR in parseAllOtherTests: " + str(e))

        #print result["TestResults"]


        main_cores = [result["General"]["run_on_cpus"]]
        num_cores = result["General"].get("num_cores", 0)
        #DEBUG
        #print "Number of cores was: %s"%num_cores
        #TODO: temporarly - search for cores, use regexp
        main_cores = [1]

        # THE MAHCINE SCIMARKS
        result["cmsSciMark"] = self.readCmsScimark(main_cores = main_cores)

        if self.missing_fields:
            self.handleParsingError("========== SOME REQUIRED FIELDS WERE NOT FOUND DURING PARSING ======= "+ str(self.missing_fields))

        return result



if __name__ == "__main__":
    from xml.dom import minidom
    import cmssw_exportdb_xml
    #steps do not get parsed corectly
    #path = "/home/vidma/Desktop/CERN_code/cmssw/data/CMSSW_3_1_0_pre7_--usersteps=RAW2DIGI-RECO_lxbuild107.cern.ch_relval/relval/CMSSW_3_1_0_pre7/work2" 
    #path = "/home/vidma/Desktop/CERN_code/cmssw/data/CMSSW_3_2_0_--usersteps=GEN-SIM,DIGI_lxbuild106.cern.ch_relval/relval/CMSSW_3_2_0/workGENSIMDIGI"
    #includes finishing time, succesfully archived tarball etc
    #path = "/home/vidma/Desktop/CERN_code/cmssw/CVS_PerfSuiteDB/COMP/PerfSuiteDB/export_data_to_xml/example_of_files/PileUp"
    path = os.path.abspath(".") #Better to point to the local dir than to some old Vidmantas' laptop dirs ;)
    #p = parserPerfsuiteMetadata("/home/vidma/Desktop/CERN_code/cmssw/CVS_PerfSuiteDB/COMP/PerfSuiteDB/export_data_to_xml/example_of_files/PerfsuiteRun")
    p = parserPerfsuiteMetadata(path)
    run_info = p.parseAll()

    #print "======= GENERAL ========= "
    #print "\n".join("%s : %s" % (k, v) for k, v in p.parseAll()["General"].items())
    #print "======= Test results ========= "
    #print "\n".join("%s : %s" % (k, v) for k, v in p.parseAll()["TestResults"].items())

    xml_doc = minidom.Document()
    cmssw_exportdb_xml.exportRunInfo(xml_doc, run_info, print_out = True)
    #print "General info:" + str(p.parseGeneralInfo())
    import doctest
    doctest.testmod()

    #print p.readCmsScimark()


