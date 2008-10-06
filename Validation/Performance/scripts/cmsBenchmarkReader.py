#!/usr/bin/env python
import pickle, os, sys

####################
#
# Parses name of datafile from the arguments
#
def optionParse():
    parser = opt.OptionParser()
    parser.add_option_group(devel)
    (options, args) = parser.parse_args()

    if not len(args) == 1:
        parser.error("you must only pass one file as an argument")
        sys.exit()

    datafile = os.path.abspath(args[0])
    if not os.path.isfile(datafile):
        parser.error("%s file does not exist" % datafile)
        sys.exit()
        
    return datafile

########################################
# (this could be cleaned up a little bit)
# Format of the data file should be:
# 
#  [ host_tuple ]    # list of host data
#
#  host_tuple = ( hostname, [ command_results ]) # hostname and list of results (one for each job specification: each run)
#
#  command_results = {cpu_id: data_output}       # contains a dictionary, one set of data for each cpu
#
# For example:
# returned data     = [ cmd_output1, cmd_output2 ... ]
# cmd_output1       = { cpuid1 : cpu_output1, cpuid2 : cpu_output2 ... }     # cpuid is "None" if there was only one cpu used
# cpu_output1       = { candle1  : profset_output1, candle2 : profset_output2 ... }
# profset_output1   = { profset1 : profile_output1, ... }
# profile_output1   = { profiletype1: step_output1, ... }
# step_output1      = { step1: list_of_cpu_times, ... }
# list_of_cpu_times = [ (evt_num1, secs1), ... ]
#
def printBenchmarkData(afile):
    ph = open(afile,"rb")
    list = pickle.load(ph)
    ph.close()
    for host, data in list:
        try:
            for options, cpudata in data:
                for cpukey in cpudata:
                    cpustr = ""
                    if cpukey == "None":
                        cpustr = "cpu: " + str(options["cpus"][0])
                    else:
                        cpustr = "cpu: " + str(cpukey)
                    candledata = cpudata[cpukey] 
                    for candlekey in candledata:
                        profsetdata = candledata[candlekey] 
                        for profsetkey in profsetdata:
                            profiledata = profsetdata[profsetkey]
                            for profilekey in profiledata:
                                stepdata = profiledata[profilekey] 
                                for stepkey in stepdata:
                                    print "Host: " + str(host)
                                    print "Options: " + str(options)
                                    print cpustr
                                    print "candle: " + str(candlekey)
                                    print "profsetdata: " + str(profsetkey)                                
                                    print "profiler: " + str(profilekey)                                
                                    print "step: " + str(stepkey)
                                    for evtdat in stepdata[stepkey]:
                                        print evtdat
        except TypeError, detail:
            print "Data: does not exist for this machine, the server failed to return a valid data structure"
            print detail


if __name__ == "__main__":
    datafile = optionParse()
    printBenchmarkData(datafile)
