#!/usr/bin/env python
import pickle, os, sys

afile = os.path.abspath(sys.argv[1])

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
