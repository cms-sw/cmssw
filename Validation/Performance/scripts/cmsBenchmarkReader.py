#!/usr/bin/env python
import pickle, os, sys

afile = os.path.abspath(sys.argv[1])

ph = open(afile,"rb")
list = pickle.load(ph)
ph.close()
for host, data in list:
    print "Host:", host
    try:
        for options, cpudata in data:
            print "Options:", options
            for cpukey in cpudata:
                if cpukey == "None":
                    print "cpu:", options["cpus"][0]
                else:
                    print "cpu:", cpukey
                candledata = cpudata[cpukey] 
                for candlekey in candledata:
                    print "candle:", candlekey
                    profsetdata = candledata[candlekey] 
                    for profsetkey in profsetdata:
                        print "profsetdata:", profsetkey
                        profiledata = profsetdata[profsetkey]
                        for profilekey in profiledata:
                            print "profiler:", profilekey
                            stepdata = profiledata[profilekey] 
                            for stepkey in stepdata:
                                print "step:", stepkey
                                for evtdat in stepdata[stepkey]:
                                    print evtdat
    except TypeError:
        print "Data: does not exist for this machine, the server failed to return a valid data structure"
