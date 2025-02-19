#!/usr/bin/env python
import cmsPerfPublish as cspp
import cmsPerfSuite      as cps
import cmsPerfHarvest    as cph
#G.Benelli
import cmsRelValCmd #Module that contains get_cmsDriverOptions() function to get a string with the options we are interested in from cmsDriver_highstats_hlt.txt
import cmsCpuInfo #Module that contains get_NumOfCores() function to get an integer with the number of cores on the current machine (through parsing /proc/cpuinfo)
from cmsPerfCommons import Candles
import optparse as opt
import socket, os, sys, SimpleXMLRPCServer, threading, exceptions

CandlesString=""
for candle in Candles:
    CandlesString=CandlesString+","+candle
print CandlesString[1:]
_outputdir  = os.getcwd()
_reqnumber  = 0
_logreturn  = False
_PROG_NAME  = os.path.basename(sys.argv[0])
_CASTOR_DIR = "/castor/cern.ch/cms/store/relval/performance/"
_DEFAULTS   = {"castordir"        : _CASTOR_DIR,
               "perfsuitedir"     : os.getcwd(),
               "TimeSizeEvents"   : 100        ,
               "TimeSizeCandles"      : "",
               "TimeSizePUCandles"      : "",
               "IgProfEvents"     : 0          ,
               "IgProfCandles"        : ""       ,
               "IgProfPUCandles"        : ""       ,
               "CallgrindEvents"  : 0          ,
               "CallgrindCandles"     : ""       ,
               "CallgrindPUCandles"     : ""       ,
               "MemcheckEvents"   : 0          ,
               "MemcheckCandles"      : ""          ,
               "MemcheckPUCandles"      : ""          ,
               "cmsScimark"       : 10         ,
               "cmsScimarkLarge"  : 10         ,
               "cmsdriverOptions" : cmsRelValCmd.get_cmsDriverOptions(), #Get these options automatically now!
               "stepOptions"      : ""         ,
               "quicktest"        : False      ,
               "profilers"        : ""         ,
               "cpus"             : "1"        ,
               "cores"            : cmsCpuInfo.get_NumOfCores(), #Get this option automatically
               "prevrel"          : ""         ,
               "isAllCandles"     : True       ,
               #"candles"          : CandlesString[1:]    ,
               "bypasshlt"        : False      ,
               "runonspare"       : True       ,
               "logfile"          : os.path.join(os.getcwd(),"cmsPerfSuite.log")}

def optionparse():
    global _outputdir
    parser = opt.OptionParser(usage=("""%s [Options]""" % _PROG_NAME))

    parser.add_option('-p',
                      '--port',
                      type="int",
                      dest='port',
                      default=8000, #Setting the default port to be 8000
                      help='Run server on a particular port',
                      metavar='<PORT>',
                      )
    
    parser.add_option('-o',
                      '--output',
                      type="string",
                      dest='outputdir',
                      default="",
                      help='The output directory for all the cmsPerfSuite runs',
                      metavar='<DIR>',
                      )

    (options, args) = parser.parse_args()

    if not options.outputdir == "":
        options.outputdir = os.path.abspath(options.outputdir)
        if not os.path.exists(options.outputdir):
            parser.error("the specified output directory %s does not exist" % options.outputdir)
            sys.exit()
        #This seems misleading naming _DEFAULTS, while we are re-initializing its keys to different values as we go...
        _DEFAULTS["perfsuitedir"] = options.outputdir

    #resetting global variable _outputdir too... do we really need this variable?
    _outputdir = options.outputdir

    return (options.port,options.outputdir)

#class ClientThread(threading.Thread):
    # Overloading the constructor to accept cmsPerfSuite parameters
    

def runserv(port):
    # Remember that localhost is the loopback network: it does not provide
    # or require any connection to the outside world. As such it is useful
    # for testing purposes. If you want your server to be seen on other
    # machines, you must use your real network address in stead of
    # 'localhost'.
    server = None
    try:
        server = SimpleXMLRPCServer.SimpleXMLRPCServer((socket.gethostname(),port))
        server.register_function(request_benchmark)
    except socket.error, detail:
        print "ERROR: Could not initialise server:", detail
        sys.stdout.flush()        
        sys.exit()

    print "Running server on port %s... " % port
    sys.stdout.flush()    
    while True:
        try:
            server.handle_request()
            sys.stdout.flush()
        except (KeyboardInterrupt, SystemExit):
            #cleanup
            server.server_close()            
            raise
        except:
            #cleanup
            server.server_close()
            raise
    server.server_close()        

#Not sure about this unused function:
#Probably left over from first server implementation tests
#def runcmd(cmd):
#    process  = os.popen(cmd)
#    cmdout   = process.read()
#    exitstat = process.close()
#
#    if True:
#        print cmd
#        print cmdout
#
#    if not exitstat == None:
#        sig     = exitstat >> 16    # Get the top 16 bits
#        xstatus = exitstat & 0xffff # Mask out all bits except the bottom 16
#        raise
#    return cmdout

def readlog(logfile):
    astr = ""    
    try:
        for line in open(logfile,"r"):
            astr += line
    except (OSError, IOError) , detail:
        print detail
    return astr

def getCPSkeyword(key,dict):
    if dict.has_key(key):
        return dict[key]
    else:
        return _DEFAULTS[key]


def request_benchmark(cmds):
    #This is the function with which the server listens on the given port
    #cmds is a list of dictionaries: each dictionary is a set of cmsPerfSuite commands to run.
    #Most common use will be only 1 dictionary, but for testing with reproducibility and statistical errors
    #one can easily think of sending the same command 10 times for example and then compare the outputs
    global _outputdir, _reqnumber
    print "Commands received running perfsuite for these jobs:"
    print cmds
    sys.stdout.flush()
    try:
        # input is a list of dictionaries each defining the
        #   keywords to cmsperfsuite
        outs = []
        cmd_num = 0
        exists = True
        #Funky way to make sure we create a directory request_n with n = serial request number (if the server is running for a while
        #and the client submits more than one request
        #This should never happen since _reqnumber is a global variable on the server side...
        while exists:
            topdir = os.path.join(_outputdir,"request_" + str(_reqnumber))
            exists = os.path.exists(topdir)
            _reqnumber += 1
        os.mkdir(topdir)
        #Going through each command dictionary in the cmds list (usually only 1 such dictionary):
        for cmd in cmds:
            curperfdir = os.path.abspath(os.path.join(topdir,str(cmd_num)))
            if not os.path.exists(curperfdir):
                os.mkdir(curperfdir)
            logfile = os.path.join(curperfdir, "cmsPerfSuite.log")
            if os.path.exists(logfile):
                logfile = logfile + str(cmd_num)
            print cmd
            if cmd.has_key('cpus'):
                if cmd['cpus'] == "All":
                    print "Running performance suite on all CPUS!\n"
                    cmd['cpus']=""
                    for cpu in range(cmsCpuInfo.get_NumOfCores()):
                        cmd["cpus"]=cmd["cpus"]+str(cpu)+","
                    cmd["cpus"]=cmd["cpus"][:-1] #eliminate the last comma for cleanliness 
                    print "I.e. on cpus %s\n"%cmd["cpus"]
                
            #Not sure this is the most elegant solution... we keep cloning dictionaries...
            cmdwdefs = {}
            cmdwdefs["castordir"       ] = getCPSkeyword("castordir"       , cmd)
            cmdwdefs["perfsuitedir"    ] = curperfdir                      
            cmdwdefs["TimeSizeEvents"  ] = getCPSkeyword("TimeSizeEvents"  , cmd)
            cmdwdefs["TimeSizeCandles" ] = getCPSkeyword("TimeSizeCandles"  , cmd)
            cmdwdefs["TimeSizePUCandles" ] = getCPSkeyword("TimeSizePUCandles"  , cmd)
            cmdwdefs["IgProfEvents"    ] = getCPSkeyword("IgProfEvents"    , cmd)
            cmdwdefs["IgProfCandles"   ] = getCPSkeyword("IgProfCandles"    , cmd)
            cmdwdefs["IgProfPUCandles"   ] = getCPSkeyword("IgProfPUCandles"    , cmd)
            cmdwdefs["CallgrindEvents" ] = getCPSkeyword("CallgrindEvents"  , cmd)
            cmdwdefs["CallgrindCandles"] = getCPSkeyword("CallgrindCandles"  , cmd)
            cmdwdefs["CallgrindPUCandles"] = getCPSkeyword("CallgrindPUCandles"  , cmd)
            cmdwdefs["MemcheckEvents"  ] = getCPSkeyword("MemcheckEvents"  , cmd)
            cmdwdefs["MemcheckCandles" ] = getCPSkeyword("MemcheckCandles"  , cmd)
            cmdwdefs["MemcheckPUCandles" ] = getCPSkeyword("MemcheckPUCandles"  , cmd)
            cmdwdefs["cmsScimark"      ] = getCPSkeyword("cmsScimark"      , cmd)
            cmdwdefs["cmsScimarkLarge" ] = getCPSkeyword("cmsScimarkLarge" , cmd)
            cmdwdefs["cmsdriverOptions"] = getCPSkeyword("cmsdriverOptions", cmd)
            cmdwdefs["stepOptions"     ] = getCPSkeyword("stepOptions"     , cmd)
            cmdwdefs["quicktest"       ] = getCPSkeyword("quicktest"       , cmd)
            cmdwdefs["profilers"       ] = getCPSkeyword("profilers"       , cmd)
            cmdwdefs["cpus"            ] = getCPSkeyword("cpus"            , cmd)
            cmdwdefs["cores"           ] = getCPSkeyword("cores"           , cmd)
            cmdwdefs["prevrel"         ] = getCPSkeyword("prevrel"         , cmd)
#            cmdwdefs["candles"         ] = getCPSkeyword("candles"         , cmd)                                    
#            cmdwdefs["isAllCandles"    ] = len(Candles) == len(cmdwdefs["candles"]) #Dangerous: in the _DEFAULTS version this is a boolean!
            cmdwdefs["bypasshlt"       ] = getCPSkeyword("bypasshlt"       , cmd)
            cmdwdefs["runonspare"      ] = getCPSkeyword("runonspare"      , cmd)
            cmdwdefs["logfile"         ] = logfile
            logh = open(logfile,"w")
            logh.write("This perfsuite run was configured with the following options:\n")
            #logh.write(str(cmdwdefs) + "\n")
            for key in cmdwdefs.keys():
                logh.write(key + "\t" +str(cmdwdefs[key])+"\n")
            logh.close()
            print "Calling cmsPerfSuite.main() function\n"
            cpsInputArgs=[
                      #"-a",cmdwdefs["castordir"],
                      "-t",cmdwdefs["TimeSizeEvents"  ],
                      "--RunTimeSize",cmdwdefs["TimeSizeCandles"],
                      "-o",cmdwdefs["perfsuitedir"    ],
                      #"-i",cmdwdefs["IgProfEvents"    ],
                      #"--RunIgProf",cmdwdefs["RunIgProf"    ],
                      #"-c",cmdwdefs["CallgrindEvents"  ],
                      #"--RunCallgrind",cmdwdefs["RunCallgrind"  ],
                      #"-m",cmdwdefs["MemcheckEvents"],
                      #"--RunMemcheck",cmdwdefs["RunMemcheck"],
                      "--cmsScimark",cmdwdefs["cmsScimark"      ],
                      "--cmsScimarkLarge",cmdwdefs["cmsScimarkLarge" ],
                      "--cmsdriver",cmdwdefs["cmsdriverOptions"],
                      "--step",cmdwdefs["stepOptions"     ],
                      #"--quicktest",cmdwdefs["quicktest"       ],
                      #"--profile",cmdwdefs["profilers"       ],
                      "--cpu",cmdwdefs["cpus"            ],
                      "--cores",cmdwdefs["cores"           ],
                      #"--prevrel",cmdwdefs["prevrel"         ],
 #                     "--candle",cmdwdefs["candles"         ],
                      #"--bypass-hlt",cmdwdefs["bypasshlt"       ],
                      "--notrunspare"#,cmdwdefs["runonspare"      ]#,
                      #"--logfile",cmdwdefs["logfile"         ]
                      ]
            print cpsInputArgs
            cps.main(cpsInputArgs)
            print "Running of the Performance Suite is done!"          
            #logreturn is false... so this does not get executed
            #Maybe we can replace this so that we can have more verbose logging of the server activity
            if _logreturn:
                outs.append(readlog(logfile))
            else:
                outs.append((cmdwdefs,cph.harvest(curperfdir)))
            #incrementing the variable for the command number:
            cmd_num += 1

            
        return outs #Not sure what James intended to return here... the contents of all logfiles in a list of logfiles?
    except exceptions.Exception, detail:
        # wrap the entire function in try except so we can log the error at client and server
        logh = open(os.path.join(os.getcwd(),"error.log"),"a")
        logh.write(str(detail) + "\n")
        logh.flush()
        logh.close()
        print detail
        sys.stdout.flush()
        raise

def _main():
    print _DEFAULTS
    (port, outputdir) = optionparse()
    server_thread = threading.Thread(target = runserv(port))
    server_thread.setDaemon(True) # Allow process to finish if this is the only remaining thread
    server_thread.start()

if __name__ == "__main__":
    _main()

