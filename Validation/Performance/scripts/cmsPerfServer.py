#!/usr/bin/env python
import cmsSimPerfPublish as cspp
import cmsPerfSuite as cps
import cmsPerfHarvest as cph
from cmsPerfCommons import Candles
import optparse as opt
import socket, os, sys, SimpleXMLRPCServer, threading, exceptions

_outputdir  = os.getcwd()
_reqnumber  = 0
_logreturn  = True
_PROG_NAME  = os.path.basename(sys.argv[0])
_CASTOR_DIR = "/castor/cern.ch/cms/store/relval/performance/"
_DEFAULTS   = {"castordir"        : _CASTOR_DIR,
               "perfsuitedir"     : os.getcwd(),
               "TimeSizeEvents"   : 100        ,
               "IgProfEvents"     : 5          ,
               "ValgrindEvents"   : 1          ,
               "cmsScimark"       : 10         ,
               "cmsScimarkLarge"  : 10         ,
               "cmsdriverOptions" : ""         ,
               "stepOptions"      : ""         ,
               "quicktest"        : False      ,
               "profilers"        : ""         ,
               "cpus"             : [1]        ,
               "cores"            : 4          ,
               "prevrel"          : ""         ,
               "isAllCandles"     : True       ,
               "candles"          : Candles    ,
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
                      default=-1,
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

    outputdir = options.outputdir
    if not outputdir == "":
        outputdir = os.path.abspath(outputdir)
        if not os.path.exists(outputdir):
            parser.error("the specified output directory %s does not exist" % outputdir)
            sys.exit()
        _DEFAULTS["perfsuitedir"] = outputdir
        
    port = 0        
    if options.port == -1:
        port = 8000
    else:
        port = options.port

    _outputdir = outputdir

    return (port,outputdir)

def runserv(sport):
    # Remember that localhost is the loopback network: it does not provide
    # or require any connection to the outside world. As such it is useful
    # for testing purposes. If you want your server to be seen on other
    # machines, you must use your real network address in stead of
    # 'localhost'.
    server = None
    try:
        server = SimpleXMLRPCServer.SimpleXMLRPCServer((socket.gethostname(),sport))
        server.register_function(request_benchmark)
    except socket.error, detail:
        print "ERROR: Could not initialise server:", detail
        sys.exit()

    print "Running server on port %s... " % sport        
    while True:
        try:
            server.handle_request()
        except (KeyboardInterrupt, SystemExit):
            #cleanup
            server.server_close()            
            raise
        except:
            #cleanup
            server.server_close()
            raise
    server.server_close()        

def runcmd(cmd):
    process  = os.popen(cmd)
    cmdout   = process.read()
    exitstat = process.close()

    if True:
        print cmd
        print cmdout

    if not exitstat == None:
        sig     = exitstat >> 16    # Get the top 16 bits
        xstatus = exitstat & 0xffff # Mask out all bits except the bottom 16
        raise
    return cmdout

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
    global _outputdir, _reqnumber
    try:
        # input is a list of dictionaries each defining the
        #   keywords to cmsperfsuite
        outs = []
        i = 0
        exists = True
        while exists:
            topdir = os.path.join(_outputdir,"request_" + str(_reqnumber))
            exists = os.path.exists(topdir)
            _reqnumber += 1
        os.mkdir(topdir)
        for cmd in cmds:
            curperfdir = os.path.join(topdir,str(i))        
            if not os.path.exists(curperfdir):
                os.mkdir(curperfdir)
            logfile = os.path.join(curperfdir, "cmsPerfSuite.log")
            #if cmd.has_key("logfile"):
            #    logfile = os.path.join(getCPSkeyword("logfile"          , cmd), "cmsPerfSuite.log")
            if os.path.exists(logfile):
                logfile = logfile + str(i)
            cps.runPerfSuite(castordir        = getCPSkeyword("castordir"       , cmd),
                             perfsuitedir     = curperfdir                             ,
                             TimeSizeEvents   = getCPSkeyword("TimeSizeEvents"  , cmd),
                             IgProfEvents     = getCPSkeyword("IgProfEvents"    , cmd),
                             ValgrindEvents   = getCPSkeyword("ValgrindEvents"  , cmd),
                             cmsScimark       = getCPSkeyword("cmsScimark"      , cmd),
                             cmsScimarkLarge  = getCPSkeyword("cmsScimarkLarge" , cmd),
                             cmsdriverOptions = getCPSkeyword("cmsdriverOptions", cmd),
                             stepOptions      = getCPSkeyword("stepOptions"     , cmd),
                             quicktest        = getCPSkeyword("quicktest"       , cmd),
                             profilers        = getCPSkeyword("profilers"       , cmd),
                             cpus             = getCPSkeyword("cpus"            , cmd),
                             cores            = getCPSkeyword("cores"           , cmd),
                             prevrel          = getCPSkeyword("prevrel"         , cmd),
                             isAllCandles     = getCPSkeyword("isAllCandles"    , cmd),
                             candles          = getCPSkeyword("candles"         , cmd),
                             bypasshlt        = getCPSkeyword("bypasshlt"       , cmd),
                             runonspare       = getCPSkeyword("runonspare"      , cmd),
                             logfile          = logfile                               )
            if _returnlog:
                outs.append(readlog(logfile))
            else:
                outs.append(cph.harvest(curperfdir))
            i += 1

        return outs
    except exceptions, detail:
        # wrap the entire function in try except so we can log the error at client and server
        print detail
        raise

def _main():
    (sport, outputdir) = optionparse()
    server_thread = threading.Thread(target = runserv(sport))
    server_thread.setDaemon(True) # Allow process to finish if this is the only remaining thread
    server_thread.start()

if __name__ == "__main__":
    _main()

