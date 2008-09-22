#!/usr/bin/env python
import socket, xml, xmlrpclib, os, sys, threading, Queue, time, random, pickle
import optparse as opt

PROG_NAME = os.path.basename(sys.argv[0])
validPerfSuitKeys= ["castordir", "perfsuitedir" ,"TimeSizeEvents", "IgProfEvents", "ValgrindEvents", "cmsScimark", "cmsScimarkLarge",
                    "cmsdriverOptions", "stepOptions", "quicktest", "profilers", "cpus", "cores", "prevrel", "isAllCandles", "candles",
                    "bypasshlt", "runonspare", "logfile"]

def optionparse():
    def _isValidPerfCmdsDef(alist):
        out = True
        for item in alist:
            isdict = type(item) == type({})
            out = out and isdict
            if isdict:
                for key in item:
                    out = out and key in validPerfSuitKeys
        return out

    parser = opt.OptionParser(usage=("""%s [Options]""" % PROG_NAME))

    parser.add_option('-p',
                      '--port',
                      type="int",
                      dest='port',
                      default=-1,
                      help='Connect to server on a particular port',
                      metavar='<PORT>',
                      )

    parser.add_option('-o',
                      '--output',
                      type="string",
                      dest='outfile',
                      default="",
                      help='File to output data to',
                      metavar='<FILE>',
                      )

    parser.add_option('-m',
                      '--machines',
                      type="string",
                      dest='machines',
                      default="",
                      help='A comma separated list of the machines to run the benchmarking on',
                      metavar='<MACHINES>',
                      )

##     parser.add_option('-c',
##                       '--cps-cmd',
##                       type="string",
##                       dest='cmsperfcmd',
##                       default="",
##                       help='The cmsPerfSuite.py command to run',
##                       metavar='<COMMAND>',
##                       )

    parser.add_option('-f',
                      '--cmd-file',
                      type="string",
                      dest='cmscmdfile',
                      default="",
                      help='A file of cmsPerfSuite.py commands to execute on the machines',
                      metavar='<PATH>',
                      )      

    (options, args) = parser.parse_args()

    outfile = options.outfile
    if not outfile == "": 
        outfile = os.path.abspath(options.outfile)
        outdir = os.path.dirname(outfile)
        if not os.path.isdir(outdir):
            parser.error("ERROR: %s is not a valid directory to create %s" % (outdir,os.path.basename(outfile)))
            sys.exit()
    else:
        outfile = os.path.join(os.getcwd(),"cmsmultiperfdata.pypickle")
        
    if os.path.exists(outfile):
        parser.error("ERROR: outfile %s already exists" % outfile)
        sys.exit()

    #if not options.cmscmdfile == "":
    #    parser.error("ERROR: You can not specify a command file and command string")
    #    sys.exit()

    cmsperf_cmds = []

    cmscmdfile = options.cmscmdfile
    if cmscmdfile == "":
        parser.error("A valid python file defining a list of dictionaries that represents a list of cmsPerfSuite keyword arguments must be passed to this program")
        sys.exit()
    else:
        
        cmdfile = os.path.abspath(cmscmdfile)
        print cmdfile
        if os.path.isfile(cmdfile):
            try:
                execfile(cmdfile)
                #cmsperf_cmds = listperfsuitekeywords
                cmsperf_cmds = listperfsuitekeywords
            except (SyntaxError), detail:
                parser.error("ERROR: %s must be a valid python file" % cmdfile)
                sys.exit()
            except (NameError), detail:
                parser.error("ERROR: %s must contain a list (variable named listperfsuitekeywords) of dictionaries that represents a list of cmsPerfSuite keyword arguments must be passed to this program: %s" % (cmdfile,str(detail)))
                sys.exit()
            except :
                raise
            if not type(cmsperf_cmds) == type([]):
                parser.error("ERROR: %s must contain a list (variable named listperfsuitekeywords) of dictionaries that represents a list of cmsPerfSuite keyword arguments must be passed to this program 2" % cmdfile)
                sys.exit()
            if not _isValidPerfCmdsDef(cmsperf_cmds):
                parser.error("ERROR: %s must contain a list (variable named listperfsuitekeywords) of dictionaries that represents a list of cmsPerfSuite keyword arguments must be passed to this program 3" % cmdfile)
                sys.exit()                
                
        else:
            parser.error("ERROR: %s is not a file" % cmdfile)
            sys.exit()

    port = 0        
    if options.port == -1:
        port = 8000
    else:
        port = options.port

    machines = []
    
    if "," in options.machines:
        machines = options.machines.split(",")
        machines = map(lambda x: x.strip(),machines)
    else:
        machines = [ options.machines.strip() ]

    if len(machines) <= 0:
        parser.error("you must specify at least one machine to benchmark")

    for machine in machines:
        try:
            output = socket.getaddrinfo(machine,port)
        except socket.gaierror:
            parser.error("ERROR: Can not resolve machine address %s (must be ip{4,6} or hostname)" % machine)

    return (cmsperf_cmds, port, machines, outfile)

def request_benchmark(perfcmds,shost,sport):
    try:
        server = xmlrpclib.ServerProxy("http://%s:%s" % (shost,sport))    
        return server.request_benchmark(perfcmds)
    except socket.error, detail:
        print "ERROR: Could not communicate with server %s:%s:" % (shost,sport), detail
    except xml.parsers.expat.ExpatError, detail:
        print "ERROR: XML-RPC could not be parsed:", detail
    except xmlrpclib.ProtocolError, detail:
        print "ERROR: XML-RPC protocol error", detail, "try using -L xxx:localhost:xxx if using ssh to forward"

class Worker(threading.Thread):

    def __init__(self, host, port, perfcmds, queue):
        self.__perfcmds = perfcmds
        self.__host  = host
        self.__port  = port
        self.__queue = queue
        threading.Thread.__init__(self)

    def run(self):
        data = request_benchmark(self.__perfcmds, self.__host, self.__port)
        self.__queue.put((self.__host, data))

def runclient(perfcmds, hosts, port, outfile):
    queue = Queue.Queue()
    # start all threads
    workers = []
    for host in hosts:
        w = Worker(host, port, perfcmds, queue)
        w.start()                
        workers.append(w)
        
    # run until all servers have returned data
    while reduce(lambda x,y: x or y, map(lambda x: x.isAlive(),workers)):
        try:            
            time.sleep(2.0)
        except (KeyboardInterrupt, SystemExit):
            #cleanup
            presentBenchmarkData(perfcmds,queue,outfile)            
            raise
        except:
            #cleanup
            presentBenchmarkData(perfcmds,queue,outfile)
            raise
    presentBenchmarkData(perfcmds,queue,outfile)    

def _main():
    (cmsperf_cmds, port, hosts, outfile) = optionparse()
    runclient(cmsperf_cmds, hosts, port, outfile)


########################################
#
# Format of the returned data from remote host should be of the form
# 
# list of command outputs [ dictionary of cpus {   }  ]
# For example:
# returned data     = [ cmd_output1, cmd_output2 ... ]
# cmd_output1       = { cpuid1 : cpu_output1, cpuid2 : cpu_output2 ... }
# cpu_output1       = { (candle1, profset1) : profset_output1, (candle2,profset2) : profset_output2 ... }
# profset_output1   = { (profiletype1, step1) : list_of_cputimes1, ... }
# list_of_cpu_times = [ (evt_num1, secs1), ... ]

###########
#
# We now massage the data so that each command output from each server has the command keywords tuple'd with it
#
def presentBenchmarkData(perfcmds,q,outfile):
    print "Pickling data to file..."
    out = []            # match up the commands with each
                        # command that was passed in the config file
    while not q.empty():
        (host, data) = q.get()
        newdata = []
        i = 0
        for dat in data:
            keywdict = {}
            if i < len(perfcmds): 
                keywdict = perfcmds[i]
            else:
                keywdict = {None: "Could not match up commands passed in via config file with cmds returned"}  
            if len(keywdict) == 0:
                keywdict = {None: "Defaults in cmsPerfServer were used"}
            
            newdata.append((keywdict,dat))
            i += 1
        out.append((host,newdata))
    oh = open(outfile,"wb")
    pickle.dump(out,oh)
    oh.close()

if __name__ == "__main__":
    _main()
