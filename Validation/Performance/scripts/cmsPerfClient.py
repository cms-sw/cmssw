#!/usr/bin/env python
import socket, xml, xmlrpclib, os, sys, threading, Queue, time, random, pickle, exceptions
import optparse as opt
#Documentation needs to follow... but for now just know that
#a template file for cmsPerfClient.py -f option is BencmarkCfg.py in Validation/Performance/python dir.
PROG_NAME = os.path.basename(sys.argv[0])
# list of valid options for the configuration file
validPerfSuitKeys= ["castordir", "perfsuitedir" ,"TimeSizeEvents", "TimeSizeCandles","IgProfEvents", "IgProfCandles", "CallgrindEvents", "CallgrindCandles", "MemcheckEvents","MemcheckCandles","cmsScimark", "cmsScimarkLarge",
                    "cmsdriverOptions", "stepOptions", "quicktest", "profilers", "cpus", "cores", "prevrel", "isAllCandles", "candles",
                    "bypasshlt", "runonspare", "logfile"]
#################
#
# Option parser
# returns : Command set to run on each (or all) machines, port to connect to server, 
#           List of machines to connect to, File to pickle results to,
#           Dictionary to index which command set to use for which machine
def optionparse():

    #########################
    # Config file type validator
    # Checks type of configuration options in the config file 
    #
    def _isValidPerfCmdsDef(alist):
        out = True
        for item in alist:
            isdict = type(item) == type({})
            out = out and isdict
            if isdict:
                for key in item:
                    out = out and key in validPerfSuitKeys
                    if   key == "cpus":
                        out = out and type(item[key]) == type("") #has to be a string not a list!
                    elif key == "cores":
                        out = out and type(item[key]) == type("")
                    elif key == "castordir":
                        out = out and type(item[key]) == type("")
                    elif key == "perfsuitedir":
                        out = out and type(item[key]) == type("")
                    elif key == "TimeSizeEvents":
                        out = out and type(item[key]) == type(123)
                    elif key == "TimeSizeCandles":
                        out = out and type(item[key]) == type("")
                    elif key == "CallgrindEvents":
                        out = out and type(item[key]) == type(123)
                    elif key == "CallgrindCandles":
                        out = out and type(item[key]) == type("")
                    elif key == "IgProfEvents":
                        out = out and type(item[key]) == type(123)
                    elif key == "IgProfCandles":
                        out = out and type(item[key]) == type("")
                    elif key == "MemcheckEvents":
                        out = out and type(item[key]) == type(123)
                    elif key == "MemcheckCandles":
                        out = out and type(item[key]) == type("")
                    elif key == "cmsScimark":
                        out = out and type(item[key]) == type(123)
                    elif key == "cmsScimarkLarge":
                        out = out and type(item[key]) == type(123)
                    elif key == "cmsdriverOptions":
                        out = out and type(item[key]) == type("")
                    elif key == "stepOptions":
                        out = out and type(item[key]) == type("")
                    elif key == "quicktest":
                        out = out and type(item[key]) == type(False)
                    elif key == "profilers":
                        out = out and type(item[key]) == type("")
                    elif key == "prevrel":
                        out = out and type(item[key]) == type("")
                    elif key == "isAllCandles":
                        out = out and type(item[key]) == type(False)
                    elif key == "candles":
                        out = out and type(item[key]) == type("")#has to be a string not a list!
                    elif key == "bypasshlt":
                        out = out and type(item[key]) == type(False)
                    elif key == "runonspare":
                        out = out and type(item[key]) == type(False)
                    elif key == "logfile":
                        out = out and type(item[key]) == type("")
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
                      action="append",
                      dest='machines',
                      default=[],
                      help='Machines to run the benchmarking on, for each machine add another one of these options',
                      metavar='<MACHINES>',
                      )

    parser.add_option('-f',
                      '--cmd-file',
                      type="string",
                      dest='cmscmdfile',
                      action="append",
                      default=[],
                      help='A files of cmsPerfSuite.py commands to execute on the machines, if more than one of these options is passed and the number of these options is the same as the number of machines, the x-th machine will use the x-th config file.',
                      metavar='<PATH>',
                      )      

    (options, args) = parser.parse_args()

    ######################
    # Check output file location
    #
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


    ###############
    # Check configuration files for errors
    #
    cmsperf_cmds = []
    cmscmdfiles = options.cmscmdfile
    if len(cmscmdfiles) <= 0:
        parser.error("A valid python file defining a list of dictionaries that represents a list of cmsPerfSuite keyword arguments must be passed to this program")
        sys.exit()
    else:
        for cmscmdfile in cmscmdfiles:
            cmdfile = os.path.abspath(cmscmdfile)
            print cmdfile
            if os.path.isfile(cmdfile):
                try:
                    execfile(cmdfile)
                    cmsperf_cmds.append(listperfsuitekeywords)
                except (SyntaxError), detail:
                    parser.error("ERROR: %s must be a valid python file" % cmdfile)
                    sys.exit()
                except (NameError), detail:
                    parser.error("ERROR: %s must contain a list (variable named listperfsuitekeywords) of dictionaries that represents a list of cmsPerfSuite keyword arguments must be passed to this program: %s" % (cmdfile,str(detail)))
                    sys.exit()
                except :
                    raise
                if not type(cmsperf_cmds[-1]) == type([]):
                    parser.error("ERROR: %s must contain a list (variable named listperfsuitekeywords) of dictionaries that represents a list of cmsPerfSuite keyword arguments must be passed to this program 2" % cmdfile)
                    sys.exit()
                if not _isValidPerfCmdsDef(cmsperf_cmds[-1]):
                    parser.error("ERROR: %s must contain a list (variable named listperfsuitekeywords) of dictionaries that represents a list of cmsPerfSuite keyword arguments must be passed to this program 3" % cmdfile)
                    sys.exit()                

            else:
                parser.error("ERROR: %s is not a file" % cmdfile)
                sys.exit()

    ########
    # Setup port number
    #
    port = 0        
    if options.port == -1:
        port = 8000
    else:
        port = options.port

    machines = options.machines

    #################
    # Check machine hostnames
    #
    if len(machines) <= 0:
        parser.error("you must specify at least one machine to benchmark")        
    else:
        machines = map(lambda x: x.strip(),machines)

    for machine in machines:
        try:
            output = socket.getaddrinfo(machine,port)
        except socket.gaierror:
            parser.error("ERROR: Can not resolve machine address %s (must be ip{4,6} or hostname)" % machine)
            sys.exit()

    ##############
    # Define which configuration file to use for which machine
    # If only one configuration file is used then it used for all machines
    cmdindex = {} # define an index that defines the commands to be run for each machine to be perfsuite'd
    if len(cmsperf_cmds) == 1:
        for machine in machines:
            # each value is the index in cmsperf_cmds that the machine will run
            # in this case all machines run the same set of commands
            cmdindex[machine] = 0 
    else:
        if not len(cmsperf_cmds) == len(machines):
            parser.error("if more than one configuration file was specified you must specify a configuration file for each machine.")
            sys.exit()
            
        for i in range(len(machines)):
            # each value is the index in cmsperf_cmds that the machine will run
            # in this case each machine runs the i-th configuration file passed as an option
            cmdindex[machine] = i         

    return (cmsperf_cmds, port, machines, outfile, cmdindex)

#################
# Request benchmark
# Connects to server and returns data
# returns: profiling data from server
#
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
    except exceptions, detail:
        print "ERROR: There was a runtime error thrown by server %s; detail follows." % shost
        print detail

#################
# Worker
# This is a subclass of thread that submits commands to the server and stores the result in a thread-safe queue
# 
class Worker(threading.Thread):

    def __init__(self, host, port, perfcmds, queue):
        self.__perfcmds = perfcmds
        self.__host  = host
        self.__port  = port
        self.__queue = queue
        threading.Thread.__init__(self)

    def run(self):
        try:
            data = request_benchmark(self.__perfcmds, self.__host, self.__port)
            #Debugging
            print "data is %s"%data
            print "Puttin it in the queue as (%s,%s)"%(self.__host,data)
            self.__queue.put((self.__host, data))
        except (exceptions.Exception, xmlrpclib.Fault), detail:
            print "Exception was thrown when receiving/submitting job information to host", self.__host, ". Exception information:"
            print detail
            sys.stdout.flush()

##########################
# runclient
# Creates a thread for each machine to profile and waits for all machines to return data (you might consider adding a timeout in the while loop)
# If the client is killed for some reason or there is an exception, dump the data to a file before throwing the exception
def runclient(perfcmds, hosts, port, outfile, cmdindex):
    queue = Queue.Queue()
    # start all threads
    workers = []
    for host in hosts:
        print "Submitting jobs to %s..." % host
        w = Worker(host, port, perfcmds[cmdindex[host]], queue)
        w.start()                
        workers.append(w)
    print "All jobs submitted, waiting for results..."
    sys.stdout.flush()
    # run until all servers have returned data
    while reduce(lambda x,y: x or y, map(lambda x: x.isAlive(),workers)):
        try:            
            time.sleep(2.0)
            sys.stdout.flush()
        except (KeyboardInterrupt, SystemExit):
            #cleanup
            presentBenchmarkData(queue,outfile)            
            raise
        except:
            #cleanup
            presentBenchmarkData(queue,outfile)
            raise
    print "All job results received"
    print "The size with the queue containing all data is: %s "%queue.qsize()
    presentBenchmarkData(queue,outfile)    

########################################
#
# Format of the returned data from remote host should be of the form (this could be cleaned up a little bit)
# 
# list of command outputs [ dictionary of cpus {   }  ]
#
# For example:
# returned data     = [ cmd_output1, cmd_output2 ... ]
# cmd_output1       = { cpuid1 : cpu_output1, cpuid2 : cpu_output2 ... }     # cpuid is "None" if there was only one cpu used
# cpu_output1       = { candle1  : profset_output1, candle2 : profset_output2 ... }
# profset_output1   = { profset1 : profile_output1, ... }
# profile_output1   = { profiletype1: step_output1, ... }
# step_output1      = { step1: list_of_cpu_times, ... }
# list_of_cpu_times = [ (evt_num1, secs1), ... ]

###########
#
# We now massage the data
#
def presentBenchmarkData(q,outfile):
    print "Pickling data to file %s"%outfile
    out = []            # match up the commands with each
                        # command that was passed in the config file
    while not q.empty():
        print "Queue size is still %s"%q.qsize()
        (host, data) = q.get()
        out.append((host,data))
    print "Dumping at screen the output!\n%s"%out
    oh = open(outfile,"wb")
    pickle.dump(out,oh)
    oh.close() 

def _main():
    (cmsperf_cmds, port, hosts, outfile, cmdindex) = optionparse()
    runclient(cmsperf_cmds, hosts, port, outfile, cmdindex)

if __name__ == "__main__":
    _main()
