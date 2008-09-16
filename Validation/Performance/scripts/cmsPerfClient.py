#!/usr/bin/env python
import socket, xml, xmlrpclib, os, sys, threading, Queue, time, random
import optparse as opt

PROG_NAME = os.path.basename(sys.argv[0])

def optionparse():

    parser = opt.OptionParser(usage=("""%s [Options]""" % PROG_NAME))

    parser.add_option('-p',
                      '--port',
                      type="int",
                      dest='port',
                      default=-1,
                      help='Connect to server on a particular port',
                      metavar='<PORT>',
                      )

    parser.add_option('-m',
                      '--machines',
                      type="string",
                      dest='machines',
                      default="",
                      help='A comma separated list of the machines to run the benchmarking on',
                      metavar='<MACHINES>',
                      )

    parser.add_option('-c',
                      '--cps-cmd',
                      type="string",
                      dest='cmsperfcmd',
                      default="",
                      help='The cmsPerfSuite.py command to run',
                      metavar='<COMMAND>',
                      )

    parser.add_option('-f',
                      '--cmd-file',
                      type="string",
                      dest='cmscmdfile',
                      default="",
                      help='A file of cmsPerfSuite.py commands to execute on the machines',
                      metavar='<PATH>',
                      )      

    (options, args) = parser.parse_args()
    

    if not options.cmsperfcmd == "" and not options.cmscmdfile == "":
        parser.error("ERROR: You can not specify a command file and command string")
        sys.exit()

    cmsperf_cmds = []

    if options.cmscmdfile == "":
        if options.cmsperfcmd == "":
            cmsperf_cmds = [ "date" ]
        else:
            cmsperf_cmds = [ options.cmsperfcmd ]


    cmdfile = options.cmscmdfile
    if not cmdfile == "":
        cmdfile = os.path.abspath(cmdfile)
        if os.path.isfile(cmdfile):
            try:
                for line in open(cmdfile):
                    line = line.strip()
                    if not line == "":
                        cmsperf_cmds.append(line)
            except OSError, detail:
                print detail
                sys.exit()
            except IOError, detail:
                print detail
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

    return (cmsperf_cmds, port, machines)

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

def runclient(perfcmds, hosts, port):
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
            presentBenchmarkData(queue)            
            raise
        except:
            #cleanup
            presentBenchmarkData(queue)
            raise
    presentBenchmarkData(queue)    

def _main():
    (cmsperf_cmds, port, hosts) = optionparse()
    runclient(cmsperf_cmds, hosts, port)

def presentBenchmarkData(q):
    while not q.empty():
        item = q.get()
        print item

if __name__ == "__main__":
    _main()
