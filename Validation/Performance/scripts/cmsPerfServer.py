#!/usr/bin/env python

import cmsSimPerfPublish as cspp
import cmsPerfSuite as cps
import socket, os, sys, SimpleXMLRPCServer
import optparse as opt

PROG_NAME = os.path.basename(sys.argv[0])

class ClientData(object):

    def __init__(self,host,clientid):
        self.host = host
        self.hasFinished = False
        self.id   = clientid

    def getHostName(self):
        return self.host

    def getID(self):
        return self.id

    def getInfo(self):
        return self.info

    def setInfo(self,info):
        self.info = info

    def isFinished(self):
        return self.hasFinished

    def markFinished(self):
        self.hasFinished = True

def optionparse():
    #global PROG_NAME, _debug, _dryrun, _verbose
    global _CMSPERF_CMDS, _HOSTS, _NEXT_ID

    parser = opt.OptionParser(usage=("""%s [Options]""" % PROG_NAME))

    parser.add_option('-p',
                      '--port',
                      type="int",
                      dest='port',
                      default=-1,
                      help='Run server on a particular port',
                      metavar='<PORT>',
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

    parser.add_option('-m',
                      '--max-clients',
                      type="int",
                      dest='clients',
                      default=-1,
                      help='Maximum number of clients, after all these clients have returned results the program will terminate',
                      metavar='<CLIENTS>',
                      )


    (options, args) = parser.parse_args()

    if not options.cmsperfcmd == "" and not options.cmscmdfile == "":
        parser.error("ERROR: You can not specify a command file and command string")
        sys.exit()


    if options.cmsperfcmd == "":
        _CMSPERF_CMDS = [ "date" ]
    else:
        _CMSPERF_CMDS = [ options.cmsperfcmd ]


    cmdfile = options.cmscmdfile
    if not cmdfile == "":
        cmdfile = os.path.abspath(cmdfile)
        if os.path.isfile(cmdfile):
            try:
                for line in open(cmdfile):
                    line = line.strip()
                    if not line == "":
                        _CMSPERF_CMDS.append(line)
            except OSError, detail:
                print detail
                sys.exit()
            except IOError, detail:
                print detail
                sys.exit()
        else:
            parser.error("ERROR: %s is not a file" % cmdfile)
            sys.exit()

    _HOSTS = {}
    _NEXT_ID = 0
    maxclients = 1

    if not options.clients == -1:
        maxclients = options.clients
        
    port = 0        
    if options.port == -1:
        port = 8000
    else:
        port = options.port

    return (options, port, maxclients)

def numFinishedHosts(dict):
    num = 0
    for key in dict:
        if dict[key].isFinished():
            num += 1
    return num

def presentBenchmarkData(dict):
    for key in dict:
        print dict[key].getInfo()

def runserv(sport,maxclients):
    # Remember that localhost is the loopback network: it does not provide
    # or require any connection to the outside world. As such it is useful
    # for testing purposes. If you want your server to be seen on other
    # machines, you must use your real network address in stead of
    # 'localhost'.
    server = None
    try:
        server = SimpleXMLRPCServer.SimpleXMLRPCServer((socket.gethostname(),sport))
        server.register_function(req_benchmark_run)
        server.register_function(store_benchmarking_data)
    except socket.error, detail:
        print "ERROR: Could not initialise server:", detail
        sys.exit()
        
    # while
    #   1 number of finished requests does not equal max clients
    #   2 and timeout has not been reached
    #   3 and user has not killed program

    print "Running server..."
    while not (numFinishedHosts(_HOSTS) == maxclients):
        try:
            server.handle_request()
        except (KeyboardInterrupt, SystemExit):
            #cleanup
            presentBenchmarkData(_HOSTS)
            server.server_close()            
            raise
        except:
            #cleanup
            presentBenchmarkData(_HOSTS)
            server.server_close()
            raise
    server.server_close()        
    # Everything was fine, present data
    presentBenchmarkData(_HOSTS)


def req_benchmark_run(host):
    global _HOSTS, _CMSPERF_CMDS, _NEXT_ID
    # check host does not exist in host array    
    # store host in host array
    # return cmsPerfSuite.py configuration options
    #      1 client runs perfsuite
    #      2 client should then run store function to send it's benchmarking data to the server
    exists_ix = None
    for key in _HOSTS:
        ahost = _HOSTS[key]
        if host == ahost.getHostName():
            exists_ix = key
            break
        
    if exists_ix == None:
        thisid = _NEXT_ID
        _HOSTS[thisid] = ClientData(host,thisid)
        _NEXT_ID += 1
        return (thisid, _CMSPERF_CMDS)
    else:
        # something is wrong the same client should not have asked to start the performance run twice
        _HOSTS[exists_ix].markFinished()
        return (-1, [])

def store_benchmarking_data(id,info):
    global _HOSTS
    # store information from client
    # mark client as having finished the benchmarking process
    if _HOSTS.has_key(id):
        _HOSTS[id].setInfo(info)
        _HOSTS[id].markFinished()
        return 0
    else:
        # something is wrong client id does not match any initiated host
        return -1

def _main():
    (options, sport, maxclients) = optionparse()
    runserv(sport,maxclients)

if __name__ == "__main__":
    _main()

