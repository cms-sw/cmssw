#!/usr/bin/env python

import cmsSimPerfPublish as cspp
import cmsPerfSuite as cps
import socket, os, sys, SimpleXMLRPCServer
import optparse as opt

PROG_NAME = os.path.basename(sys.argv[0])

def optionparse():
    #global PROG_NAME, _debug, _dryrun, _verbose

    parser = opt.OptionParser(usage=("""%s [Options]"""))

    parser.add_option('-p',
                      '--port',
                      type="int",
                      dest='port',
                      default=-1,
                      help='Run server on a particular port',
                      metavar='<PORT>',
                      )

    parser.add_option('-mc',
                      '--max-clients',
                      type="int",
                      dest='clients',
                      default=-1,
                      help='Maximum number of clients, after all these clients have returned results the program will terminate',
                      metavar='<CLIENTS>',
                      )    

    (options, args) = parser.parse_args()

    maxclient = 1
    port = 0
    if options
    if options.port == -1:
        port = 8000
    else:
        port = options.port

    return (options, port, maxclients)

def runserv(sport):
    # Remember that localhost is the loopback network: it does not provide
    # or require any connection to the outside world. As such it is useful
    # for testing purposes. If you want your server to be seen on other
    # machines, you must use your real network address in stead of
    # 'localhost'.    
    try:
        server = SimpleXMLRPCServer.SimpleXMLRPCServer(('localhost',sport))
        server.register_function(testfn)
        server.register_function(testfn2)        
        # while
        #   1 number of finished requests does not equal max clients
        #   2 and timeout has not been reached
        #   3 and user has not killed program
        server.serve_forever()        
    except socket.error, detail:
        print "ERROR: Could not communicate with server:", detail

def getBenchmarkNumbers():
    cps.main("")
    cspp.main()

def testfn(host):
    # check host does not exist in host array    
    # store host in host array
    # return cmsPerfSuite.py configuration options
    #      1 client runs perfsuite
    #      2 client should then run store function to send it's benchmarking data to the server
    return "test"

def testfn2(host,info):
    # store information from client
    # mark client as finished benchmarking

def _main():
    (options, sport) = optionparse()
    runserv(sport)

if __name__ == "__main__":
    _main()

