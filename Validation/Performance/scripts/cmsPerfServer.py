#!/usr/bin/env python

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
                      #metavar='<STEPS>',
                      )

    (options, args) = parser.parse_args()

    port = 0
    if options.port == -1:
        port = 8000
    else:
        port = options.port

    return (options, port)

def runserv(sport):
# Remember that localhost is the loopback network: it does not provide
# or require any connection to the outside world. As such it is useful
# for testing purposes. If you want your server to be seen on other
# machines, you must use your real network address in stead of
# 'localhost'.    
    try:
        server = SimpleXMLRPCServer.SimpleXMLRPCServer(('localhost',sport))
        server.register_function(somefn)
        server.serve_forever()        
    except socket.error, detail:
        print "ERROR: Could not communicate with server:", detail

def somefn():
    return "bob"

def _main():
    (options, sport) = optionparse()
    runserv(sport)

if __name__ == "__main__":
    _main()

