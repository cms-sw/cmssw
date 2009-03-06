#!/usr/bin/env python
import socket, xml, xmlrpclib, os, sys
import optparse as opt

#
# Forward ports (Crtl-a d : to detach screen session)
# screen ssh -v -L 8000:localhost:8000 nicolson@lxbuild066
# screen -list (to list screen sessions)
# screen -r XXX (to reattach screen session num XXX)
# type exit in screen session to kill it

PROG_NAME = os.path.basename(sys.argv[0])

def optionparse():
    #global PROG_NAME, _debug, _dryrun, _verbose

    parser = opt.OptionParser(usage=("""%s [HOST] [Options]"""))

    parser.add_option('-p',
                      '--port',
                      type="int",
                      dest='port',
                      default=-1,
                      help='Connect to server on a particular port',
                      metavar='<PORT>',
                      )

    (options, args) = parser.parse_args()

    port = 0
    if options.port == -1:
        port = 8000
    else:
        port = options.port

    if len(args) == 0:
        args.append("localhost")

    if not len(args) == 1:
        parser.error("You must specify only one server")
        sys.exit()

    return (options, args[0], port)

def runclient(shost,sport):
    try:
        server = xmlrpclib.ServerProxy("http://%s:%s" % (shost,sport))    
        print server.testfn()
    except socket.error, detail:
        print "ERROR: Could not communicate with server:", detail
    except xml.parsers.expat.ExpatError, detail:
        print "ERROR: XML-RPC could not be parsed:", detail
    except xmlrpclib.ProtocolError, detail:
        print "ERROR: XML-RPC protocol error", detail, "possibly try using -L xxx:localhost:xxx if using ssh to forward"

def _main():
    (options, shost, sport) = optionparse()
    runclient(shost,sport)

# Remember that localhost is the loopback network: it does not provide
# or require any connection to the outside world. As such it is useful
# for testing purposes. If you want your server to be seen on other
# machines, you must use your real network address in stead of
# 'localhost'.

if __name__ == "__main__":
    _main()
