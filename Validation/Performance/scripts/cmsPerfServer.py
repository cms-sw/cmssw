#!/usr/bin/env python
import cmsSimPerfPublish as cspp
import cmsPerfSuite as cps
import optparse as opt
import socket, os, sys, SimpleXMLRPCServer, threading

PROG_NAME = os.path.basename(sys.argv[0])

def optionparse():

    parser = opt.OptionParser(usage=("""%s [Options]""" % PROG_NAME))

    parser.add_option('-p',
                      '--port',
                      type="int",
                      dest='port',
                      default=-1,
                      help='Run server on a particular port',
                      metavar='<PORT>',
                      )    

    (options, args) = parser.parse_args()
        
    port = 0        
    if options.port == -1:
        port = 8000
    else:
        port = options.port

    return port

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

def request_benchmark(cmds):
    outs = []
    for cmd in cmds:
        outs.append(runcmd(cmd))
    return outs

def _main():
    sport = optionparse()
    server_thread = threading.Thread(target = runserv(sport))
    server_thread.setDaemon(True) # Allow process to finish if this is the only remaining thread
    server_thread.start()

if __name__ == "__main__":
    _main()

