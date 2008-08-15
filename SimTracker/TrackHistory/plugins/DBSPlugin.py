
# DBSInterce 
# Author:  Victor E. Bazterra (UIC) 2008

def get(dataset, site, verbose = 1):
    """
       Get all the files of a given dataset in a given site. 
    """

    host = "cmsweb.cern.ch/dbs_discovery/"
    port = 443
    dbsInst = "cms_dbs_prod_global"

    query = "find file where dataset=" + dataset + " and " + "site=" + site

    result = send(
        host = host,
        port = port,
        dbsInst = dbsInst,
        userInput = query,
        page = 0,
        limit = -1,
        debug = verbose
    )

    lfns = result.split('\n')

    if len(lfns) == 4:
       return []
    else:
       return list(reversed(lfns[3:-1]))

import httplib, urllib 

def send(host,port,dbsInst,userInput,page,limit,xml=0,case='on',iface='dd',details=0,cff=0,debug=0):
    """
       Send message to server, message should be an well formed XML document.
    """
    if xml: xml=1
    else:   xml=0
    if cff: cff=1
    else:   cff=0
    input=urllib.quote(userInput)
    if debug:
       httplib.HTTPConnection.debuglevel = 1
       print "Contact",host,port
    _port=443
    if host.find("http://")!=-1:
       _port=80
    if host.find("https://")!=-1:
       _port=443
    host=host.replace("http://","").replace("https://","")
    if host.find(":")==-1:
       port=_port
    prefix_path=""
    if host.find("/")!=-1:
       hs=host.split("/")
       host=hs[0]
       prefix_path='/'.join(hs[1:])
    if host.find(":")!=-1:
       host,port=host.split(":")
    port=int(port)
#    print "\n\n+++",host,port
    if port==443:
       http_conn = httplib.HTTPS(host,port)
    else:
       http_conn = httplib.HTTP(host,port)
    if details: details=1
    else:       details=0
    path='/aSearch?dbsInst=%s&html=0&caseSensitive=%s&_idx=%s&pagerStep=%s&userInput=%s&xml=%s&details=%s&cff=%s&method=%s'%(dbsInst,case,page,limit,input,xml,details,cff,iface)
    if prefix_path:
       path="/"+prefix_path+path[1:]
    http_conn.putrequest('POST',path)
    http_conn.putheader('Host',host)
    http_conn.putheader('Content-Type','text/html; charset=utf-8')
    http_conn.putheader('Content-Length',str(len(input)))
    http_conn.endheaders()
    http_conn.send(input)

    (status_code,msg,reply)=http_conn.getreply()
    data=http_conn.getfile().read()
    if debug or msg!="OK":
       print
       print http_conn.headers
       print "*** Send message ***"
       print input
       print "************************************************************************"
       print "status code:",status_code
       print "message:",msg
       print "************************************************************************"
       print reply
    return data
  
