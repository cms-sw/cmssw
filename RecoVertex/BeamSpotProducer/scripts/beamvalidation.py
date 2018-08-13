#!/usr/bin/env python
#____________________________________________________________
#
#
# A very simple 
#
# Francisco Yumiceva
# yumiceva@fnal.gov
#
# Fermilab, 2010
#
#____________________________________________________________

"""
   beam spot validation

   A very simple script 

   usage: %prog -t <tag name>
   -o, --output    = OUTPUT: filename of output html file.
   -p, --path      = PATH: path to beam spot scripts.
   -w, --web       = WEB: html path to website.
   
   Francisco Yumiceva (yumiceva@fnal.gov)
   Fermilab 2010
   
"""
from __future__ import print_function


import os, string, re, sys, math
import commands, time
import six

#_______________OPTIONS________________
import optparse

USAGE = re.compile(r'(?s)\s*usage: (.*?)(\n[ \t]*\n|$)')

def nonzero(self): # will become the nonzero method of optparse.Values
    "True if options were given"
    for v in six.itervalues(self.__dict__):
        if v is not None: return True
    return False

optparse.Values.__nonzero__ = nonzero # dynamically fix optparse.Values

class ParsingError(Exception): pass

optionstring=""

def exit(msg=""):
    raise SystemExit(msg or optionstring.replace("%prog",sys.argv[0]))

def parse(docstring, arglist=None):
    global optionstring
    optionstring = docstring
    match = USAGE.search(optionstring)
    if not match: raise ParsingError("Cannot find the option string")
    optlines = match.group(1).splitlines()
    try:
        p = optparse.OptionParser(optlines[0])
        for line in optlines[1:]:
            opt, help=line.split(':')[:2]
            short,long=opt.split(',')[:2]
            if '=' in opt:
                action='store'
                long=long.split('=')[0]
            else:
                action='store_true'
            p.add_option(short.strip(),long.strip(),
                         action = action, help = help.strip())
    except (IndexError,ValueError):
        raise ParsingError("Cannot parse the option string correctly")
    return p.parse_args(arglist)

#_______________________________

def cmp_tags(a,b):

    yeara = int(a.split('_')[1])
    yearb = int(b.split('_')[1])

    if yeara < yearb: return -1
    if yeara > yearb: return 1

    suffix = "_offline"
    if a.find("_express") != -1:
        suffix = "_express"
    if a.find("_prompt") != -1:
        suffix = "_prompt"

    tmpa = a.replace("BeamSpotObjects_2009_v","")
    tmpa = tmpa.replace(suffix,"")

    tmpb = b.replace("BeamSpotObjects_2009_v","")
    tmpb = tmpb.replace(suffix,"")

    na = int(tmpa)
    nb = int(tmpb)
    if na < nb: return -1
    if na == nb: return 0
    if na > nb: return 1

#___

def dump_header(lines):

    lines.append('''
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd"><html>
<head><title>Beam Spot Calibration Status</title></head>

<body>

<H1>Status of the Beam Spot Calibration</H1>
<BR>
<BR>
This page is updated every 10 minutes:
<BR>
<strong><script src="datemod.js"
type="text/javascript"></script></strong>

''')

#____

def dump_footer(lines):

    lines.append('</body>\n</html>\n')

#______________
def write_tags(tags, lines):

    end = '\n'
    br = '<BR>'+end

    lines.append('<tr>'+end)
    for i in tags.keys():

        lines.append('<th>'+i)
        lines.append('</th>'+end)
    lines.append('</tr>'+end)

    for ntags in range(0,len(tags['offline'])):
        lines.append('<tr>'+end)
        for i in tags.keys():
            alist = tags[i]
            if ntags < len(tags[i]):
                lines.append('<td> '+alist[ntags]+' </td>'+end)
            else:
                lines.append('<td> </td>')
        lines.append('</tr>'+end)

#______________
def write_iovs(iovs, lines):

    end = '\n'
    br = '<BR>'+end
    # hearder
    lines.append('<tr>'+end)
    for i in iovs.keys():        
        lines.append('<th>'+i)
        lines.append('</th>'+end)
    lines.append('</tr>'+end)
    # lumi type
    lines.append('<tr>'+end)
    for i in iovs.keys():
        aIOVlist = iovs[i]
        aIOV = IOV()
        if len(aIOVlist) > 0:
            aIOV = aIOVlist[0]
        lines.append('<td> '+aIOV.type+' </td>'+end)
    lines.append('</tr>'+end)
    # print iovs
    for niovs in range(0,len(iovs[iovs.keys()[0]])):
        lines.append('<tr>'+end)
        for i in iovs.keys():
            aIOVlist = iovs[i]
            aIOV = IOV()
            if len(aIOVlist) > niovs:
                aIOV = aIOVlist[niovs]
            lines.append('<td> '+aIOV.IOVfirst +' - '+aIOV.IOVlast+' </td>'+end)
        lines.append('</tr>'+end)

#______________
def get_listoftags(dest, auth,):

    queryTags_cmd = "cmscond_list_iov -c "+dest+" -P "+auth+" -a | grep BeamSpotObjects"
    print(queryTags_cmd)
    outcmd = commands.getstatusoutput( queryTags_cmd )
    print(outcmd[1])

    listtags = outcmd[1].split()

    listtags_offline = []
    for itag in listtags:
        if itag[len(itag)-7:len(itag)] == "offline":
            listtags_offline.append(itag)
    listtags_express = []
    for itag in listtags:
        if itag[len(itag)-7:len(itag)] == "express":
            listtags_express.append(itag)
    listtags_prompt = []
    for itag in listtags:
        if itag[len(itag)-6:len(itag)] == "prompt":
            listtags_prompt.append(itag)

    listtags_offline.sort( cmp = cmp_tags )
    listtags_offline.reverse()
    listtags_express.sort( cmp = cmp_tags )
    listtags_express.reverse()
    listtags_prompt.sort( cmp = cmp_tags )
    listtags_prompt.reverse()

    result = {}
    result['offline'] = listtags_offline
    result['express'] = listtags_express
    result['prompt'] = listtags_prompt

    return result

#______________________
class IOV:
    def __init__(self):
        self.type = ''
        self.IOVfirst = ''
        self.IOVlast  = ''

#_________________
def get_lastIOVs( listoftags, dest, auth ):

    dbtags = ['offline','express','prompt']

    results = {}
    for itag in dbtags:

        lasttag = listoftags[itag][0]
        #fix for the moment to read old tags
        if itag != "offline":
            lasttag = listoftags[itag][1]

        queryIOVs_cmd = "cmscond_list_iov -c "+dest+" -P "+auth+" -t "+ lasttag
        print(queryIOVs_cmd)

        outcmd = commands.getstatusoutput( queryIOVs_cmd )

        tmparr = outcmd[1].split('\n')

        TimeType = tmparr[1].split()[1]
        listIOVs = []

        # look at number of payloads
        lastline =  tmparr[len(tmparr)-1].split()
        npayloads = int( lastline[len(lastline)-1] )

        maxIOVs = 3
        if npayloads < 3:
            maxIOVs = npayloads
        # pick the last three IOVs
        for i in range(0,maxIOVs):
            tmpline = tmparr[len(tmparr) -2 -i]
            aIOV = IOV()
            aIOV.IOVfirst = tmpline.split()[0] 
            aIOV.IOVlast =  tmpline.split()[1] 
            aIOV.type = TimeType
            listIOVs.append( aIOV )

        results[lasttag] = listIOVs

    return results

#
# lumi tools CondCore/Utilities/python/timeUnitHelper.py
def pack(high,low):
    """pack high,low 32bit unsigned int to one unsigned 64bit long long
       Note:the print value of result number may appear signed, if the sign bit is used.
    """
    h=high<<32
    return (h|low)

def unpack(i):
    """unpack 64bit unsigned long long into 2 32bit unsigned int, return tuple (high,low)
    """
    high=i>>32
    low=i&0xFFFFFFFF
    return(high,low)

def unpackLumiid(i):
    """unpack 64bit lumiid to dictionary {'run','lumisection'}
    """
    j=unpack(i)
    return {'run':j[0],'lumisection':j[1]}

#____________________________
def get_plots(path,output, iovs, tag):

    initial=iovs[len(iovs)-1].IOVfirst
    final =iovs[0].IOVfirst
    if iovs[0].type == "lumiid":
        initial = str(unpack(initial)[0])+":"+str(unpack(initial)[1])
        final =  str(unpack(final)[0])+":"+str(unpack(final)[1])

    initial = str(int(initial) -100 )
    cmd = path+"/plotBeamSpotDB.py -b -P -t "+tag+" -i "+initial +" -f "+final
    print(cmd)
    outcmd = commands.getstatusoutput( cmd )
    print(outcmd[1])

    cmd = "ls *.png"
    outcmd = commands.getstatusoutput( cmd )

    pngfiles = outcmd[1].split('\n')
    print(pngfiles)

    cmd = "cp *.png "+os.path.dirname(output)
    outcmd = commands.getstatusoutput( cmd )
    cmd = "rm *.png"
    outcmd = commands.getstatusoutput( cmd )

    pngfiles.sort()
    return pngfiles

#_______________________________
def write_plots(lines, plots,web):

    end = '\n'
    br = '<BR>'+end

    lines.append(br)
    lines.append('''
<table border="1">

''')
    for i in range(0,len(plots)):
        plot = plots[i]
        plot = os.path.basename(plot)
        if i%2 == 0:
            lines.append("<tr>"+end)
        lines.append("<td> <a href=\""+web+"/"+plot+"\"> <img src="+plot+" alt="+plot+" width='700' height='250' /> </a> </td>"+end)
        if i%2 == 1:
            lines.append("</tr>"+end)

    lines.append('</table>'+end)

#________________________________
def get_productionFiles( directory ):

    list = commands.getstatusoutput('ls -t '+directory)
    list = list[1].split()
    newlist = []
    for i in list:
        if i.find('BeamFit_')!=-1:
            newlist.append(i)

    return newlist
#_______________________________
def get_productionIOVs( directory ):

    files = get_productionFiles( directory )
    listofruns = []
    for f in files:
        ii = f.find('Run')
        arun = f[ii:len(f)-4]
        listofruns.append(arun)
    return listofruns

#______________________________
if __name__ == '__main__':


    # COMMAND LINE OPTIONS
    #################################
    option,args = parse(__doc__)
    if not args and not option: exit()

    #htmlwebsite = "http://cmsrocstor.fnal.gov/lpc1/cmsroc/yumiceva/tmp/"
    htmlwebsite = "https://yumiceva.web.cern.ch/yumiceva/beamspot/"
    if option.web: htmlwebsite = option.web

    ## Get the latest tags
    #dest = "frontier://cmsfrontier.cern.ch:8000/Frontier/CMS_COND_31X_BEAMSPOT"
    dest = "oracle://cms_orcoff_prod/CMS_COND_31X_BEAMSPOT"
    auth = "/afs/cern.ch/cms/DB/conddb"
    #cmscond_list_iov -c oracle://cms_orcoff_prep/CMS_COND_BEAMSPOT -P /afs/cern.ch/cms/DB/conddb  -a

    list_tags = get_listoftags( dest, auth)

    # Get the latest IOVs from last tag
    list_lastIOVs = get_lastIOVs( list_tags, dest, auth)

    # get latest processed runs
    processedruns = get_productionIOVs('/afs/cern.ch/cms/CAF/CMSCOMM/COMM_BSPOT/yumiceva/tmp_lumi_workflow/')

    # create web page
    lines = []
    end = '\n'
    br = '<BR>'+end

    dump_header(lines)

    lines.append('<h2>The three latest IOVs</h2>'+end)
    lines.append(br)
    lines.append('''
<table border="1">
''')
    write_iovs( list_lastIOVs, lines )
    lines.append('</table>'+end)

    lines.append('<h2>Latest data processed</h2>'+end)
    lines.append(br)
    #lines.append('to be written'+end)
    lines.append(processedruns[0]+', '+processedruns[1]+', '+processedruns[2])
    print(processedruns)
    lines.append(br)

    lines.append('<h2>The Latest Tags</h2>'+end)
    lines.append(br)
    lines.append('''
<table border="1">
''')
    write_tags( list_tags, lines)
    lines.append('</table>'+end)

    lasttag = list_lastIOVs.keys()[0]
    lines.append('<h2>Plots of the latest IOVs from condDB tag: '+lasttag+' </h2>'+end)
    lines.append(br)

    pngfiles = get_plots(option.path,option.output, list_lastIOVs[lasttag], lasttag)
    write_plots( lines, pngfiles, htmlwebsite )

    dump_footer(lines)

    outfile = open(option.output,'w')
    #print lines
    outfile.writelines( lines )








