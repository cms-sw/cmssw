#!/usr/bin/env python
#____________________________________________________________
#
#  createPayload
#
# A very simple way to create condition DB payloads
#
# Francisco Yumiceva
# yumiceva@fnal.gov
#
# Fermilab, 2009
#
#____________________________________________________________

"""
   createPayload.py

   A very simple script to handle payload for beam spot results

   usage: %prog -d <data file/directory> -t <tag name>
   -d, --data    = DATA: data file, or directory with data files.
   -n, --nodropbox : Do not upload files to the drop box.
   -t, --tag     = TAG: tag name.
   
   
   Francisco Yumiceva (yumiceva@fnal.gov)
   Fermilab 2010
   
"""


import sys,os
import commands, re, time

#_______________OPTIONS________________
import optparse

USAGE = re.compile(r'(?s)\s*usage: (.*?)(\n[ \t]*\n|$)')

def nonzero(self): # will become the nonzero method of optparse.Values
    "True if options were given"
    for v in self.__dict__.itervalues():
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

#__________END_OPTIONS_______________________________________________

if __name__ == '__main__':

    #if len(sys.argv) < 2:
#	print "\n [usage] createPayload <beamspot file> <tag name> <IOV since> <IOV till=-1=inf> <IOV comment> <destDB=oracle://cms_orcon_prod/CMS_COND_31X_BEAMSPOT>"
	#print " e.g. createPayload BeamFitResults_template.txt BeamSpotObjects_2009_v1_express 122745 \"\" \"beam spot for early collisions\" \"oracle://cms_orcon_prod/CMS_COND_31X_BEAMSPOT\"\n"
	#sys.exit()

     # COMMAND LINE OPTIONS
    #################################
    option,args = parse(__doc__)
    if not args and not option: exit()

    tagname = ''
    if not option.tag and not option.data:
	print " need to provide DB tag name and beam spot data"
	exit()

    tagname = option.tag
    workflowdir = 'test/workflow/'
    if not os.path.isdir(workflowdir):
	print " make directory to store temporal files in "+ workflowdir
	os.mkdir(workflowdir)
    else:
	print workflowdir + " directory already exists. Notice that we will use files from this directory."
	
    
    listoffiles = []

    # get list of data files
    if option.data.find('castor') != -1:
	print " get files from castor ..."
	acommand = 'nsls '+ option.data
	tmpstatus = commands.getstatusoutput( acommand )
	tmplistoffiles = tmpstatus[1].split('\n')
	for ifile in tmplistoffiles:
	    if ifile.find('.txt') != -1:
		listoffiles.append( workflowdir + ifile )
		# copy to local disk
		acommand = 'rfcp '+ option.data + ifile + " "+ workflowdir+"/."
		tmpstatus = commands.getstatusoutput( acommand )

    elif os.path.isdir( option.data ):
	print" get files from directory "+ option.data
	acommand = 'ls '+ option.data
	tmpstatus = commands.getstatusoutput( acommand )
	tmplistoffiles = tmpstatus[1].split('\n')
	dir = option.data
	if option.data[len(option.data)-1] != '/':
	    dir = dir + '/'
	for ifile in tmplistoffiles:
	    if ifile.find('.txt') != -1:
		listoffiles.append( dir + ifile )
    else:
	listoffiles.append( option.data )


    # sort list of data files in chronological order
    sortedlist = {}

    for beam_file in listoffiles:

	tmpfile = open(beam_file)
	atime = ''
	arun = ''
	alumis = ''
	skip = False
	for line in tmpfile:
	    if line.find('Runnumber') != -1:
		arun = line.split()[1]
	    if line.find("LumiRange") != -1:
		alumis = line.strip('LumiRange ')
	    if line.find('Type') != -1 and line.split()[1] == '0':
		skip = True		
	    if line.find("EndTimeOfFit") != -1:
		atime = time.strptime(line.split()[1] +  " " + line.split()[2] + " " + line.split()[3],"%Y.%m.%d %H:%M:%S %Z")
	if skip:
	    print " zero fit result, skip file " + beam_file + " with time stamp:"
	    print " run " + arun + " lumis " + alumis
	else:
	    sortedlist[atime] = beam_file
		
	tmpfile.close()

    keys = sortedlist.keys()
    keys.sort()

    allbeam_file = workflowdir + tagname + "_all_IOVs.txt"
    allfile = open( allbeam_file, 'w')
    
    nfile = 0

    for key in keys:
	
	iov_since = ''
	iov_till = ''
	iov_comment = ''
	destDB = 'oracle://cms_orcon_prod/CMS_COND_31X_BEAMSPOT'
	iov_comment = 'Beam spot position'
	
	suffix = "_"+str(nfile)
	writedb_template = "test/write2DB_template.py"
	readdb_template = "test/readDB_template.py"
	sqlite_file = "sqlite_file:"+ workflowdir + tagname + suffix +".db"
	nfile = nfile + 1
    #### WRITE sqlite file
	
	beam_file = sortedlist[key]

	print "read input beamspot file: " + beam_file
	tmpfile = open(beam_file)
	beam_file_tmp = beam_file+".tmp"
	newtmpfile = open(beam_file_tmp,"w")
	for line in tmpfile:
	    if line.find("Runnumber") != -1:
		iov_since = line.split()[1]
		iov_till = iov_since
	    elif line.find("BeginTimeOfFit") == -1 and line.find("EndTimeOfFit") == -1 and line.find("LumiRange") == -1:
		newtmpfile.write(line)
            allfile.write(line)
        
	tmpfile.close()
	newtmpfile.close()
	beam_file = beam_file_tmp

	writedb_out = workflowdir+"write2DB_"+tagname+suffix+".py"

	wfile = open(writedb_template)
	wnewfile = open(writedb_out,'w')
	
	writedb_tags = [('SQLITEFILE',sqlite_file),
			('TAGNAME',tagname),
			('BEAMSPOTFILE',beam_file)]
    
	for line in wfile:

	    for itag in writedb_tags:

		line = line.replace(itag[0],itag[1])

	    wnewfile.write(line)

	wnewfile.close()
	print "write sqlite file ..."
    #os.system("cmsRun "+ writedb_out)
        
	status_wDB = commands.getstatusoutput('cmsRun '+ writedb_out)
    #print status_wDB[1]
	commands.getstatusoutput('rm ' + beam_file)

    ##### READ and check sqlite file
	
	print "read back sqlite file to check content ..."
    
	readdb_out = "readDB_"+tagname+".py"
    
	rfile = open(readdb_template)
	rnewfile = open(readdb_out,'w')
    
	readdb_tags = [('SQLITEFILE',sqlite_file),
		       ('TAGNAME',tagname)]

	for line in rfile:

	    for itag in readdb_tags:

		line = line.replace(itag[0],itag[1])

	    rnewfile.write(line)

	rnewfile.close()
	status_rDB = commands.getstatusoutput('cmsRun '+ readdb_out)
    
	outtext = status_rDB[1]
	print outtext

    #### CREATE payload files for dropbox

        if not option.nodropbox:
            
            print " create payload card for dropbox ..."
            sqlitefile = sqlite_file.strip("sqlite_file")
            sqlitefile = sqlitefile.replace(":","")
            sqlitefile = sqlitefile.strip(".db")
            dropbox_file = sqlitefile+".txt"
            dfile = open(dropbox_file,'w')
            
            dfile.write('destDB '+ destDB +'\n')
            dfile.write('inputtag' +'\n')
            dfile.write('tag '+ tagname +'\n')
            dfile.write('since ' + iov_since +'\n')
            dfile.write('till ' + iov_till +'\n')
            dfile.write('usertext ' + "\""+ iov_comment +"\"" +'\n')
            
            dfile.close()
            
            uuid = commands.getstatusoutput('uuidgen -t')[1]
            
            
            commands.getstatusoutput('mv '+sqlitefile+".db "+sqlitefile+"@"+uuid+".db")
            commands.getstatusoutput('mv '+sqlitefile+".txt "+sqlitefile+"@"+uuid+".txt")
            
            print " scp files to Drop Box"
            commands.getstatusoutput("scp " + sqlitefile+"@"+uuid+".db  webcondvm.cern.ch:/DropBox")
            commands.getstatusoutput("scp " + sqlitefile+"@"+uuid+".txt webcondvm.cern.ch:/DropBox")
        
        print " done. Clean up."
    #### CLEAN up
	os.system("rm "+ writedb_out)
	os.system("rm "+ readdb_out)
	
	#print "DONE.\n"
	#print "Files ready to be move to beamspot dropbox:"
	#print tagname+"@"+uuid+".db"
	#print tagname+"@"+uuid+".txt"
    
    allfile.close()
    
        
