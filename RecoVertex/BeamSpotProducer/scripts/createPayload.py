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
   -c, --copy   : Only copy files from input directory to test/workflow/files/
   -d, --data   = DATA: Data file, or directory with data files.
   -I, --IOVbase = IOVBASE: options: runbase(default), lumibase, timebase
   -o, --overwrite : Overwrite results files when copying.
   -m, --mergedSqlite = MERGEDSQLITE: Filename of merged sqlite file. default is Combined.db
   -n, --newarchive : Create a new archive directory when copying.
   -t, --tag    = TAG: Database tag name.
   -u, --upload : Upload files to offline drop box via scp.
   -z, --zlarge : Enlarge sigmaZ to 10 +/- 0.005 cm.
   
   Francisco Yumiceva (yumiceva@fnal.gov)
   Fermilab 2010
   
"""


import sys,os
import commands, re, time
import datetime
#_______________OPTIONS________________
import optparse

workflowdir             = 'test/workflow/'
workflowdirLastPayloads = workflowdir + 'lastPayloads/'
workflowdirTmp          = workflowdir + 'tmp/'
workflowdirArchive      = workflowdir + 'archive/'
optionstring            = ''
tagType                 = ''
USAGE = re.compile(r'(?s)\s*usage: (.*?)(\n[ \t]*\n|$)')

def nonzero(self): # will become the nonzero method of optparse.Values
    "True if options were given"
    for v in self.__dict__.itervalues():
        if v is not None: return True
    return False

optparse.Values.__nonzero__ = nonzero # dynamically fix optparse.Values

class ParsingError(Exception): pass


def exit(msg=""):
    raise SystemExit(msg or optionstring.replace("%prog",sys.argv[0]))

def parse(docstring, arglist=None):
    global optionstring
    global tagType
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

def copyToWorkflowdir(path):
    global workflowdirArchive
    lsCommand      = ''
    cpCommand      = ''
    listoffiles    = []
    tmplistoffiles = []
    if path.find('castor') != -1:
    	print "Getting files from castor ..."
    	lsCommand = 'ns'
    	cpCommand = 'rf'
    elif not os.path.exists(path):
        exit("ERROR: File or directory " + path + " doesn't exist") 

    if path[len(path)-4:len(path)] != '.txt':
 	if path[len(path)-1] != '/':
 	    path = path + '/'
	
 	aCommand  = lsCommand  + 'ls '+ path + " | grep .txt"

 	tmpstatus = commands.getstatusoutput( aCommand )
 	tmplistoffiles = tmpstatus[1].split('\n')
        if len(tmplistoffiles) == 1:
            if tmplistoffiles[0] == '':
                exit('ERROR: No files found in directory ' + path)
            if tmplistoffiles[0].find('No such file or directory') != -1:
                exit("ERROR: File or directory " + path + " doesn't exist") 
            
    else:
        tmplistoffiles.append(path[path.rfind('/')+1:len(path)])
	path = path[0:path.rfind('/')+1]

 
    archiveName = path
    if path == './':
        archiveName = os.getcwd() + '/'
        archiveName = archiveName[archiveName[:len(archiveName)-1].rfind('/')+1:len(archiveName)]
    if path[:len(path)-1].rfind('/') != -1:
        archiveName = path[path[:len(path)-1].rfind('/')+1:len(path)]

    workflowdirArchive = workflowdirArchive + archiveName
    if tagType != '' :
        workflowdirArchive = workflowdirArchive[:len(workflowdirArchive)-1] + '_' + tagType + '/'
    if not os.path.isdir(workflowdirArchive):
    	os.mkdir(workflowdirArchive)
    elif(option.newarchive):
#        tmpTime = str(datetime.datetime.now())
#        tmpTime = tmpTime.replace(' ','-')
#        tmpTime = tmpTime.replace('.','-')
#        workflowdirArchive = workflowdirArchive[:len(workflowdirArchive)-1] + '_' + tmpTime + '/'
#        os.mkdir(workflowdirArchive)
        for n in range(1,100000):
            tryDir = workflowdirArchive[:len(workflowdirArchive)-1] + '_' + str(n) + '/'
            if not os.path.isdir(tryDir):
                workflowdirArchive = tryDir
                os.mkdir(workflowdirArchive)
                break
            elif n == 100000-1:
                exit('ERROR: Unbelievable! do you ever clean ' + workflowdir + '?. I think you have to remove some directories!') 
                 
    for ifile in tmplistoffiles:
    	if ifile.find('.txt') != -1:
            if os.path.isfile(workflowdirArchive+"/"+ifile):
                if option.overwrite:
                    print "File " + ifile + " already exists in destination. We will overwrite it."
                else:
                    print "File " + ifile + " already exists in destination. Keep original file."
                    continue
    	    listoffiles.append( workflowdirArchive + ifile )
    	    # copy to local disk
    	    aCommand = cpCommand + 'cp '+ path + ifile + " " + workflowdirArchive
    	    print " >> " + aCommand
            tmpstatus = commands.getstatusoutput( aCommand )
    return listoffiles

def mkWorkflowdir():
    global workflowdir
    global workflowdirLastPayloads
    global workflowdirTmp
    global workflowdirArchive
    if not os.path.isdir(workflowdir):
	print "Making " + workflowdir + " directory..."
	os.mkdir(workflowdir)

    if not os.path.isdir(workflowdirLastPayloads):
    	os.mkdir(workflowdirLastPayloads)
    else:
    	os.system("rm -f "+ workflowdirLastPayloads + "*")
       
    if not os.path.isdir(workflowdirTmp):
    	os.mkdir(workflowdirTmp)
    else:
    	os.system("rm -f "+ workflowdirTmp + "*")

    if not os.path.isdir(workflowdirArchive):
    	os.mkdir(workflowdirArchive)

#__________END_OPTIONS_______________________________________________

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

if __name__ == '__main__':
    #if len(sys.argv) < 2:
#	print "\n [usage] createPayload <beamspot file> <tag name> <IOV since> <IOV till=-1=inf> <IOV comment> <destDB=oracle://cms_orcon_prod/CMS_COND_31X_BEAMSPOT>"
	#print " e.g. createPayload BeamFitResults_template.txt BeamSpotObjects_2009_v1_express 122745 \"\" \"beam spot for early collisions\" \"oracle://cms_orcon_prod/CMS_COND_31X_BEAMSPOT\"\n"
	#sys.exit()

    workflowdir             = os.getenv("CMSSW_BASE") + "/src/RecoVertex/BeamSpotProducer/test/workflow/"
    workflowdirLastPayloads = workflowdir + "lastPayloads/"
    workflowdirTmp          = workflowdir + "tmp/"
    workflowdirArchive      = workflowdir + "archive/"
     
     # COMMAND LINE OPTIONS
    #################################
    option,args = parse(__doc__)
    if not args and not option: exit()

    if ( (option.data and option.tag) or (option.data and option.copy)):
        mkWorkflowdir()
    
    if not option.data:
	print "ERROR: You must provide the data file or the a directory with data files"
	exit()

    if option.copy:
      copyToWorkflowdir(option.data)
      exit("Files copied in " + workflowdirArchive)
    
    tagname = ''
    if option.tag:
	tagname = option.tag
        if tagname.find("offline") != -1:
            tagType = "offline"
        elif tagname.find("prompt") != -1:
            tagType = "prompt"
        elif tagname.find("express") != -1 :
            tagType = "express"
        elif tagname.find("hlt") != -1:
            tagType = "hlt"
        else:
            print "I am assuming your tag is for the offline database..."
            tagType = "offline"

    else:	
	print "ERROR: You must provide the database tag name"
	exit()

    IOVbase = 'runbase'
    timetype = 'runnumber'
    if option.IOVbase:
        if option.IOVbase != "runbase" and option.IOVbase != "lumibase" and option.IOVbase != "timebase":
            print "\n\n unknown iov base option: "+ option.IOVbase +" \n\n\n"
            exit()
	IOVbase = option.IOVbase
    
    listoffiles = copyToWorkflowdir(option.data)

    # check if we have a single combined data file
    if len(listoffiles)==1:

        #split combined file
        os.system("split -l 23 "+listoffiles[0]+" "+listoffiles[0]+"_part_")
        
        listoffiles = copyToWorkflowdir( listoffiles[0].replace( os.path.basename(listoffiles[0]), "") )
    
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
	    if line.find("EndTimeOfFit") != -1:
		atime = time.strptime(line.split()[1] +  " " + line.split()[2] + " " + line.split()[3],"%Y.%m.%d %H:%M:%S %Z")
	    if line.find("LumiRange") != -1:
		alumis = line.strip('LumiRange ')
	    if line.find('Type') != -1 and line.split()[1] == '0':
		skip = True		
	if skip:
	    print " zero fit result, skip file " + beam_file + " with time stamp:"
	    print " run " + arun + " lumis " + alumis
	else:
	    sortedlist[atime] = beam_file
		
	tmpfile.close()

    keys = sortedlist.keys()
    keys.sort()

    # write combined data file
    if not os.path.isdir(workflowdirArchive + "AllIOVs"):
        os.mkdir(workflowdirArchive + "AllIOVs")
    allbeam_file = workflowdirArchive + "AllIOVs/" + tagname + "_all_IOVs.txt"
#    if os.path.isfile(allbeam_file):
        
    allfile = open( allbeam_file, 'a')
    print " merging all results into file: " + allbeam_file

    # check if merged sqlite file exists
    if os.path.exists(workflowdirArchive+"payloads/Combined.db"):
        os.system("rm "+workflowdirArchive+"payloads/Combined.db")
    
    
    nfile = 0
    for key in keys:
	
	iov_since = '1'
	iov_till = ''
	iov_comment = ''
	destDB = 'oracle://cms_orcon_prod/CMS_COND_31X_BEAMSPOT'
	iov_comment = 'Beam spot position'
	
	suffix = "_" + str(nfile)
	writedb_template = os.getenv("CMSSW_BASE") + "/src/RecoVertex/BeamSpotProducer/test/write2DB_template.py"
	readdb_template  = os.getenv("CMSSW_BASE") + "/src/RecoVertex/BeamSpotProducer/test/readDB_template.py"
        sqlite_file_name = tagname + suffix
	sqlite_file   = workflowdirTmp + sqlite_file_name + '.db'
	metadata_file = workflowdirTmp + sqlite_file_name + '.txt'
	nfile = nfile + 1
               
    #### WRITE sqlite file
	
	beam_file = sortedlist[key]

	print "read input beamspot file: " + beam_file
	tmpfile = open(beam_file)
        beam_file_tmp = workflowdirTmp + beam_file[beam_file.rfind('/')+1:] + ".tmp"
	newtmpfile = open(beam_file_tmp,"w")
	tmp_run = ""
	tmp_lumi = ""
	for line in tmpfile:
	    if line.find("Runnumber") != -1:
		iov_since = line.split()[1]
		iov_till = iov_since
		tmp_run = line.split()[1]
	    elif line.find("LumiRange") != -1:
                # pick the last lumi
		tmp_lumi = line.split()[3]
	    elif line.find("BeginTimeOfFit") == -1 and line.find("EndTimeOfFit") == -1 and line.find("LumiRange") == -1:
                if line.find("sigmaZ0") != -1 and option.zlarge:
                    line = "sigmaZ0 10\n"
                if line.find("Cov(3,j)") != -1 and option.zlarge:
                    line = "Cov(3,j) 0 0 0 2.5e-05 0 0 0\n"
		newtmpfile.write(line)
            allfile.write(line)
	
	# pack run number and lumi section
	if IOVbase == "lumibase":
            timetype = "lumiid"
	    iov_since = iov_till = str( pack(int(tmp_run), int(tmp_lumi)) )

	tmpfile.close()
	newtmpfile.close()
        if option.copy:
            continue
        
	beam_file = beam_file_tmp

	writedb_out = workflowdirTmp + "write2DB_" + tagname + suffix + ".py"
	wfile = open(writedb_template)
	wnewfile = open(writedb_out,'w')
	
	writedb_tags = [('SQLITEFILE','sqlite_file:' + sqlite_file),
			('TAGNAME',tagname),
                        ('TIMETYPE',timetype),
			('BEAMSPOTFILE',beam_file)]
    
	for line in wfile:

	    for itag in writedb_tags:

		line = line.replace(itag[0],itag[1])

	    wnewfile.write(line)

	wnewfile.close()
	print "writing sqlite file ..."
    #os.system("cmsRun "+ writedb_out)
        
	status_wDB = commands.getstatusoutput('cmsRun '+ writedb_out)
        #print status_wDB[1]
        
        #### Merge sqlite files
        time.sleep(5)
        print " merge sqlite file ..."
        acmd = "cmscond_export_iov -d sqlite_file:"+workflowdirArchive+"payloads/Combined.db -s sqlite_file:"+sqlite_file+ " -i "+tagname+" -t "+tagname+" -l sqlite_file:"+workflowdirTmp+"log.db"+" -b "+iov_since+" -e "+iov_since
        print acmd
        std = commands.getstatusoutput(acmd)
        print std[1]
        time.sleep(5)
    #print status_wDB[1]
	commands.getstatusoutput('rm -f ' + beam_file)
	os.system("rm "+ writedb_out)
    ##### READ and check sqlite file
	
	print "read back sqlite file to check content ..."
    
	readdb_out = workflowdirTmp + "readDB_" + tagname + ".py"
    
	rfile = open(readdb_template)
	rnewfile = open(readdb_out,'w')
    
	readdb_tags = [('SQLITEFILE','sqlite_file:' + sqlite_file),
		       ('TAGNAME',tagname)]

	for line in rfile:

	    for itag in readdb_tags:

		line = line.replace(itag[0],itag[1])

	    rnewfile.write(line)

	rnewfile.close()
	status_rDB = commands.getstatusoutput('cmsRun '+ readdb_out)
    
	outtext = status_rDB[1]
	print outtext
	os.system("rm "+ readdb_out)

    #### CREATE payload files for dropbox
            
        print " create payload card for dropbox ..."
        dfile = open(metadata_file,'w')
        
        dfile.write('destDB '+ destDB +'\n')
        dfile.write('tag '+ tagname +'\n')
        dfile.write('inputtag' +'\n')
        dfile.write('since ' + iov_since +'\n')
#        dfile.write('till ' + iov_till +'\n')
        if IOVbase == "runbase":
            dfile.write('Timetype runnumber\n')
        elif IOVbase == "lumibase":
            dfile.write('Timetype lumiid\n')
        checkType = tagType
        if tagType == "express":
            checkType = "hlt"
        dfile.write('IOVCheck ' + checkType + '\n')
        dfile.write('usertext ' + iov_comment +'\n')
        
        dfile.close()
        
        uuid = commands.getstatusoutput('uuidgen -t')[1]
        final_sqlite_file_name = tagname + '@' + uuid
        
        if not os.path.isdir(workflowdirArchive + 'payloads'):
            os.mkdir(workflowdirArchive + 'payloads')
        commands.getstatusoutput('cp ' + sqlite_file   + ' ' + workflowdirArchive + 'payloads/' + final_sqlite_file_name + '.db')
        commands.getstatusoutput('cp ' + metadata_file + ' ' + workflowdirArchive + 'payloads/' + final_sqlite_file_name + '.txt')

        commands.getstatusoutput('mv ' + sqlite_file   + ' ' + workflowdirLastPayloads + final_sqlite_file_name + '.db')
        commands.getstatusoutput('mv ' + metadata_file + ' ' + workflowdirLastPayloads + final_sqlite_file_name + '.txt')

        print workflowdirLastPayloads + final_sqlite_file_name + '.db'
        print workflowdirLastPayloads + final_sqlite_file_name + '.txt'
        
        if option.upload:
            print " scp files to offline Drop Box"
#            shellScriptName = 'dropBoxOffline_test.sh'
#            shellScript     = os.getenv("CMSSW_BASE") + "/src/RecoVertex/BeamSpotProducer/scripts/" + shellScriptName
#            if not os.path.exists(shellScript) :
#                wgetStatus = commands.getstatusoutput("wget http://condb.web.cern.ch/condb/DropBoxOffline/" + shellScriptName )
#                if os.path.exists(shellScriptName) :
#                    if not os.path.exists(shellScript):
#                        os.system("mv -f " + shellScriptName + " " + shellScript)
#                else:
#                    exit("Can't get the shell script to upload payloads. Check this twiki page: https://twiki.cern.ch/twiki/bin/viewauth/CMS/DropBoxOffline")
            commands.getstatusoutput("scp " + workflowdirLastPayloads + final_sqlite_file_name + ".db  webcondvm.cern.ch:/DropBox")
            commands.getstatusoutput("scp " + workflowdirLastPayloads + final_sqlite_file_name + ".txt webcondvm.cern.ch:/DropBox")

        print " done. Clean up."
    #### CLEAN up
	
	#print "DONE.\n"
	#print "Files ready to be move to beamspot dropbox:"
	#print tagname+"@"+uuid+".db"
	#print tagname+"@"+uuid+".txt"
    
    allfile.close()

    #### CREATE payload for merged output
            
    print " create MERGED payload card for dropbox ..."
    
    sqlite_file   = workflowdirArchive+'payloads/Combined.db'
    metadata_file = workflowdirArchive+'payloads/Combined.txt'
    dfile = open(metadata_file,'w')
        
    dfile.write('destDB '+ destDB +'\n')
    dfile.write('tag '+ tagname +'\n')
    dfile.write('inputtag' +'\n')
    dfile.write('since ' + iov_since +'\n')
    #        dfile.write('till ' + iov_till +'\n')
    if IOVbase == "runbase":
        dfile.write('Timetype runnumber\n')
    elif IOVbase == "lumibase":
        dfile.write('Timetype lumiid\n')
    checkType = tagType
    if tagType == "express":
        checkType = "hlt"
    dfile.write('IOVCheck ' + checkType + '\n')
    dfile.write('usertext ' + iov_comment +'\n')
        
    dfile.close()
        
    uuid = commands.getstatusoutput('uuidgen -t')[1]
    final_sqlite_file_name = tagname + '@' + uuid
        
    if not os.path.isdir(workflowdirArchive + 'payloads'):
        os.mkdir(workflowdirArchive + 'payloads')
    commands.getstatusoutput('cp ' + sqlite_file   + ' ' + workflowdirArchive + 'payloads/' + final_sqlite_file_name + '.db')
    commands.getstatusoutput('cp ' + metadata_file + ' ' + workflowdirArchive + 'payloads/' + final_sqlite_file_name + '.txt')

    commands.getstatusoutput('mv ' + sqlite_file   + ' ' + workflowdirLastPayloads + final_sqlite_file_name + '.db')
    commands.getstatusoutput('mv ' + metadata_file + ' ' + workflowdirLastPayloads + final_sqlite_file_name + '.txt')

    print workflowdirLastPayloads + final_sqlite_file_name + '.db'
    print workflowdirLastPayloads + final_sqlite_file_name + '.txt'
        
    
