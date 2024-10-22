#!/usr/bin/env python3
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
   -O, --Output = OUTPUT: Output directory for data files (workflow directory)
   -m, --merged : Use when data file contains combined results.
   -n, --newarchive : Create a new archive directory when copying.
   -t, --tag    = TAG: Database tag name.
   -T, --Test   : Upload files to Test dropbox for data validation.
   -u, --upload : Upload files to offline drop box via scp.
   -z, --zlarge : Enlarge sigmaZ to 10 +/- 0.005 cm.
   
   Francisco Yumiceva (yumiceva@fnal.gov)
   Fermilab 2010
   
"""
from __future__ import print_function


from builtins import range
import sys,os
import subprocess, re, time
import datetime
from CommonMethods import *

workflowdir             = 'test/workflow/'
workflowdirLastPayloads = workflowdir + 'lastPayloads/'
workflowdirTmp          = workflowdir + 'tmp/'
workflowdirArchive      = workflowdir + 'archive/'
optionstring            = ''
tagType                 = ''

def copyToWorkflowdir(path):
    global workflowdirArchive
    lsCommand      = ''
    cpCommand      = ''
    listoffiles    = []
    tmplistoffiles = []
    if path.find('castor') != -1:
        print("Getting files from castor ...")
        lsCommand = 'ns'
        cpCommand = 'rf'
    elif not os.path.exists(path):
        exit("ERROR: File or directory " + path + " doesn't exist") 

    if path[len(path)-4:len(path)] != '.txt':
        if path[len(path)-1] != '/':
            path = path + '/'

        aCommand  = lsCommand  + 'ls '+ path + " | grep .txt"

        tmpstatus = subprocess.getstatusoutput( aCommand )
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
                    print("File " + ifile + " already exists in destination. We will overwrite it.")
                else:
                    print("File " + ifile + " already exists in destination. Keep original file.")
                    listoffiles.append( workflowdirArchive + ifile )
                    continue
            listoffiles.append( workflowdirArchive + ifile )
            # copy to local disk
            aCommand = cpCommand + 'cp '+ path + ifile + " " + workflowdirArchive
            print(" >> " + aCommand)
            tmpstatus = subprocess.getstatusoutput( aCommand )
    return listoffiles

def mkWorkflowdir():
    global workflowdir
    global workflowdirLastPayloads
    global workflowdirTmp
    global workflowdirArchive
    if not os.path.isdir(workflowdir):
        print("Making " + workflowdir + " directory...")
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

###############################################################################################3
if __name__ == '__main__':
    #if len(sys.argv) < 2:
#	print "\n [usage] createPayload <beamspot file> <tag name> <IOV since> <IOV till=-1=inf> <IOV comment> <destDB=oracle://cms_orcon_prod/CMS_COND_31X_BEAMSPOT>"
        #print " e.g. createPayload BeamFitResults_template.txt BeamSpotObjects_2009_v1_express 122745 \"\" \"beam spot for early collisions\" \"oracle://cms_orcon_prod/CMS_COND_31X_BEAMSPOT\"\n"
        #sys.exit()


     # COMMAND LINE OPTIONS
    #################################
    option,args = parse(__doc__)
    if not args and not option: exit()

    workflowdir             = os.getenv("CMSSW_BASE") + "/src/RecoVertex/BeamSpotProducer/test/workflow/"
    if option.Output:
        workflowdir = option.Output
        if workflowdir[len(workflowdir)-1] != '/':
            workflowdir = workflowdir + '/'
    workflowdirLastPayloads = workflowdir + "lastPayloads/"
    workflowdirTmp          = workflowdir + "tmp/"
    workflowdirArchive      = workflowdir + "archive/"

    if ( (option.data and option.tag) or (option.data and option.copy)):
        mkWorkflowdir()

    if not option.data:
        print("ERROR: You must provide the data file or the a directory with data files")
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
            print("I am assuming your tag is for the offline database...")
            tagType = "offline"

    else:	
        print("ERROR: You must provide the database tag name")
        exit()

    IOVbase = 'runbase'
    timetype = 'runnumber'
    if option.IOVbase:
        if option.IOVbase != "runbase" and option.IOVbase != "lumibase" and option.IOVbase != "timebase":
            print("\n\n unknown iov base option: "+ option.IOVbase +" \n\n\n")
            exit()
        IOVbase = option.IOVbase

    listoffiles = copyToWorkflowdir(option.data)
    # sort list of data files in chronological order
    sortedlist = {}

    for beam_file in listoffiles:

        if len(listoffiles)==1 and option.merged:
            mergedfile = open(beam_file)
            alllines = mergedfile.readlines()
            npayloads = int(len(alllines)/23)
            for i in range(0,npayloads):
                block = alllines[i * 23: (i+1)*23]
                arun  = ''
                atime = ''
                alumi = ''
                for line in block:
                    if line.find('Runnumber') != -1:
                        arun = line.split()[1]
                    if line.find("EndTimeOfFit") != -1:
                        atime = time.strptime(line.split()[1] +  " " + line.split()[2] + " " + line.split()[3],"%Y.%m.%d %H:%M:%S %Z")
                    if line.find("LumiRange") != -1:
                        alumi = line.split()[3]
                    if line.find('Type') != -1 and line.split()[1] != '2':
                        continue
                sortedlist[int(pack(int(arun), int(alumi)))] = block
            break

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
                alumi = line.split()[3]
            if line.find('Type') != -1 and line.split()[1] == '0':
                skip = True		
        if skip:
            print(" zero fit result, skip file " + beam_file + " with time stamp:")
            print(" run " + arun + " lumis " + alumis)
        else:
            sortedlist[int(pack(int(arun), int(alumi)))] = beam_file

        tmpfile.close()

    keys = sorted(sortedlist.keys())

    # write combined data file
    if not os.path.isdir(workflowdirArchive + "AllIOVs"):
        os.mkdir(workflowdirArchive + "AllIOVs")
    allbeam_file = workflowdirArchive + "AllIOVs/" + tagname + "_all_IOVs.txt"
#    if os.path.isfile(allbeam_file):

    allfile = open( allbeam_file, 'a')
    print(" merging all results into file: " + allbeam_file)

    # check if merged sqlite file exists
    if os.path.exists(workflowdirArchive+"payloads/Combined.db"):
        os.system("rm "+workflowdirArchive+"payloads/Combined.db")


    nfile = 0
    iov_since_first = '1'
    total_files = len(keys)

    destDB = 'oracle://cms_orcon_prod/CMS_COND_31X_BEAMSPOT'
    if option.Test:
        destDB = 'oracle://cms_orcoff_prep/CMS_COND_BEAMSPOT'

    iov_comment = 'Beam spot position'
    for key in keys:

        iov_since = '1'
        iov_till = ''

        suffix = "_" + str(nfile)
        writedb_template = os.getenv("CMSSW_BASE") + "/src/RecoVertex/BeamSpotProducer/test/write2DB_template.py"
        readdb_template  = os.getenv("CMSSW_BASE") + "/src/RecoVertex/BeamSpotProducer/test/readDB_template.py"
        sqlite_file_name = tagname + suffix
        sqlite_file   = workflowdirTmp + sqlite_file_name + '.db'
        metadata_file = workflowdirTmp + sqlite_file_name + '.txt'
        nfile = nfile + 1

    #### WRITE sqlite file

        beam_file = sortedlist[key]
        tmp_datafilename = workflowdirTmp+"tmp_datafile.txt"
        if option.merged:
            tmpfile = open(tmp_datafilename,'w')
            tmpfile.writelines(sortedlist[key])
            tmpfile.close()
            beam_file = tmp_datafilename

        print("read input beamspot file: " + beam_file)
        tmpfile = open(beam_file)
        beam_file_tmp = workflowdirTmp + beam_file[beam_file.rfind('/')+1:] + ".tmp"
        newtmpfile = open(beam_file_tmp,"w")
        tmp_run = ""
        tmp_lumi_since = ""
        tmp_lumi_till = ""
        for line in tmpfile:
            if line.find("Runnumber") != -1:
                iov_since = line.split()[1]
                iov_till = iov_since
                tmp_run = line.split()[1]
            elif line.find("LumiRange") != -1:
                tmp_lumi_since = line.split()[1]
                tmp_lumi_till = line.split()[3]
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
            iov_since = str( pack(int(tmp_run), int(tmp_lumi_since)) )
            iov_till = str( pack(int(tmp_run), int(tmp_lumi_till)) )
        # keep first iov for merged output metafile
        if nfile == 1:
            iov_since_first = iov_since

        tmpfile.close()
        newtmpfile.close()
        if option.copy:
            continue

        beam_file = beam_file_tmp

        if not writeSqliteFile(sqlite_file,tagname,timetype,beam_file,writedb_template,workflowdirTmp):
            print("An error occurred while writing the sqlite file: " + sqlite_file)

        subprocess.getstatusoutput('rm -f ' + beam_file)
        ##### READ and check sqlite file
        readSqliteFile(sqlite_file,tagname,readdb_template,workflowdirTmp)

        #### Merge sqlite files
        if not os.path.isdir(workflowdirArchive + 'payloads'):
            os.mkdir(workflowdirArchive + 'payloads')

        print(" merge sqlite file ...")
        appendSqliteFile("Combined.db", sqlite_file, tagname, iov_since, iov_till ,workflowdirTmp)

        # keep last payload for express, and prompt tags
        if nfile == total_files:
            print(" this is the last IOV. You can use this payload for express and prompt conditions.")
            os.system("cp "+sqlite_file+ " "+workflowdirArchive+"payloads/express.db")
            print("a copy of this payload has been placed at:")
            print(workflowdirArchive+"payloads/express.db")

        # clean up
        os.system("rm "+ sqlite_file)
        print(" clean up done.")

    os.system("mv " + workflowdirTmp + "Combined.db " + workflowdirArchive + "payloads/")
    allfile.close()

    #### CREATE payload for merged output

    print(" create MERGED payload card for dropbox ...")

    sqlite_file   = workflowdirArchive+'payloads/Combined.db'
    metadata_file = workflowdirArchive+'payloads/Combined.txt'
    dfile = open(metadata_file,'w')

    dfile.write('destDB '+ destDB +'\n')
    dfile.write('tag '+ tagname +'\n')
    dfile.write('inputtag' +'\n')
    dfile.write('since ' + iov_since_first +'\n')
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

    uuid = subprocess.getstatusoutput('uuidgen -t')[1]
    final_sqlite_file_name = tagname + '@' + uuid

    if not os.path.isdir(workflowdirArchive + 'payloads'):
        os.mkdir(workflowdirArchive + 'payloads')
    subprocess.getstatusoutput('cp ' + sqlite_file   + ' ' + workflowdirArchive + 'payloads/' + final_sqlite_file_name + '.db')
    subprocess.getstatusoutput('cp ' + metadata_file + ' ' + workflowdirArchive + 'payloads/' + final_sqlite_file_name + '.txt')

    subprocess.getstatusoutput('mv ' + sqlite_file   + ' ' + workflowdirLastPayloads + final_sqlite_file_name + '.db')
    subprocess.getstatusoutput('mv ' + metadata_file + ' ' + workflowdirLastPayloads + final_sqlite_file_name + '.txt')

    print(workflowdirLastPayloads + final_sqlite_file_name + '.db')
    print(workflowdirLastPayloads + final_sqlite_file_name + '.txt')

    if option.upload:
        print(" scp files to offline Drop Box")
        dropbox = "/DropBox"
        if option.Test:
            dropbox = "/DropBox_test"

        uploadSqliteFile(workflowdirLastPayloads,final_sqlite_file_name,dropbox)
