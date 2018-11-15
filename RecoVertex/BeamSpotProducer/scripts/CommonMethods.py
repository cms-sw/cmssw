from __future__ import print_function
import math, re, optparse, commands, os, sys, time, datetime
from BeamSpotObj import BeamSpot
from IOVObj import IOV
import six

lockFile = ".lock"

###########################################################################################
def timeoutManager(type,timeout=-1,fileName=".timeout"):
    if timeout == 0:
        return 1
    timeFormat = "%a,%Y/%m/%d,%H:%M:%S"
    currentTime = time.gmtime()
    timeoutLine = type + ' ' + time.strftime(timeFormat, currentTime) + '\n'
    isTimeout = False
    alreadyThere = False
    timeoutType = -1;
    fileExist = os.path.isfile(fileName)
    text = ''
    fields = []
    reset = False
    if timeout == -1:
        reset = True
    if fileExist:
        file = open(fileName)
        for line in file:
            text += line
            fields = line.strip('\n').split(' ')
            if fields[0] == type:
                alreadyThere = True
                if reset:
                    text = text.replace(line,'')
                    continue

                fileTime = time.strptime(fields[1],timeFormat)
                myTime = time.mktime(fileTime)
                referenceTime = time.mktime(time.gmtime())
                daylight = 0
                if currentTime.tm_isdst == 0:
                    daylight = 3600
                elapsedTime = referenceTime-myTime-daylight
                if elapsedTime > timeout:
                    isTimeout = True
                    timeoutType = 1
                    print("Timeout! " + str(elapsedTime) + " seconds passed since the " + type + " timeout was set and you can't tolerate more than " + str(timeout) + " seconds!")
                else:
                    timeoutType = 0
                    print("Timeout of type " + type + " already exist and was generated " + str(elapsedTime) + " seconds ago at " + fields[1])

        file.close()

    if not fileExist or not alreadyThere and not reset:
        timeoutType = -1
        text += timeoutLine

    if not fileExist or not alreadyThere or isTimeout or (reset and alreadyThere):
        if fileExist:
            commands.getstatusoutput("rm -rf " + fileName)
        file = open(fileName,'w')
        file.write(text)
        file.close()

    return timeoutType


###########################################################################################
def setLockName(name):
    global lockFile
    lockFile = name

###########################################################################################
def checkLock():
    global lockFile
    if os.path.isfile(lockFile):
        return True
    else:
        return False

###########################################################################################
def lock():
    global lockFile
    commands.getstatusoutput( "touch " + lockFile)

###########################################################################################
def rmLock():
    global lockFile
    if checkLock():
        commands.getstatusoutput( "rm " + lockFile)

###########################################################################################
def exit(msg=""):
    rmLock()
    raise SystemExit(msg or optionstring.replace("%prog",sys.argv[0]))

###########################################################################################
def isnan(num):
    fnum = float(num)
    return fnum != fnum

###########################################################################################
# OPTIONS
###########################################################################################
USAGE = re.compile(r'(?s)\s*usage: (.*?)(\n[ \t]*\n|$)')

###########################################################################################
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

###########################################################################################
def nonzero(self): # will become the nonzero method of optparse.Values
    "True if options were given"
    for v in six.itervalues(self.__dict__):
        if v is not None: return True
    return False

###########################################################################################
optparse.Values.__nonzero__ = nonzero # dynamically fix optparse.Values

# END OPTIONS
###########################################################################################

###########################################################################################
class ParsingError(Exception): pass

###########################################################################################
# General utilities
###########################################################################################
###########################################################################################
def sendEmail(mailList,error):
    print("Sending email to " + mailList + " with body: " + error)
    list = mailList.split(',')
    for email in list:
        p = os.popen("mail -s \"Automatic workflow error\" " + email ,"w")
        p.write(error)
        status = p.close() 

###########################################################################################
def dirExists(dir):
    if dir.find("castor") != -1:
        lsCommand = "nsls " + dir
        output = commands.getstatusoutput( lsCommand )
        return not output[0]
    else:
        return os.path.exists(dir)

########################################################################
def ls(dir,filter=""):
    lsCommand      = ''
    listOfFiles    = []
    if dir.find('castor') != -1:
        lsCommand = 'ns'
    elif not os.path.exists(dir):
        print("ERROR: File or directory " + dir + " doesn't exist")
        return listOfFiles

    aCommand  = lsCommand  + 'ls '+ dir
    #aCommand  = lsCommand  + 'ls '+ dir + " | grep .txt"
    if filter != "":
        aCommand  += " | grep " + filter 

    tmpStatus = commands.getstatusoutput( aCommand )
    listOfFiles = tmpStatus[1].split('\n')
    if len(listOfFiles) == 1:
        if listOfFiles[0].find('No such file or directory') != -1:
            exit("ERROR: File or directory " + dir + " doesn't exist") 

    return listOfFiles            

########################################################################
def cp(fromDir,toDir,listOfFiles,overwrite=False,smallList=False):
    cpCommand   = ''
    copiedFiles = []
    if fromDir.find('castor') != -1 or toDir.find('castor') != -1 :
        cpCommand = 'rf'
    elif fromDir.find('resilient') != -1:
        cpCommand = 'dc'
    if fromDir[len(fromDir)-1] != '/':
        fromDir += '/'

    if toDir[len(toDir)-1] != '/':
        toDir += '/'

    for file in listOfFiles:
        if os.path.isfile(toDir+file):
            if overwrite:
                print("File " + file + " already exists in destination directory. We will overwrite it.")
            else:
                print("File " + file + " already exists in destination directory. We will Keep original file.")
                if not smallList:
                    copiedFiles.append(file)
                continue
        # copy to local disk
        aCommand = cpCommand + 'cp '+ fromDir + file + " " + toDir
        print(" >> " + aCommand)
        tmpStatus = commands.getstatusoutput( aCommand )
        if tmpStatus[0] == 0:
            copiedFiles.append(file)
        else:
            print("[cp()]\tERROR: Can't copy file " + file)
    return copiedFiles

########################################################################


###########################################################################################
# lumi tools CondCore/Utilities/python/timeUnitHelper.py
###########################################################################################
def pack(high,low):
    """pack high,low 32bit unsigned int to one unsigned 64bit long long
       Note:the print value of result number may appear signed, if the sign bit is used.
    """
    h=high<<32
    return (h|low)

###########################################################################################
def unpack(i):
    """unpack 64bit unsigned long long into 2 32bit unsigned int, return tuple (high,low)
    """
    high=i>>32
    low=i&0xFFFFFFFF
    return(high,low)

###########################################################################################
def unpackLumiid(i):
    """unpack 64bit lumiid to dictionary {'run','lumisection'}
    """
    j=unpack(i)
    return {'run':j[0],'lumisection':j[1]}
###########################################################################################
# end lumi tools
###########################################################################################

###########################################################################################
def cmp_list_run(a,b):
    if int(a.IOVfirst) < int(b.IOVfirst): return -1
    if int(a.IOVfirst) == int(b.IOVfirst): return 0
    if int(a.IOVfirst) > int(b.IOVfirst): return 1

###########################################################################################
def cmp_list_lumi(a,b):
    if int(a.Run) < int(b.Run): return -1
    if int(a.Run) == int(b.Run):
        if int(a.IOVfirst) < int(b.IOVfirst): return -1
        if int(a.IOVfirst) == int(b.IOVfirst): return 0
        if int(a.IOVfirst) > int(b.IOVfirst): return 1
    if int(a.Run) > int(b.Run) : return 1

###########################################################################################
def weight(x1, x1err,x2,x2err):
    #print "x1 = "+str(x1)+" +/- "+str(x1err)+" x2 = "+str(x2)+" +/- "+str(x2err)
    x1     = float(x1)
    x1err  = float(x1err)
    x2     = float(x2)
    x2err  = float(x2err)
    tmperr = 0.
    if x2err < 1e-6 :
        x2err = 1e-6
    if x1err < 1e-6:
        x1 = x2/(x2err * x2err)
        tmperr = 1/(x2err*x2err)
    else:
        x1 = x1/(x1err*x1err) + x2/(x2err * x2err)
        tmperr = 1/(x1err*x1err) + 1/(x2err*x2err)
    x1 = x1/tmperr
    x1err = 1/tmperr
    x1err = math.sqrt(x1err)
    return (str(x1), str(x1err))

###########################################################################################
def dump( beam, file):
    end = "\n"
    file.write("Runnumber "+beam.Run+end)
    file.write("BeginTimeOfFit "+str(beam.IOVBeginTime)+end)
    file.write("EndTimeOfFit "+str(beam.IOVEndTime)+end)
    file.write("LumiRange "+str(beam.IOVfirst)+" - "+str(beam.IOVlast)+end)
    dumpValues(beam, file)

###########################################################################################
def dumpValues( beam, file):
    end = "\n"
    file.write("Type "+str(beam.Type)+end)
    file.write("X0 "+str(beam.X)+end)
    file.write("Y0 "+str(beam.Y)+end)
    file.write("Z0 "+str(beam.Z)+end)
    file.write("sigmaZ0 "+str(beam.sigmaZ)+end)
    file.write("dxdz "+str(beam.dxdz)+end)
    file.write("dydz "+str(beam.dydz)+end)
    file.write("BeamWidthX "+beam.beamWidthX+end)
    file.write("BeamWidthY "+beam.beamWidthY+end)
    file.write("Cov(0,j) "+str(math.pow(float(beam.Xerr),2))+" 0 0 0 0 0 0"  +end)
    file.write("Cov(1,j) 0 "+str(math.pow(float(beam.Yerr),2))+" 0 0 0 0 0"  +end)
    file.write("Cov(2,j) 0 0 "+str(math.pow(float(beam.Zerr),2))+" 0 0 0 0"  +end)
    file.write("Cov(3,j) 0 0 0 "+str(math.pow(float(beam.sigmaZerr),2))+" 0 0 0"  +end)
    file.write("Cov(4,j) 0 0 0 0 "+str(math.pow(float(beam.dxdzerr),2))+" 0 0"  +end)
    file.write("Cov(5,j) 0 0 0 0 0 "+str(math.pow(float(beam.dydzerr),2))+" 0"  +end)
    file.write("Cov(6,j) 0 0 0 0 0 0 "+str(math.pow(float(beam.beamWidthXerr),2))  +end)
    file.write("EmittanceX 0"+end)
    file.write("EmittanceY 0"+end)
    file.write("BetaStar 0"+end)

###########################################################################################
def delta(x,xerr,nextx,nextxerr):
    #return math.fabs( float(x) - float(nextx) )/math.sqrt(math.pow(float(xerr),2) + math.pow(float(nextxerr),2))
    return ( float(x) - float(nextx), math.sqrt(math.pow(float(xerr),2) + math.pow(float(nextxerr),2)) )

def deltaSig( x ):
    return math.fabs(x[0])/x[1]

###########################################################################################
def readBeamSpotFile(fileName,listbeam=[],IOVbase="runbase", firstRun='1',lastRun='4999999999'):
    tmpbeam = BeamSpot()
    tmpbeamsize = 0

    #firstRun = "1"
    #lastRun  = "4999999999"
    if IOVbase == "lumibase" and firstRun=='1' and lastRun=='4999999999' :
        firstRun = "1:1"
        lastRun = "4999999999:4999999999"

    inputfiletype = 0
    #print "first = " +firstRun
    #print "last = " +lastRun

    # for bx
    maplist = {}
    hasBX = False

    tmpfile = open(fileName)
    atmpline = tmpfile.readline()
    if atmpline.find('Runnumber') != -1:
        inputfiletype = 1
        if len(atmpline.split()) > 2:
            hasBX = True
            print(" Input data has been calculated as function of BUNCH CROSSINGS.")
    tmpfile.seek(0)


    if inputfiletype ==1:

        tmpBX = 0
        for line in tmpfile:

            if line.find('Type') != -1:
                tmpbeam.Type = int(line.split()[1])
                tmpbeamsize += 1
            if line.find('X0') != -1:
                tmpbeam.X = line.split()[1]
                #tmpbeam.Xerr = line.split()[4]
                tmpbeamsize += 1
            #print " x = " + str(tmpbeam.X)
            if line.find('Y0') != -1:
                tmpbeam.Y = line.split()[1]
                #tmpbeam.Yerr = line.split()[4]
                tmpbeamsize += 1
            #print " y =" + str(tmpbeam.Y)
            if line.find('Z0') != -1 and line.find('sigmaZ0') == -1:
                tmpbeam.Z = line.split()[1]
                #tmpbeam.Zerr = line.split()[4]
                tmpbeamsize += 1
            if line.find('sigmaZ0') !=-1:
                tmpbeam.sigmaZ = line.split()[1]
                #tmpbeam.sigmaZerr = line.split()[5]
                tmpbeamsize += 1
            if line.find('dxdz') != -1:
                tmpbeam.dxdz = line.split()[1]
                #tmpbeam.dxdzerr = line.split()[4]
                tmpbeamsize += 1
            if line.find('dydz') != -1:
                tmpbeam.dydz = line.split()[1]
                #tmpbeam.dydzerr = line.split()[4]
                tmpbeamsize += 1
            if line.find('BeamWidthX') != -1:
                tmpbeam.beamWidthX = line.split()[1]
                #tmpbeam.beamWidthXerr = line.split()[6]
                tmpbeamsize += 1
            if line.find('BeamWidthY') != -1:
                tmpbeam.beamWidthY = line.split()[1]
                #tmpbeam.beamWidthYerr = line.split()[6]
                tmpbeamsize += 1
            if line.find('Cov(0,j)') != -1:
                tmpbeam.Xerr = str(math.sqrt( float( line.split()[1] ) ) )
                tmpbeamsize += 1
            if line.find('Cov(1,j)') != -1:
                tmpbeam.Yerr = str(math.sqrt( float( line.split()[2] ) ) )
                tmpbeamsize += 1
            if line.find('Cov(2,j)') != -1:
                tmpbeam.Zerr = str(math.sqrt( float( line.split()[3] ) ) )
                tmpbeamsize += 1
            if line.find('Cov(3,j)') != -1:
                tmpbeam.sigmaZerr = str(math.sqrt( float( line.split()[4] ) ) )
                tmpbeamsize += 1
            if line.find('Cov(4,j)') != -1:
                tmpbeam.dxdzerr = str(math.sqrt( float( line.split()[5] ) ) )
                tmpbeamsize += 1
            if line.find('Cov(5,j)') != -1:
                tmpbeam.dydzerr = str(math.sqrt( float( line.split()[6] ) ) )
                tmpbeamsize += 1
            if line.find('Cov(6,j)') != -1:
                tmpbeam.beamWidthXerr = str(math.sqrt( float( line.split()[7] ) ) )
                tmpbeam.beamWidthYerr = tmpbeam.beamWidthXerr
                tmpbeamsize += 1
            if line.find('LumiRange')  != -1:
                if IOVbase=="lumibase":
                    tmpbeam.IOVfirst = line.split()[1]
                    tmpbeam.IOVlast = line.split()[3]
                tmpbeamsize += 1
            if line.find('Runnumber') != -1:
                tmpbeam.Run = line.split()[1]
                if IOVbase == "runbase":
                    tmpbeam.IOVfirst = line.split()[1]
                    tmpbeam.IOVlast = line.split()[1]
                if hasBX:
                    tmpBX = line.split()[3]
                tmpbeamsize += 1
            if line.find('BeginTimeOfFit') != -1:
                tmpbeam.IOVBeginTime = line.split()[1] +" "+line.split()[2] +" "+line.split()[3]
                if IOVbase =="timebase":
                    tmpbeam.IOVfirst =  time.mktime( time.strptime(line.split()[1] +  " " + line.split()[2] + " " + line.split()[3],"%Y.%m.%d %H:%M:%S %Z") )
                tmpbeamsize += 1
            if line.find('EndTimeOfFit') != -1:
                tmpbeam.IOVEndTime = line.split()[1] +" "+line.split()[2] +" "+line.split()[3]
                if IOVbase =="timebase":
                    tmpbeam.IOVlast = time.mktime( time.strptime(line.split()[1] +  " " + line.split()[2] + " " + line.split()[3],"%Y.%m.%d %H:%M:%S %Z") )
                tmpbeamsize += 1
            if tmpbeamsize == 20:
                if IOVbase=="lumibase":
                    tmprunfirst = int(firstRun.split(":")[0])
                    tmprunlast  = int(lastRun.split(":")[0])
                    tmplumifirst = int(firstRun.split(":")[1])
                    tmplumilast  = int(lastRun.split(":")[1])
                    acceptiov1 = acceptiov2 = False
                    # check lumis in the same run
                    if tmprunfirst == tmprunlast and int(tmpbeam.Run)==tmprunfirst:
                        if int(tmpbeam.IOVfirst) >= tmplumifirst and int(tmpbeam.IOVlast)<=tmplumilast:
                            acceptiov1 = acceptiov2 = True
                    # if different runs make sure you select the correct range of lumis
                    elif int(tmpbeam.Run) == tmprunfirst:
                        if int(tmpbeam.IOVfirst) >= tmplumifirst: acceptiov1 = True
                    elif int(tmpbeam.Run) == tmprunlast:
                        if int(tmpbeam.IOVlast) <= tmplumilast: acceptiov2 = True
                    elif tmprunfirst <= int(tmpbeam.Run) and tmprunlast >= int(tmpbeam.Run): 
                        acceptiov1 = acceptiov2 = True

                    if acceptiov1 and acceptiov2:
                        if tmpbeam.Type != 2:
                            print("invalid fit, skip Run "+str(tmpbeam.Run)+" IOV: "+str(tmpbeam.IOVfirst) + " to "+ str(tmpbeam.IOVlast))
                        elif isnan(tmpbeam.Z) or isnan(tmpbeam.Zerr) or isnan(tmpbeam.sigmaZerr) or isnan(tmpbeam.beamWidthXerr) or isnan(tmpbeam.beamWidthYerr):
                            print("invalid fit, NaN values!! skip Run "+str(tmpbeam.Run)+" IOV: "+str(tmpbeam.IOVfirst) + " to "+ str(tmpbeam.IOVlast))                       
                        elif hasBX:
                            if (tmpBX in maplist) == False:
                                maplist[tmpBX] = [tmpbeam]
                            else:
                                maplist[tmpBX].append(tmpbeam)
                        else:
                            listbeam.append(tmpbeam)

                elif int(tmpbeam.IOVfirst) >= int(firstRun) and int(tmpbeam.IOVlast) <= int(lastRun):
                    if tmpbeam.Type != 2:
                        print("invalid fit, skip Run "+str(tmpbeam.Run)+" IOV: "+str(tmpbeam.IOVfirst) + " to "+ str(tmpbeam.IOVlast))
                    elif isnan(tmpbeam.Z) or isnan(tmpbeam.Zerr) or isnan(tmpbeam.sigmaZerr) or isnan(tmpbeam.beamWidthXerr) or isnan(tmpbeam.beamWidthYerr):
                        print("invalid fit, NaN values!! skip Run "+str(tmpbeam.Run)+" IOV: "+str(tmpbeam.IOVfirst) + " to "+ str(tmpbeam.IOVlast))
                    else:
                        listbeam.append(tmpbeam)

                tmpbeamsize = 0
                tmpbeam = BeamSpot()
                tmpBX = 0
    else:

        for line in tmpfile:

            if line.find('X0') != -1:
                tmpbeam.X = line.split()[2]
                tmpbeam.Xerr = line.split()[4]
                tmpbeamsize += 1
            #print " x = " + str(tmpbeam.X)
            if line.find('Y0') != -1:
                tmpbeam.Y = line.split()[2]
                tmpbeam.Yerr = line.split()[4]
                tmpbeamsize += 1
            #print " y =" + str(tmpbeam.Y)
            if line.find('Z0') != -1 and line.find('Sigma Z0') == -1:
                tmpbeam.Z = line.split()[2]
                tmpbeam.Zerr = line.split()[4]
                tmpbeamsize += 1
            #print " z =" + str(tmpbeam.Z)
            if line.find('Sigma Z0') !=-1:
                tmpbeam.sigmaZ = line.split()[3]
                tmpbeam.sigmaZerr = line.split()[5]
                tmpbeamsize += 1
            if line.find('dxdz') != -1:
                tmpbeam.dxdz = line.split()[2]
                tmpbeam.dxdzerr = line.split()[4]
                tmpbeamsize += 1
            if line.find('dydz') != -1:
                tmpbeam.dydz = line.split()[2]
                tmpbeam.dydzerr = line.split()[4]
                tmpbeamsize += 1
            if line.find('Beam Width X') != -1:
                tmpbeam.beamWidthX = line.split()[4]
                tmpbeam.beamWidthXerr = line.split()[6]
                tmpbeamsize += 1
            if line.find('Beam Width Y') != -1:
                tmpbeam.beamWidthY = line.split()[4]
                tmpbeam.beamWidthYerr = line.split()[6]
                tmpbeamsize += 1
        #if line.find('Run ') != -1:
            if line.find('for runs')  != -1:
            #tmpbeam.IOVfirst = line.split()[6].strip(',')
                tmpbeam.Run      = line.split()[2]
                if IOVbase == "runbase":
                    tmpbeam.IOVfirst = line.split()[2]
                    tmpbeam.IOVlast = line.split()[4]
                tmpbeamsize += 1
            if line.find('LumiSection')  != -1:
                if IOVbase=="lumibase":
                    tmpbeam.IOVfirst = line.split()[10]
                    tmpbeam.IOVlast = line.split()[10]
                tmpbeamsize += 1
            if tmpbeamsize == 10:

                if IOVbase=="lumibase":
                    tmprunfirst = int(firstRun.split(":")[0])
                    tmprunlast  = int(lastRun.split(":")[0])
                    tmplumifirst = int(firstRun.split(":")[1])
                    tmplumilast  = int(lastRun.split(":")[1])
                    acceptiov1 = acceptiov2 = False
                    # check lumis in the same run
                    if tmprunfirst == tmprunlast and int(tmpbeam.Run)==tmprunfirst:
                        if int(tmpbeam.IOVfirst) >= tmplumifirst and int(tmpbeam.IOVlast)<=tmplumilast:
                            acceptiov1 = acceptiov2 = True
                    # if different runs make sure you select the correct range of lumis
                    elif int(tmpbeam.Run) == tmprunfirst:
                        if int(tmpbeam.IOVfirst) >= tmplumifirst: acceptiov1 = True
                    elif int(tmpbeam.Run) == tmprunlast:
                        if int(tmpbeam.IOVlast) <= tmplumilast: acceptiov2 = True
                    elif tmprunfirst <= int(tmpbeam.Run) and tmprunlast >= int(tmpbeam.Run): 
                        acceptiov1 = acceptiov2 = True

                    if acceptiov1 and acceptiov2:
                        if isnan(tmpbeam.Z) or isnan(tmpbeam.Zerr) or isnan(tmpbeam.sigmaZerr) or isnan(tmpbeam.beamWidthXerr) or isnan(tmpbeam.beamWidthYerr):
                            print("invalid fit, NaN values!! skip Run "+str(tmpbeam.Run)+" IOV: "+str(tmpbeam.IOVfirst) + " to "+ str(tmpbeam.IOVlast))                       
                        elif hasBX:
                            if (tmpBX in maplist) == False:
                                maplist[tmpBX] = [tmpbeam]
                            else:
                                maplist[tmpBX].append(tmpbeam)
                        else:
                            listbeam.append(tmpbeam)

                elif int(tmpbeam.IOVfirst) >= int(firstRun) and int(tmpbeam.IOVlast) <= int(lastRun):
                    if isnan(tmpbeam.Z) or isnan(tmpbeam.Zerr) or isnan(tmpbeam.sigmaZerr) or isnan(tmpbeam.beamWidthXerr) or isnan(tmpbeam.beamWidthYerr):
                        print("invalid fit, NaN values!! skip Run "+str(tmpbeam.Run)+" IOV: "+str(tmpbeam.IOVfirst) + " to "+ str(tmpbeam.IOVlast))
                    else:
                        listbeam.append(tmpbeam)

                tmpbeamsize = 0
                tmpbeam = BeamSpot()
                tmpBX = 0

    tmpfile.close()
    print(" got total number of IOVs = " + str(len(listbeam)) + " from file " + fileName)
    #print " run " + str(listbeam[3].IOVfirst ) + " " + str( listbeam[3].X )
    if hasBX:
        return maplist
    else:
        return listbeam

###########################################################################################
# Sort and clean list of data for consecutive duplicates and bad fits
def sortAndCleanBeamList(listbeam=[],IOVbase="lumibase"):
    # sort the list
    if IOVbase == "lumibase":
        listbeam.sort( cmp = cmp_list_lumi )
    else:
        listbeam.sort( cmp = cmp_list_run )

    # first clean list of data for consecutive duplicates and bad fits
    tmpremovelist = []
    for ii in range(0,len(listbeam)):
        ibeam = listbeam[ii]
        datax = ibeam.IOVfirst
        #print str(ii) + "  " +datax
        if datax == '0' and IOVbase =="runbase":
            print(" iov = 0? skip this IOV = "+ str(ibeam.IOVfirst) + " to " + str(ibeam.IOVlast))
            tmpremovelist.append(ibeam)

        if ii < len(listbeam) -1:
            #print listbeam[ii+1].IOVfirst
            if IOVbase =="lumibase":
                if ibeam.Run == listbeam[ii+1].Run and ibeam.IOVfirst == listbeam[ii+1].IOVfirst:
                    print(" duplicate IOV = "+datax+", keep only last duplicate entry")
                    tmpremovelist.append(ibeam)
            elif datax == listbeam[ii+1].IOVfirst:
                print(" duplicate IOV = "+datax+", keep only last duplicate entry")
                tmpremovelist.append(ibeam)

    for itmp in tmpremovelist:
        listbeam.remove(itmp)

###########################################################################################
# CREATE FILE FOR PAYLOADS
def createWeightedPayloads(fileName,listbeam=[],weighted=True):
    newlistbeam = []
    tmpbeam = BeamSpot()
    docreate = True
    countlumi = 0
    tmprun = ""
    maxNlumis = 60
    if weighted:
        maxNlumis = 999999999
    for ii in range(0,len(listbeam)):
        ibeam = listbeam[ii]
        inextbeam = BeamSpot()
        iNNbeam = BeamSpot()
        if docreate:
            tmpbeam.IOVfirst = ibeam.IOVfirst
            tmpbeam.IOVBeginTime = ibeam.IOVBeginTime
            tmpbeam.Run = ibeam.Run
            tmpbeam.Type = 2
        docheck = False
        docreate = False
        #print "Currently testing ii="+str(ii)+" Lumi1: "+str(ibeam.IOVfirst)

        # check last iov
        if ii < len(listbeam) - 1: 
            inextbeam = listbeam[ii+1]
            docheck = True
            if ii < len(listbeam) -2:
                iNNbeam = listbeam[ii+2]
        else:
            print("close payload because end of data has been reached. Run "+ibeam.Run)
            docreate = True
        # check we run over the same run
        if ibeam.Run != inextbeam.Run:
            print("close payload because end of run "+ibeam.Run)
            docreate = True
        # check maximum lumi counts
        if countlumi == maxNlumis -1:
            print("close payload because maximum lumi sections accumulated within run "+ibeam.Run)
            docreate = True
            countlumi = 0
        # weighted average position
        (tmpbeam.X, tmpbeam.Xerr) = weight(tmpbeam.X, tmpbeam.Xerr, ibeam.X, ibeam.Xerr)
        (tmpbeam.Y, tmpbeam.Yerr) = weight(tmpbeam.Y, tmpbeam.Yerr, ibeam.Y, ibeam.Yerr)
        (tmpbeam.Z, tmpbeam.Zerr) = weight(tmpbeam.Z, tmpbeam.Zerr, ibeam.Z, ibeam.Zerr)
        (tmpbeam.sigmaZ, tmpbeam.sigmaZerr) = weight(tmpbeam.sigmaZ, tmpbeam.sigmaZerr, ibeam.sigmaZ, ibeam.sigmaZerr)
        (tmpbeam.dxdz, tmpbeam.dxdzerr) = weight(tmpbeam.dxdz, tmpbeam.dxdzerr, ibeam.dxdz, ibeam.dxdzerr)
        (tmpbeam.dydz, tmpbeam.dydzerr) = weight(tmpbeam.dydz, tmpbeam.dydzerr, ibeam.dydz, ibeam.dydzerr)
        #print "wx = " + ibeam.beamWidthX + " err= "+ ibeam.beamWidthXerr
        (tmpbeam.beamWidthX, tmpbeam.beamWidthXerr) = weight(tmpbeam.beamWidthX, tmpbeam.beamWidthXerr, ibeam.beamWidthX, ibeam.beamWidthXerr)
        (tmpbeam.beamWidthY, tmpbeam.beamWidthYerr) = weight(tmpbeam.beamWidthY, tmpbeam.beamWidthYerr, ibeam.beamWidthY, ibeam.beamWidthYerr)

        if weighted:
            docheck = False
        # check offsets
        #if False:
        if docheck:

            # define minimum limit
            min_limit = 0.0025

            # limit for x and y
            limit = float(ibeam.beamWidthX)/2.
            if limit < min_limit: limit = min_limit

            # check movements in X
            adelta1 = delta(ibeam.X, ibeam.Xerr, inextbeam.X, inextbeam.Xerr)
            adelta2 = (0.,1.e9)
            adelta1dxdz = delta(ibeam.dxdz, ibeam.dxdzerr, inextbeam.dxdz, inextbeam.dxdzerr)
            adelta2dxdz = (0.,1.e9)
            adelta1dydz = delta(ibeam.dydz, ibeam.dydzerr, inextbeam.dydz, inextbeam.dydzerr)
            adelta2dydz = (0.,1.e9)
            adelta1widthx = delta(ibeam.beamWidthX, ibeam.beamWidthXerr, inextbeam.beamWidthX, inextbeam.beamWidthXerr)
            adelta2widthx = (0.,1.e9)
            adelta1widthy = delta(ibeam.beamWidthY, ibeam.beamWidthYerr, inextbeam.beamWidthY, inextbeam.beamWidthYerr)
            adelta2widthy = (0.,1.e9)
            adelta1z0 = delta(ibeam.Z, ibeam.Zerr, inextbeam.Z, inextbeam.Zerr)
            adelta1sigmaZ = delta(ibeam.sigmaZ, ibeam.sigmaZerr, inextbeam.sigmaZ, inextbeam.sigmaZerr)

            if iNNbeam.Type != -1:
                adelta2 = delta(inextbeam.X, inextbeam.Xerr, iNNbeam.X, iNNbeam.Xerr)
                adelta2dxdz = delta(inextbeam.dxdz, inextbeam.dxdzerr, iNNbeam.dxdz, iNNbeam.dxdzerr)
                adelta2dydz = delta(inextbeam.dydz, inextbeam.dydzerr, iNNbeam.dydz, iNNbeam.dydzerr)
                adelta2widthx = delta(inextbeam.beamWidthX, inextbeam.beamWidthXerr, iNNbeam.beamWidthX, iNNbeam.beamWidthXerr)
                adelta2widthy = delta(inextbeam.beamWidthY, inextbeam.beamWidthYerr, iNNbeam.beamWidthY, iNNbeam.beamWidthYerr)

            deltaX = deltaSig(adelta1) > 3.5 and adelta1[0] >= limit
            if ii < len(listbeam) -2:
                if deltaX==False and adelta1[0]*adelta2[0] > 0. and  math.fabs(adelta1[0]+adelta2[0]) >= limit:
                    #print " positive, "+str(adelta1[0]+adelta2[0])+ " limit="+str(limit)
                    deltaX = True
                elif deltaX==True and adelta1[0]*adelta2[0]<=0 and adelta2[0] != 0 and math.fabs(adelta1[0]/adelta2[0]) > 0.33 and math.fabs(adelta1[0]/adelta2[0]) < 3:
                    deltaX = False
                    #print " negative, "+str(adelta1[0]/adelta2[0])
                #else:
                #    print str(adelta1[0]/adelta2[0])

            # check movemnts in Y
            adelta1 = delta(ibeam.Y, ibeam.Yerr, inextbeam.Y, inextbeam.Yerr)
            adelta2 = (0.,1.e9)
            if iNNbeam.Type != -1:
                adelta2 = delta(inextbeam.Y, inextbeam.Yerr, iNNbeam.Y, iNNbeam.Yerr)

            deltaY = deltaSig(adelta1) > 3.5 and adelta1[0] >= limit
            if ii < len(listbeam) -2:
                if deltaY==False and adelta1[0]*adelta2[0] > 0. and  math.fabs(adelta1[0]+adelta2[0]) >= limit:
                    deltaY = True
                elif deltaY==True and adelta1[0]*adelta2[0]<=0 and adelta2[0] != 0 and math.fabs(adelta1[0]/adelta2[0]) > 0.33 and math.fabs(adelta1[0]/adelta2[0]) < 3:
                    deltaY = False
            # check movements in Z                                                    

            limit = float(ibeam.sigmaZ)/2.
            deltaZ = deltaSig(adelta1z0) > 3.5 and math.fabs(adelta1z0[0]) >= limit

            deltasigmaZ = deltaSig(adelta1sigmaZ) > 5.0

            # check dxdz
            adelta = delta(ibeam.dxdz, ibeam.dxdzerr, inextbeam.dxdz, inextbeam.dxdzerr)
            deltadxdz   = deltaSig(adelta) > 5.0
            if deltadxdz and adelta1dxdz[0]*adelta2dxdz[0]<=0 and adelta2dxdz[0] != 0 and math.fabs(adelta1dxdz[0]/adelta2dxdz[0]) > 0.33 and math.fabs(adelta1dxdz[0]/adelta2dxdz[0]) < 3:
                deltadxdz = False
            # check dydz
            adelta = delta(ibeam.dydz, ibeam.dydzerr, inextbeam.dydz, inextbeam.dydzerr)
            deltadydz   = deltaSig(adelta) > 5.0
            if deltadydz and adelta1dydz[0]*adelta2dydz[0]<=0 and adelta2dydz[0] != 0 and math.fabs(adelta1dydz[0]/adelta2dydz[0]) > 0.33 and math.fabs(adelta1dydz[0]/adelta2dydz[0]) < 3:
                deltadydz = False

            adelta = delta(ibeam.beamWidthX, ibeam.beamWidthXerr, inextbeam.beamWidthX, inextbeam.beamWidthXerr)
            deltawidthX = deltaSig(adelta) > 5
            if deltawidthX and adelta1widthx[0]*adelta2widthx[0]<=0 and adelta2widthx[0] != 0 and math.fabs(adelta1widthx[0]/adelta2widthx[0]) > 0.33 and math.fabs(adelta1widthx[0]/adelta2widthx[0]) < 3:
                deltawidthX = False

            adelta = delta(ibeam.beamWidthY, ibeam.beamWidthYerr, inextbeam.beamWidthY, inextbeam.beamWidthYerr) 
            deltawidthY = deltaSig(adelta) > 5
            if deltawidthY and adelta1widthy[0]*adelta2widthy[0]<=0 and adelta2widthy[0] != 0 and math.fabs(adelta1widthy[0]/adelta2widthy[0]) > 0.33 and math.fabs(adelta1widthy[0]/adelta2widthy[0]) < 3:
                deltawidthY = False
            #if iNNbeam.Type != -1:
            #    deltaX = deltaX and delta(ibeam.X, ibeam.Xerr, iNNbeam.X, iNNbeam.Xerr) > 1.5
            #    deltaY = deltaY and delta(ibeam.Y, ibeam.Yerr, iNNbeam.Y, iNNbeam.Yerr) > 1.5
            #    deltaZ = deltaZ and delta(ibeam.Z, ibeam.Zerr, iNNbeam.Z, iNNbeam.Zerr) > 1.5
            #		
            #    deltasigmaZ = deltasigmaZ and delta(ibeam.sigmaZ, ibeam.sigmaZerr, iNNbeam.sigmaZ, iNNbeam.sigmaZerr) > 2.5
            #    deltadxdz   = deltadxdz and delta(ibeam.dxdz, ibeam.dxdzerr, iNNbeam.dxdz, iNNbeam.dxdzerr) > 2.5
            #    deltadydz   = deltadydz and delta(ibeam.dydz, ibeam.dydzerr, iNNbeam.dydz, iNNbeam.dydzerr) > 2.5
            #
            #    deltawidthX = deltawidthX and delta(ibeam.beamWidthX, ibeam.beamWidthXerr, iNNbeam.beamWidthX, iNNbeam.beamWidthXerr) > 3
            #    deltawidthY = deltawidthY and delta(ibeam.beamWidthY, ibeam.beamWidthYerr, iNNbeam.beamWidthY, iNNbeam.beamWidthYerr) > 3

            if deltaX or deltaY or deltaZ or deltasigmaZ or deltadxdz or deltadydz or deltawidthX or deltawidthY:
                docreate = True
                #print "shift here: x="+str(deltaX)+" y="+str(deltaY)
                #print "x1 = "+ibeam.X + " x1err = "+ibeam.Xerr
                #print "x2 = "+inextbeam.X + " x2err = "+inextbeam.Xerr
                #print "Lumi1: "+str(ibeam.IOVfirst) + " Lumi2: "+str(inextbeam.IOVfirst)
                #print " x= "+ibeam.X+" +/- "+ibeam.Xerr
                #print "weighted average x = "+tmpbeam.X +" +//- "+tmpbeam.Xerr
                print("close payload because of movement in X= "+str(deltaX)+", Y= "+str(deltaY) + ", Z= "+str(deltaZ)+", sigmaZ= "+str(deltasigmaZ)+", dxdz= "+str(deltadxdz)+", dydz= "+str(deltadydz)+", widthX= "+str(deltawidthX)+", widthY= "+str(deltawidthY))
        if docreate:
            #if ii == len(listbeam)-1:
            tmpbeam.IOVlast = ibeam.IOVlast
            tmpbeam.IOVEndTime = ibeam.IOVEndTime
            print("  Run: "+tmpbeam.Run +" Lumi1: "+str(tmpbeam.IOVfirst) + " Lumi2: "+str(tmpbeam.IOVlast))
            newlistbeam.append(tmpbeam)
            tmpbeam = BeamSpot()
            countlumi = 0
        tmprun = ibeam.Run
        countlumi += 1

    payloadfile = open(fileName,"w")
    for iload in newlistbeam:
        dump( iload, payloadfile )
    payloadfile.close()
    return newlistbeam
###########################################################################################
def createWeightedPayloadsNew(fileName,listbeam=[],weighted=True):
    newlistbeam = []
    docreate = False
    docheck = False
    lastPayload = listbeam[0]

    firstToUse = 0
    lastToUse = 0
    for ii in range(0,len(listbeam)):
        docreate = False
        if docheck:
            deltaX = delta(ibeam.X, ibeam.Xerr, inextbeam.X, inextbeam.Xerr) > 1.5
            deltaY = delta(ibeam.Y, ibeam.Yerr, inextbeam.Y, inextbeam.Yerr) > 1.5
            deltaZ = delta(ibeam.Z, ibeam.Zerr, inextbeam.Z, inextbeam.Zerr) > 2.5

            deltasigmaZ = delta(ibeam.sigmaZ, ibeam.sigmaZerr, inextbeam.sigmaZ, inextbeam.sigmaZerr) > 2.5
            deltadxdz   = delta(ibeam.dxdz, ibeam.dxdzerr, inextbeam.dxdz, inextbeam.dxdzerr) > 2.5
            deltadydz   = delta(ibeam.dydz, ibeam.dydzerr, inextbeam.dydz, inextbeam.dydzerr) > 2.5

            deltawidthX = delta(ibeam.beamWidthX, ibeam.beamWidthXerr, inextbeam.beamWidthX, inextbeam.beamWidthXerr) > 3
            deltawidthY = delta(ibeam.beamWidthY, ibeam.beamWidthYerr, inextbeam.beamWidthY, inextbeam.beamWidthYerr) > 3

            #if iNNbeam.Type != -1:
            #    deltaX = deltaX and delta(ibeam.X, ibeam.Xerr, iNNbeam.X, iNNbeam.Xerr) > 1.5
            #    deltaY = deltaY and delta(ibeam.Y, ibeam.Yerr, iNNbeam.Y, iNNbeam.Yerr) > 1.5
            #    deltaZ = deltaZ and delta(ibeam.Z, ibeam.Zerr, iNNbeam.Z, iNNbeam.Zerr) > 1.5
            #		
            #    deltasigmaZ = deltasigmaZ and delta(ibeam.sigmaZ, ibeam.sigmaZerr, iNNbeam.sigmaZ, iNNbeam.sigmaZerr) > 2.5
            #    deltadxdz   = deltadxdz and delta(ibeam.dxdz, ibeam.dxdzerr, iNNbeam.dxdz, iNNbeam.dxdzerr) > 2.5
            #    deltadydz   = deltadydz and delta(ibeam.dydz, ibeam.dydzerr, iNNbeam.dydz, iNNbeam.dydzerr) > 2.5
            #
            #    deltawidthX = deltawidthX and delta(ibeam.beamWidthX, ibeam.beamWidthXerr, iNNbeam.beamWidthX, iNNbeam.beamWidthXerr) > 3
            #    deltawidthY = deltawidthY and delta(ibeam.beamWidthY, ibeam.beamWidthYerr, iNNbeam.beamWidthY, iNNbeam.beamWidthYerr) > 3

            if deltaX or deltaY or deltaZ or deltasigmaZ or deltadxdz or deltadydz or deltawidthX or deltawidthY:
                if ii != 0:
                    docreate = True
                    lastToUse = ii-1
                #print "shift here: x="+str(deltaX)+" y="+str(deltaY)
                #print "x1 = "+ibeam.X + " x1err = "+ibeam.Xerr
                #print "x2 = "+inextbeam.X + " x2err = "+inextbeam.Xerr
                #print "Lumi1: "+str(ibeam.IOVfirst) + " Lumi2: "+str(inextbeam.IOVfirst)
                #print " x= "+ibeam.X+" +/- "+ibeam.Xerr
                #print "weighted average x = "+tmpbeam.X +" +//- "+tmpbeam.Xerr
                print("close payload because of movement in X= "+str(deltaX)+", Y= "+str(deltaY) + ", Z= "+str(deltaZ)+", sigmaZ= "+str(deltasigmaZ)+", dxdz= "+str(deltadxdz)+", dydz= "+str(deltadydz)+", widthX= "+str(deltawidthX)+", widthY= "+str(deltawidthY))

        #WARNING this will only be fine for Run based IOVs
        if ii >= len(listbeam) - 1 or listbeam[ii].Run != listbeam[ii+1].Run :
            print("close payload because end of run has been reached. Run " + listbeam[ii].Run)
            docreate = True
            lastToUse = ii


         # check maximum lumi counts
#?        if countlumi == maxNlumis:
#?            print "close payload because maximum lumi sections accumulated within run "+ibeam.Run
#?            docreate = True
#?            countlumi = 0
        if docreate:
            tmpbeam = BeamSpot()
            for ibeam in listbeam[firstToUse:lastToUse]:
                (tmpbeam.X, tmpbeam.Xerr) = weight(tmpbeam.X, tmpbeam.Xerr, ibeam.X, ibeam.Xerr)
                (tmpbeam.Y, tmpbeam.Yerr) = weight(tmpbeam.Y, tmpbeam.Yerr, ibeam.Y, ibeam.Yerr)
                (tmpbeam.Z, tmpbeam.Zerr) = weight(tmpbeam.Z, tmpbeam.Zerr, ibeam.Z, ibeam.Zerr)
                (tmpbeam.sigmaZ, tmpbeam.sigmaZerr) = weight(tmpbeam.sigmaZ, tmpbeam.sigmaZerr, ibeam.sigmaZ, ibeam.sigmaZerr)
                (tmpbeam.dxdz, tmpbeam.dxdzerr) = weight(tmpbeam.dxdz, tmpbeam.dxdzerr, ibeam.dxdz, ibeam.dxdzerr)
                (tmpbeam.dydz, tmpbeam.dydzerr) = weight(tmpbeam.dydz, tmpbeam.dydzerr, ibeam.dydz, ibeam.dydzerr)
                #print "wx = " + ibeam.beamWidthX + " err= "+ ibeam.beamWidthXerr
                (tmpbeam.beamWidthX, tmpbeam.beamWidthXerr) = weight(tmpbeam.beamWidthX, tmpbeam.beamWidthXerr, ibeam.beamWidthX, ibeam.beamWidthXerr)
                (tmpbeam.beamWidthY, tmpbeam.beamWidthYerr) = weight(tmpbeam.beamWidthY, tmpbeam.beamWidthYerr, ibeam.beamWidthY, ibeam.beamWidthYerr)
            tmpbeam.IOVfirst     = listbeam[firstToUse].IOVfirst
            tmpbeam.IOVBeginTime = listbeam[firstToUse].IOVBeginTime
            tmpbeam.Run          = listbeam[firstToUse].Run
            tmpbeam.Type         = 2
            tmpbeam.IOVlast      = listbeam[lastToUse].IOVlast
            tmpbeam.IOVEndTime   = listbeam[lastToUse].IOVEndTime
            newlistbeam.append(tmpbeam)
            firstToUse = lastToUse+1
            print("Run: " + tmpbeam.Run + " Lumi1: " + str(tmpbeam.IOVfirst) + " Lumi2: " + str(tmpbeam.IOVlast))

    payloadfile = open(fileName,"w")
    for iload in newlistbeam:
        dump( iload, payloadfile )
    payloadfile.close()

###########################################################################################
def writeSqliteFile(sqliteFileName,tagName,timeType,beamSpotFile,sqliteTemplateFile,tmpDir="/tmp/"):
    writeDBOut = tmpDir + "write2DB_" + tagName + ".py"
    wFile      = open(sqliteTemplateFile)
    wNewFile   = open(writeDBOut,'w')

    writeDBTags = [('SQLITEFILE','sqlite_file:' + sqliteFileName),
                   ('TAGNAME',tagName),
                   ('TIMETYPE',timeType),
                   ('BEAMSPOTFILE',beamSpotFile)]

    for line in wFile:
        for itag in writeDBTags:
            line = line.replace(itag[0],itag[1])
        wNewFile.write(line)

    wNewFile.close()
    print("writing sqlite file ...")
    status_wDB = commands.getstatusoutput('cmsRun '+ writeDBOut)
    print(status_wDB[1])

    os.system("rm -f " + writeDBOut)
    return not status_wDB[0]

###########################################################################################
def readSqliteFile(sqliteFileName,tagName,sqliteTemplateFile,tmpDir="/tmp/"):
    readDBOut = tmpDir + "readDB_" + tagName + ".py"

    rFile = open(sqliteTemplateFile)
    rNewFile = open(readDBOut,'w')

    readDBTags = [('SQLITEFILE','sqlite_file:' + sqliteFileName),
                  ('TAGNAME',tagName)]

    for line in rFile:
        for itag in readDBTags:
            line = line.replace(itag[0],itag[1])
        rNewFile.write(line)

    rNewFile.close()
    status_rDB = commands.getstatusoutput('cmsRun '+ readDBOut)

    outtext = status_rDB[1]
    print(outtext)
    os.system("rm -f " + readDBOut)
    return not status_rDB[0]

###########################################################################################
def appendSqliteFile(combinedSqliteFileName, sqliteFileName, tagName, IOVSince, IOVTill ,tmpDir="/tmp/"):
    aCommand = "conddb_import -c sqlite_file:" + tmpDir + combinedSqliteFileName + " -f sqlite_file:" + sqliteFileName + " -i " + tagName + " -t " + tagName + " -b " + IOVSince + " -e " + IOVTill
    print(aCommand)
    std = commands.getstatusoutput(aCommand)
    print(std[1])
    return not std[0]

###########################################################################################
def uploadSqliteFile(sqliteFileDirName, sqliteFileName, dropbox="/DropBox"):
    # Changing permissions to metadata
    acmd = "chmod a+w " + sqliteFileDirName + sqliteFileName + ".txt"
    outcmd = commands.getstatusoutput(acmd)
    print(acmd)
#    print outcmd[1]
    if outcmd[0]:
        print("Can't change permission to file: " + sqliteFileDirName + sqliteFileName + ".txt")
        return False

    acmd = "cp " + sqliteFileDirName + sqliteFileName + ".db " + sqliteFileDirName + sqliteFileName + ".txt ." 
    print(acmd)
    outcmd = commands.getstatusoutput(acmd)
    print(outcmd[1])
    if outcmd[0]:
        print("Couldn't cd to " + sqliteFileDirName)
        return False

    acmd = "tar -cvjf " + sqliteFileName + ".tar.bz2 " + sqliteFileName + ".db " + sqliteFileName + ".txt"
    print(acmd)
    outcmd = commands.getstatusoutput(acmd)
    print(outcmd[1])
    if outcmd[0]:
        print("Couldn't zip the files!")
        return False

    acmd = "chmod a+w " + sqliteFileName + ".tar.bz2"
    outcmd = commands.getstatusoutput(acmd)
    print(acmd)
#    print outcmd[1]
    if outcmd[0]:
        print("Can't change permission to file: " + sqliteFileDirName + sqliteFileName + ".tar.bz2")
        return False

    acmd = "scp -p " + sqliteFileName + ".tar.bz2" + " webcondvm.cern.ch:" + dropbox
    print(acmd)
    outcmd = commands.getstatusoutput(acmd)
    print(outcmd[1])
    if outcmd[0]:
        print("Couldn't scp the files to DropBox!")
        return False


    acmd = "mv " + sqliteFileName + ".tar.bz2 " + sqliteFileDirName
    print(acmd)
    outcmd = commands.getstatusoutput(acmd)
    print(outcmd[1])
    if outcmd[0]:
        print("Couldn't mv the file to " + sqliteFileDirName)
        return False

    acmd = "rm " + sqliteFileName + ".db " + sqliteFileName + ".txt"
    print(acmd)
    outcmd = commands.getstatusoutput(acmd)
    print(outcmd[1])
    if outcmd[0]:
        print("Couldn't rm the db and txt files")
        return False

#    acmd = "scp -p " + sqliteFileDirName + sqliteFileName + ".txt webcondvm.cern.ch:/tmp"
#    outcmd = commands.getstatusoutput(acmd)
#    print acmd
#    print outcmd[1]
#    if outcmd[0]:
#        print "Can't change permission to file: " + sqliteFileName + ".txt"
#        return False

#    acmd = "ssh webcondvm.cern.ch \"mv /tmp/" + sqliteFileName + ".db /tmp/" + sqliteFileName + ".txt " + dropbox +"\""
#    print acmd
#    outcmd = commands.getstatusoutput(acmd)
#    print outcmd[1]
#    if outcmd[0]:
#        print "Can't move files from tmp to dropbox!"
        return False

#    acmd = "ssh webcondvm.cern.ch \"mv /tmp/" + final_sqlite_file_name + ".txt "+dropbox +"\""
#    outcmd = commands.getstatusoutput(acmd)
#    print acmd
#    print outcmd[1]
#    if outcmd[0]:
#        print "Can't change permission to file: " + sqliteFileName + ".txt"
#        return False

    return True

