import math,re
import optparse
from BeamSpotObj import BeamSpot
from IOVObj import IOV

###########################################################################################
def exit(msg=""):
    raise SystemExit(msg or optionstring.replace("%prog",sys.argv[0]))

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
    for v in self.__dict__.itervalues():
        if v is not None: return True
    return False

###########################################################################################
optparse.Values.__nonzero__ = nonzero # dynamically fix optparse.Values

# END OPTIONS
###########################################################################################

###########################################################################################
class ParsingError(Exception): pass

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
    return math.fabs( float(x) - float(nextx) )/math.sqrt(math.pow(float(xerr),2) + math.pow(float(nextxerr),2))

###########################################################################################
def readBeamSpotFile(fileName,listbeam=[],IOVbase="lumibase"):
    tmpbeam = BeamSpot()
    tmpbeamsize = 0

    firstRun = "1"
    lastRun  = "4999999999"
    if IOVbase == "lumibase":
	firstRun = "1:1"
	lastRun = "4999999999:4999999999"

    inputfiletype = 0

    tmpfile = open(fileName)
    if tmpfile.readline().find('Runnumber') != -1:
	inputfiletype = 1
    tmpfile.seek(0)

    if inputfiletype ==1:
	
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
	    if line.find('LumiRange')  != -1 and IOVbase=="lumibase":
	    #tmpbeam.IOVfirst = line.split()[6].strip(',')
		tmpbeam.IOVfirst = line.split()[1]
		tmpbeam.IOVlast = line.split()[3]
		tmpbeamsize += 1
            if line.find('Runnumber') != -1:
		tmpbeam.Run = line.split()[1]
		if IOVbase == "runbase":
		    tmpbeam.IOVfirst = line.split()[1]
		    tmpbeam.IOVlast = line.split()[1]
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
			    print "invalid fit, skip Run "+str(tmpbeam.Run)+" IOV: "+str(tmpbeam.IOVfirst) + " to "+ str(tmpbeam.IOVlast)
			else:
			    listbeam.append(tmpbeam)

		elif int(tmpbeam.IOVfirst) >= int(firstRun) and int(tmpbeam.IOVlast) <= int(lastRun):
		    if tmpbeam.Type != 2:
			print "invalid fit, skip Run "+str(tmpbeam.Run)+" IOV: "+str(tmpbeam.IOVfirst) + " to "+ str(tmpbeam.IOVlast)
		    else:
			listbeam.append(tmpbeam)
	
		tmpbeamsize = 0
		tmpbeam = BeamSpot()
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
		tmpbeam.IOVfirst = line.split()[2]
		tmpbeam.IOVlast = line.split()[4]
		tmpbeamsize += 1
	    if tmpbeamsize == 9:
            #print " from object " + str(tmpbeam.X)
		if int(tmpbeam.IOVfirst) >= int(firstRun) and int(tmpbeam.IOVlast) <= int(lastRun):
		    listbeam.append(tmpbeam)
		tmpbeamsize = 0
		tmpbeam = BeamSpot()
	    
    tmpfile.close()
    print " got total number of IOVs = " + str(len(listbeam)) + " from file " + fileName
    #print " run " + str(listbeam[3].IOVfirst ) + " " + str( listbeam[3].X )
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
            print " iov = 0? skip this IOV = "+ str(ibeam.IOVfirst) + " to " + str(ibeam.IOVlast)
            tmpremovelist.append(ibeam)
        
        if ii < len(listbeam) -1:
            #print listbeam[ii+1].IOVfirst
	    if IOVbase =="lumibase":
		if ibeam.Run == listbeam[ii+1].Run and ibeam.IOVfirst == listbeam[ii+1].IOVfirst:
		    print " duplicate IOV = "+datax+", keep only last duplicate entry"
		    tmpremovelist.append(ibeam)
	    elif datax == listbeam[ii+1].IOVfirst:
                print " duplicate IOV = "+datax+", keep only last duplicate entry"
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
    maxNlumis = 100
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
	    
	    # check last iov
	    if ii < len(listbeam) - 1: 
		inextbeam = listbeam[ii+1]
		docheck = True
		if ii < len(listbeam) -2:
		    iNNbeam = listbeam[ii+2]
	    else:
		print "close payload because end of data has been reached. Run "+ibeam.Run
		docreate = True
            # check we run over the same run
	    if ibeam.Run != inextbeam.Run:
		print "close payload because end of run "+ibeam.Run
		docreate = True
	    # check maximum lumi counts
	    if countlumi == maxNlumis:
		print "close payload because maximum lumi sections accumulated within run "+ibeam.Run
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
		    docreate = True
		    #print "shift here: x="+str(deltaX)+" y="+str(deltaY)
		    #print "x1 = "+ibeam.X + " x1err = "+ibeam.Xerr
		    #print "x2 = "+inextbeam.X + " x2err = "+inextbeam.Xerr
		    #print "Lumi1: "+str(ibeam.IOVfirst) + " Lumi2: "+str(inextbeam.IOVfirst)
		    #print " x= "+ibeam.X+" +/- "+ibeam.Xerr
		    #print "weighted average x = "+tmpbeam.X +" +//- "+tmpbeam.Xerr
		    print "close payload because of movement in X= "+str(deltaX)+", Y= "+str(deltaY) + ", Z= "+str(deltaZ)+", sigmaZ= "+str(deltasigmaZ)+", dxdz= "+str(deltadxdz)+", dydz= "+str(deltadydz)+", widthX= "+str(deltawidthX)+", widthY= "+str(deltawidthY)
	    if docreate:
            #if ii == len(listbeam)-1:
		tmpbeam.IOVlast = ibeam.IOVlast
		tmpbeam.IOVEndTime = ibeam.IOVEndTime
		print "  Run: "+tmpbeam.Run +" Lumi1: "+str(tmpbeam.IOVfirst) + " Lumi2: "+str(tmpbeam.IOVlast)
		newlistbeam.append(tmpbeam)
		tmpbeam = BeamSpot()
	    tmprun = ibeam.Run
	    countlumi += 1


    payloadfile = open(fileName,"w")
    for iload in newlistbeam:
        dump( iload, payloadfile )
    payloadfile.close()

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
                print "close payload because of movement in X= "+str(deltaX)+", Y= "+str(deltaY) + ", Z= "+str(deltaZ)+", sigmaZ= "+str(deltasigmaZ)+", dxdz= "+str(deltadxdz)+", dydz= "+str(deltadydz)+", widthX= "+str(deltawidthX)+", widthY= "+str(deltawidthY)
        
        #WARNING this will only be fine for Run based IOVs
        if ii >= len(listbeam) - 1 or listbeam[ii].Run != listbeam[ii+1].Run :
            print "close payload because end of run has been reached. Run " + listbeam[ii].Run
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
            print "Run: " + tmpbeam.Run + " Lumi1: " + str(tmpbeam.IOVfirst) + " Lumi2: " + str(tmpbeam.IOVlast)
            
    payloadfile = open(fileName,"w")
    for iload in newlistbeam:
        dump( iload, payloadfile )
    payloadfile.close()
