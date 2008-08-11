#!/usr/bin/python
#G.Benelli Jan 22 2007
#A little script to move Simulation Performance Suite
#relevant html and log files into our public area
#/afs/cern.ch/cms/sdt/web/performance/simulation/
#Set here the standard number of events (could become an option... or could be read from the log...)

import dircache as dc
import re, os, sys, time
from stat import *

def getDate():
    return time.ctime()

def getcmd(command):
    return os.popen4(command)[1].read()

TimeSizeNumOfEvents = 100
IgProfNumOfEvents   = 5
ValgrindNumOfEvents = 1
UserWebArea         = sys.argv[1]

#Some nomenclature
Candle=( #These need to match the directory names in the work area
    "HiggsZZ4LM190",
    "MinBias",
    "SingleElectronE1000",
    "SingleMuMinusPt10",
    "SinglePiMinusE1000",
    "TTbar",
    "QCD_80_120"
    )
CmsDriverCandle={ #These need to match the cmsDriver.py output filenames
    Candle[0] : "HZZLLLL_190",
    Candle[1] : "MINBIAS",
    Candle[2] : "E_1000",
    Candle[3] : "MU-_pt_10",
    Candle[4] : "PI-_1000",
    Candle[5] : "TTBAR",
    Candle[6] : "QCD_80_120"
    )
Profile=( #These need to match the profile directory names ending within the candle directories
          "TimingReport",
          "TimeReport",
          "SimpleMemReport",
          "EdmSize",
          "IgProfperf",
          "IgProfMemTotal",
          "IgProfMemLive",
          "IgProfMemAnalyse",
          "valgrind",
          "memcheck_valgrind"
	   )
DirName=( #These need to match the candle directory names ending (depending on the type of profiling)
	  "TimeSize",
	  "IgProf",
	  "Valgrind"
	  )
Step=(
       "GEN,SIM",
       "DIGI",
       "L1",
       "DIGI2RAW",
       "HLT",
       "RAW2DIGI",
       "RECO",
       "DIGI_PILEUP",
       "L1_PILEUP",
       "DIGI2RAW_PILEUP",
       "HLT_PILEUP",
       "RAW2DIGI_PILEUP",
       "RECO_PILEUP"
       )
#This hash and the code that checked for this is obsolete: it was for backward compatibility with cfg version
#of the suite (need to clean it up sometime)
StepLowCaps=(
	      "SIM"  : "sim",
	      "DIGI" : "digi",
	      "RECO" : "reco",
	      "DIGI_PILEUP" : "digi_pileup",
	      "RECO_PILEUP" : "reco_pileup",
	      )
NumOfEvents=( #These numbers are used in the index.html they are not automatically matched to the actual
	       #ones (one should automate this, by looking into the cmsCreateSimPerfTestPyRelVal.log logfile)
	    DirName[0] : TimeSizeNumOfEvents,
	    DirName[1] : IgProfNumOfEvents,
	    DirName[2] : ValgrindNumOfEvents
	    )
OutputHtml=( #These are the filenames to be linked in the index.html page for each profile
	     Profile[0] : "*TimingReport.html", #The wildcard spares the need to know the candle name
	     Profile[1] : "TimeReport.html", #This is always the same (for all candles)
	     Profile[2] : "*.html", #This is supposed to be *SimpleMemoryCheck.html, but there is a bug in cmsRelvalreport.py and it is called TimingReport.html!
	     Profile[3] : "objects_pp.html", #This is one of 4 objects_*.html files, it's indifferent which one to pick, just need consistency
	     Profile[4] : "overall.html", #This is always the same (for all candles)
	     Profile[5] : "overall.html", #This is always the same (for all candles)
	     Profile[6] : "overall.html", #This is always the same (for all candles)
	     Profile[7] : "doBeginJob_output.html", #This is complicated... there are 3 html to link... (see IgProf MemAnalyse below)
	     Profile[8] : "overall.html", #This is always the same (for all candles)
	     Profile[9] : "beginjob.html" #This is complicated there are 3 html to link here too... (see Valgrind MemCheck below)
	     )
IgProfMemAnalyseOut=( #This is the special case of IgProfMemAnalyse
		      "doBeginJob_output.html",
		      "doEvent_output.html",
		      "mem_live.html",
                      "mem_total.html" 
		      )
memcheck_valgrindOut=( #This is the special case of Valgrind MemCheck (published via Giovanni's script)
		       "beginjob.html",
		       "edproduce.html",
		       "esproduce.html"
		       )
#Get the CMSSW_VERSION from the environment

try:
    CMSSW_VERSION=os.environ('CMSSW_VERSION')
    CMSSW_RELEASE_BASE=os.environ('CMSSW_RELEASE_BASE')
    CMSSW_BASE=os.environ('CMSSW_BASE')
    HOST=os.environ('HOST')
    USER=os.environ('USER')
except KeyError:
    print "ERROR: Could not retrieve some necessary environment variables. Have you ran scramv1 runtime -csh yet?"
    sys.exit()

LocalPath=getcmd("pwd")
ShowTagsResult=getcmd("showtags -r")

#Adding a check for a local version of the packages
PerformancePkg="%s/src/Validation/Performance" % CMSSW_BASE
if (os.path.exists(PerformancePkg)):
    BASE_PERFORMANCE=PerformancePkg
    print "**[cmsSimPerfPublish.pl]Using LOCAL version of Validation/Performance instead of the RELEASE version**"
else:
    BASE_PERFORMANCE="%s/src/Validation/Performance" % CMS_RELEASE_BASE

#Define the web publishing area
if (UserWebArea == "simulation"): 
    WebArea="/afs/cern.ch/cms/sdt/web/performance/simulation/%s" % CMSSW_VERSION
    print "Publication web area: %s" % WebArea
elif (UserWebArea == "relval"):
    WebArea="/afs/cern.ch/cms/sdt/web/performance/RelVal/%s" % CMSSW_VERSION
    print "Publication web area: %s" % WebArea
elif (UserWebArea == "local"):
    WebArea="/tmp/%s/%s" % (USER,CMSSW_VERSION);
    print "**User chose to publish results in a local directory**"
    print "Creating local directory %s" % WebArea
    system("mkdir %s" % WebArea);
else:
    print "No publication directory specified!\nPlease choose between simulation, relval or local\nE.g.: cmsSimPerfPublish.pl local"
    sys.exit()

#Dump some info in a file   
Contents = dc.opendir(WebArea) # || die "The area $WebArea does not exist!\nRun the appropriate script to request AFS space first, or wait for it to create it!\n";
#    $CheckDir=`ls $WebArea`;
#    if ($CheckDir eq "")
if (len(Contents) > 0):#@Contents will have only 2 entries . and .. if the dir is empty.
                       #In reality things are more complicated there could be .AFS files...
                       #but it's too much of a particular case to handle it by script
    print "The area %s is ready to be populated!" % WebArea
else:
    print "The area %s already exists!" % WebArea
    sys.exit()

date=getDate()
cmslogreg = re.compile("^cms.*\.log$")
LogFiles  = filter(cmslogreg.search,dc.opendir("."))
print "Found the following log files:"
print LogFiles

cmsscidirreg = re.compile("^cmsScimarkResults_.*/$")
listdir = dc.opendir(".")
dc.annotate("/",listdir)
cmsScimarkDir = filter(cmsscidirreg.search,listdir) #`ls -d cmsScimarkResults_*`;
print "Found the following cmsScimark2 results directories:"
print cmsScimarkDir

htmlreg = re.compile(".*\.html")
cmsScimarkResults = []
for dir in cmsScimarkDir:
    htmlfiles = filter(htmlreg.search,dc.listdir(dir))
    htmlfiles = map(lambda x : dir + "/" + x,htmlfiles)
    cmsScimarkResults.append(htmlfiles)

ExecutionDateSec=0
cmsreg.compile("^cmsCreateSimPerfTest")
for logf in LogFiles:
    chomp(logf);
    if cmsreg.search(logf):
	ExecutionDateLastSec = os.stat(logf)[ST_CTIME]
	ExecutionDateLast    = os.stat(logf)[ST_MTIME]
	print "Execution (completion) date for %s was: %s" % (logf,ExecutionDateLast)
	if (ExecutionDateLastSec > ExecutionDateSec):
	    ExecutionDateSec = ExecutionDateLastSec
	    ExecutionDate    = ExecutionDateLast
     

print "Copying the logfiles to %s/." % WebArea
os.system("cp -pR cms*.log %s/." % WebArea);
print "Copying the cmsScimark2 results to the %s/." % WebArea
os.system("cp -pR cmsScimarkResults_* %s/." % WebArea)

#Copy the perf_style.css file from Validation/Performance/doc
print "Copying %s/doc/perf_style.css style file to %s/." % (BASE_PERFORMANCE,WebArea)

os.system("cp -pR %s/doc/perf_style.css %s/." % (BasePerformance,WebArea))
Dir = os.listdir(".")

#Produce a small logfile with basic info on the Production area
LogFile = "%s/ProductionLog.txt" % WebArea

try:
    LOG = open(LogFile) #||die "CAnnot open file $_!\n$!\n"
    print "Writing Production Host, Location, Release and Tags information in " % LogFile 
    LOG.write("These performance tests were executed on host %s and published on %s" % (HOST,date))
    LOG.write("They were run in %s" % LocalPath)
    LOG.write("Results of showtags -r in the local release:\n%s" % ShowTagsResult)
    LOG.close()
except IOError:
    print "Could not correct create the log file for some reason"

#Produce a "small" index.html file to navigate the html reports/logs etc
IndexFile="%s/index.html" % WebArea
INDEX = open(IndexFile) # ||die "Cannot open file $IndexFile!\n$!\n"
print "Writing an index.html file with links to the profiles report information for easier navigation\n"
TemplateHtml="%s/doc/index.html" % BASE_PERFORMANCE
print "Template used: %s" % TemplateHtml

cmsverreg = re.compile("CMSSW_VERSION")
hostreg   = re.compile("HOST")
lpathreg  = re.compile("LocalPath")
proddreg  = re.compile("ProductionDate")
logfreg   = re.compile("LogfileLinks")
dirbreg   = re.compile("DirectoryBrowsing")
pubdreg   = re.compile("PublicationDate")
candhreg  = re.compile("CandlesHere")
#Loop line by line to build our index.html based on the template one
#TEMPLATE =  #||die "Couldn't open file $TemplateHtml - $!\n"

for NewFileLine in open(TemplateHtml) 
    if cmsverreg.search(NewFileLine):
	INDEX.write(CMSSW_VERSION)
	continue
    if hostreg.search(NewFileLine):
        INDEX.write(HOST)
	continue
    if lpathreg.search(NewFileLine):
        INDEX.write(LocalPath)
        continue
    if proddreg.search(NewFileLine):
	INDEX.write(ExecutionDate)
	continue
    if logfreg.search(NewFileLine):
	INDEX.write("<br><br>")
	for log in LogFiles:
	    #chomp($_)
	    #$LogFileLink="$WebArea/"."$_";
	    INDEX.write("<a href=%s> %s <\/a>" % (log,log))
	    INDEX.write("<br><br>")
	#Add the cmsScimark results here:
	INDEX.write("Results for cmsScimark2 benchmark (running on the other cores) available at:")
        INDEX.write("<br><br>")
	for cmssci in cmsScimarkResults:
	    INDEX.write("<a href=%s> %s <\/a>" % (cmssci,cmssci))
	    INDEX.write("<br><br>")
	continue

    if dirbreg.search(NewFileLine):
	#Create a subdirectory DirectoryBrowsing to circumvent the fact the dir is not browsable if there is an index.html in it.
	os.system("mkdir %s/DirectoryBrowsing" % WebArea)
	INDEX.write("Click <a href=\.\/DirectoryBrowsing\/\.>here<\/a> to browse the directory containing all results (except the root files)")
	continue
    if pubdreg.search(NewFileLine):
        INDEX.write(date)
	continue
    if candhreg.search(NewFileLine):
	for CurrentCandle in Candle:
	    INDEX.write("<table cellpadding=20px border=1><td width=25% bgcolor=#FFFFFE >")
	    INDEX.write("<h2>")
	    INDEX.write(CurrentCandle)
	    INDEX.write("<\/h2>")
	    INDEX.write("<dir style=\"font-size: 13\"> \n <p>")
	    for CurDir in DirName:
                LocalPath="%s_%s" % (CurrentCandle,CurDir)
		CandleLogFiles=getcmd("find %s \-name \"\*.log\"" % LocalPath)
                CandleLogFiles = CandleLogFiles.split("\n")
		if (len(CandleLogFiles)>0):
                    INDEX.write("<br><b>Logfiles for %s<\/b><br>" % CurDir)
		    for cand in CandleLogFiles:
                        print "Found %s in %s\n" (cand,LocalPath)
			os.system("cp -pR %s %s/." % (cand,WebArea))
			INDEX.write("<a href=%s>%s <\/a>" % (cand,cand))
			INDEX.write("<br>")
		
		for CurrentProfile in Profile:
		    for step in Step :
                        ProfileTemplate="%s_%s/*_%s_%s*/%s" % (CurrentCandle,CurDir,step,CurrentProfile,OutputHtml[CurrentProfile])
			#There was the issue of SIM vs sim (same for DIGI) between the previous RelVal based performance suite and the current.
                        ProfileTemplateLowCaps="%s_%s/*_%s_%s*/%s" % (CurrentCandle,CurDir,StepLowCaps[step],CurrentProfile,OutputHtml[CurrentProfile])
			ProfileReportLink=getcmd("ls %s 2>/dev/null" % ProfileTemplate)
			if ( CurrentCandle not in ProfileReportLink) : #no match with caps try low caps
			    ProfileReportLink=getcmd("ls %s 2>/dev/null" % ProfileTemplateLowCaps)
		  
			if (CurrentProfile in ProfileReportLink):#It could also not be there
		     
			    if (PrintedOnce==0): #Making sure it's printed only once per directory (TimeSize, IgProf, Valgrind) each can have multiple profiles
			    	INDEX.write("<br>")
				INDEX.write("<b>%s</b>" % CurDir)#This is the "title" of a series of profiles, (TimeSize, IgProf, Valgrind)
				PrintedOnce=1
			    #Special cases first (IgProf MemAnalyse and Valgrind MemCheck)
			    if (CurrentProfile == Profile[7]):
			    	for i in range(0,3,1):
                                    ProfileTemplate       ="%s_%s/*_%s_%s*/%s" % (CurrentCandle,CurDir,step,CurrentProfile,IgProfMemAnalyseOut[i])
                                    ProfileTemplateLowCaps="%s_%s/*_%s_%s*/%s" % (CurrentCandle,CurDir,StepLowCaps[step],CurrentProfile,IgProfMemAnalyseOut[i])
				    ProfileReportLink= getcmd("ls %s 2>/dev/null" % ProfileTemplate)
				    if ( CurrentCandle not in ProfileReportLink ): # no match with caps try low caps
				    	ProfileReportLink=getcmd("ls %s 2>/dev/null"% ProfileTemplateLowCaps) 
				    if (CurrentProfile in ProfileReportLink ) :#It could also not be there
                                        INDEX.write("<li><a href=%s>%s %s %s (%s events)<\/a>" % ( ProfileReportLink,CurrentProfile,IgProfMemAnalyseOut[i],step,NumOfEvents[CurDir])
				    
				
		       
			    elif (CurrentProfile == Profile[9]):
			    
				for i in range(0,3,1):
                                    ProfileTemplate="%s_%s/*_%s_%s*/%s" % (CurrentCandle,CurDir,step,CurrentProfile,memcheck_valgrindOut[i])
                                    ProfileTemplateLowCaps="%s_%s/*_%s_%s*/%s" % (CurrentCandle,CurDir,StepLowCaps[step],CurrentProfile,memcheck_valgrindOut[i])
                                    ProfileReportLink=getcmd("ls %s 2> /dev/null" % ProfileTemplate)
				    if ( CurrentCandle not in ProfileReportLink ):#no match with caps try low caps
                                        ProfileReportLink=getcmd("ls %s 2> /dev/null" % ProfileTemplateLowCaps)
				    if (CurrentProfile in ProfileReportLink)#It could also not be there
                                        INDEX.write("<li><a href=%s>%s %s %s (%s events)<\/a>" % (ProfileReportLink,CurrentProfile,memcheck_valgrindOut[i],step,NumOfEvents[CurDir])		
			   
			    else
                                INDEX.write("<li><a href=%s>%s %s (%s events)<\/a>" % (ProfileReportLink,CurrentProfile,step,NumOfEvents[CurDir])
			
		    
		
		PrintedOnce=0
	    
	    INDEX.write("<\/p>")
	    INDEX.write("</dir>")
	    INDEX.write("<hr>")
	    INDEX.write("<br>")
	    INDEX.write("<\/td><\/table>")
		    
	continue
    
    INDEX.write(NewFileLine)
    
#End of while loop on template html file

for CurrentDir in Dir:
    for dirn in DirName:
	if (dirn in CurrentDir): # get rid of possible spurious dirs
	    print "Copying %s to %s\n"            % (CurrentDir,WebArea)
	    CopyDir=os.system("cp -pR %s %s/."    % (CurrentDir,WebArea))
	    RemoteDirRootFiles="%s/%s/*.root"     % (WebArea,CurrentDir) 
	    RemoveRootFiles=os.system("rm -Rf %s" % RemoteDirRootFiles)
	

#Creating symbolic links to the web area in subdirectory to allow directory browsing:
DirectoryContent=getcmd("ls %s" % WebArea)
for content in DirectoryContent:
    if ((not (content == "index.html")) and (not(content == "DirectoryBrowsing"))):
	system("ln -s %s/%s %s/DirectoryBrowsing/%s") % (WebArea,Content,WebArea,Content)
  
INDEX.write("\<\/body\>\n")
INDEX.write("\<\/html\>\n")
INDEX.close()



