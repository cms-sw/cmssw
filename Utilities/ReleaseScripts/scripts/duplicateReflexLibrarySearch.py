#! /usr/bin/env python

import optparse
import os
import commands
import re
import sys
import pprint
import commands
import subprocess
from XML2Python import xml2obj

# These aren't all typedefs, but can sometimes make the output more
# readable
typedefsDict = \
             {
    # What we want <=  What we have
    'unsigned int' : ['unsignedint', 'UInt32_t', 'uint32_t'],
    'unsigned long': ['unsignedlong'],
    'int'          : ['Int32_t'],
    'float'        : ['Float_t'],
    'double'       : ['Double_t'],
    'char'         : ['Char_t'],
    '< '           : ['<', '&lt;'],
    ' >'           : ['>', '&gt;'],
    ', '           : [','],
    }


# Equivalent names for packages - lets script know that, for example,
# 'TrackReco' package should have objects 'reco::Track'.
#Ordered List to search for matched packages
equivDict = \
     [
         {'GsfTracking'           : ['reco::GsfTrack(Collection|).*(MomentumConstraint|VertexConstraint)', 'Trajectory.*reco::GsfTrack']},
         {'ParallelAnalysis'      : ['examples::TrackAnalysisAlgorithm']},
         {'PatCandidates'         : ['pat::PATObject','pat::Lepton']},
         {'BTauReco'              : ['reco::SoftLeptonProperties','reco::SecondaryVertexTagInfo']},
         {'CastorReco'            : ['reco::CastorJet']},
         {'JetMatching'           : ['reco::JetFlavour','reco::MatchedPartons']},
         {'TrackingAnalysis'      : ['TrackingParticle']},
         {'Egamma'                : ['reco::ElectronID']},
         {'TopObjects'            : ['reco::CATopJetProperties']},
         {'TauReco'               : ['reco::L2TauIsolationInfo','reco::RecoTauPiZero','reco::BaseTau']},
	 {'ValidationFormats'     : ['PGlobalDigi::.+','PGlobalRecHit::.+']},
         {'TrajectorySeed'        : ['TrajectorySeed']},
         {'TrackCandidate'        : ['TrackCandidate']},
	 {'PatternTools'          : ['MomentumConstraint','VertexConstraint','Trajectory']},
	 {'TrackerRecHit2D'       : ['SiStrip(Matched|)RecHit[12]D','SiTrackerGSRecHit[12]D','SiPixelRecHit']},
	 {'MuonReco'              : ['reco::Muon(Ref|)(Vector|)']},
	 {'MuonSeed'              : ['L3MuonTrajectorySeed']},
	 {'HepMCCandidate'        : ['reco::GenParticle.*']},
	 {'L1Trigger'             : ['l1extra::L1.+Particle']},
	 {'TrackInfo'             : ['reco::TrackingRecHitInfo']},
	 {'EgammaCandidates'      : ['reco::GsfElectron.*','reco::Photon.*']},
	 {'HcalIsolatedTrack'     : ['reco::IsolatedPixelTrackCandidate', 'reco::EcalIsolatedParticleCandidate']},
	 {'HcalRecHit'            : ['HFRecHit','HORecHit','ZDCRecHit','HBHERecHit']},
         {'PFRootEvent'           : ['EventColin::']},
	 {'CaloTowers'            : ['CaloTower.*']},
         {'GsfTrackReco'          : ['GsfTrack.*']},
         {'METReco'               : ['reco::(Calo|PF|Gen|)MET','reco::PFClusterMET']},
         {'ParticleFlowReco'      : ['reco::RecoPFClusterRefCandidateRef.*']},
         {'ParticleFlowCandidate' : ['reco::PFCandidateRef','reco::PFCandidateFwdRef','reco::PFCandidate']},
         {'PhysicsToolsObjects'   : ['PhysicsTools::Calibration']},
         {'RecoCandidate'         : ['reco::Candidate']},
         {'TrackReco'             : ['reco::Track']},
         {'VertexReco'            : ['reco::Vertex']},
         {'TFWLiteSelectorTest'   : ['tfwliteselectortest']},
         {'PatCandidates'         : ['reco::RecoCandidate','pat::[A-Za-z]+Ref(Vector|)']},
         {'JetReco'               : ['reco::.*Jet','reco::.*Jet(Collection|Ref)']},
     ]

ignoreEdmDP = {
  'LCGReflex/__gnu_cxx::__normal_iterator<std::basic_string<char>*,std::vector<std::basic_string<char>%>%>' : 1,
  '' : 1
}

def getReleaseBaseDir ():
    """ return CMSSW_RELEASE_BASE or CMSSW_BASE depending on the
    dev area of release area """
    baseDir = os.environ.get('CMSSW_RELEASE_BASE')
    if not len (baseDir):
        baseDir = os.environ.get('CMSSW_BASE')
    return baseDir


def searchClassDefXml (srcDir):
    """ Searches through the requested directory looking at
    'classes_def.xml' files looking for duplicate Reflex definitions."""
    # compile necessary RE statements
    classNameRE    = re.compile (r'class\s+name\s*=\s*"([^"]*)"')
    spacesRE       = re.compile (r'\s+')
    stdRE          = re.compile (r'std::')
    srcClassNameRE = re.compile (r'(\w+)/src/classes_def.xml')
    ignoreSrcRE    = re.compile (r'.*/FWCore/Skeletons/scripts/mkTemplates/.+')
    braketRE       = re.compile (r'<.+>')
    # get the source directory we want
    if not len (srcDir):
        try:
            srcDir = getReleaseBaseDir() + '/src'
        except:
            raise RuntimeError, "$CMSSW_RELEASE_BASE not found."
    try:
        os.chdir (srcDir)
    except:
        raise RuntimeError, "'%s' is not a valid directory." % srcDir
    print "Searching for 'classes_def.xml' in '%s'." % srcDir
    xmlFiles = commands.getoutput ('find . -name "*classes_def.xml" -print').\
               split ('\n')
    # print out the XML files, if requested
    if options.showXMLs:
        pprint.pprint (xmlFiles)
    # try and figure out the names of the packages
    xmlPackages = []
    packagesREs = {}
    equivREs    = {}
    explicitREs = []
    for item in equivDict:
        for pack in item:
            for equiv in item[pack]:
                explicitREs.append( (re.compile(r'\b' + equiv + r'\b'),pack))
    if options.lostDefs:
        for filename in xmlFiles:
            if (not filename) or (ignoreSrcRE.match(filename)): continue
            match = srcClassNameRE.search (filename)
            if not match: continue
            packageName = match.group(1)
            xmlPackages.append (packageName)
            matchString = r'\b' + packageName + r'\b'
            packagesREs[packageName] = re.compile (matchString)
            equivList = equivREs.setdefault (packageName, [])
            for item in equivDict:
                for equiv in item.get (packageName, []):
                    matchString = re.compile(r'\b' + equiv + r'\b')
                    equivList.append( (matchString, equiv) )
            equivList.append( (packagesREs[packageName], packageName) )
    #pprint.pprint (equivREs, width=109)
    classDict = {}
    ncdict = {'class' : 'className'}
    for filename in xmlFiles:
        if (not filename) or (ignoreSrcRE.match(filename)): continue
        dupProblems     = ''
        exceptName      = ''
        regexList       = []
        localObjects    = []
        simpleObjectREs = []
        if options.lostDefs:
            lostMatch = srcClassNameRE.search (filename)
	    if lostMatch:
                exceptName = lostMatch.group (1)
                regexList = equivREs[exceptName]
                xcount = len(regexList)-1
                if not regexList[xcount][0].search (exceptName):
                    print '%s not found in' % exceptName,
                    print regexList[xcount][0]
                    sys.exit()
            else: continue
	if options.verbose:
            print "filename", filename
        try:
            xmlObj = xml2obj (filename = filename,
                              filtering = True,
                              nameChangeDict = ncdict)
        except Exception as detail:
            print "File %s is malformed XML.  Please fix." % filename
            print "  ", detail
            continue
        try:
            classList = xmlObj.selection.className
        except:
            try:
                classList = xmlObj.className
            except:
                # this isn't a real classes_def.xml file.  Skip it
                print "**** SKIPPING '%s' - Doesn't seem to have proper information." % filename
                continue
        for piece in classList:
            try:
                className = spacesRE.sub ('', piece.name)
            except:
                # must be one of these class pattern things.  Skip it
                #print "     skipping %s" % filename, piece.__repr__()
                continue
            className = stdRE.sub    ('', className)
            # print "  ", className
            # Now get rid of any typedefs
            for typedef, tdList in typedefsDict.iteritems():
                for alias in tdList:
                    className = re.sub (alias, typedef, className)
            classDict.setdefault (className, set()).add (filename)
            # should we check for lost definitions?
            if not options.lostDefs:
                continue
            localObjects.append (className)
            if options.lazyLostDefs and not braketRE.search (className):
                #print "  ", className
                matchString = r'\b' + className + r'\b'
                simpleObjectREs.append( (re.compile (matchString), className ) )
        for className in localObjects:
            # if we see our name (or equivalent) here, then let's
            # skip complaining about this
            foundEquiv = False
            for equivRE in regexList:
                #print "searching %s for %s" % (equivRE[1], className)
                if equivRE[0].search (className):
                    foundEquiv = True
                    break
            for simpleRE in simpleObjectREs:
                if simpleRE[0].search (className):
                    foundEquiv = True
                    if options.verbose and simpleRE[1] != className:
                        print "    Using %s to ignore %s" \
                              % (simpleRE[1], className)                    
                    break
            if foundEquiv: continue
            for exRes in explicitREs:
                if exRes[0].search(className):
                    dupProblems += "  %s : %s\n" % (exRes[1], className)
                    foundEquiv = True
                    break
            if foundEquiv: continue
            for packageName in xmlPackages:
                # don't bother looking for the name of this
                # package in this package
                if packagesREs[packageName].search (className):
                    dupProblems += "  %s : %s\n" % (packageName, className)
                    break
        # for piece
        if dupProblems:
            print '\n%s\n%s\n' % (filename, dupProblems)
    # for filename
    if options.dups:
        for name, fileSet in sorted( classDict.iteritems() ):
            if len (fileSet) < 2:
                continue
            print name
            fileList = list (fileSet)
            fileList.sort()
            for filename in fileList:
                print "  ", filename
            print
        # for name, fileSet
    # if not noDups
    #pprint.pprint (classDict)


def searchDuplicatePlugins (edmpluginFile):
    """ Searches the edmpluginFile to find any duplicate
    plugins."""
    cmd = "cat %s | awk '{print $2\" \"$1}' | sort | uniq | awk '{print $1}' | sort | uniq -c | grep '2 ' | awk '{print $2}'" % edmpluginFile
    output = commands.getoutput (cmd).split('\n')
    for line in output:
      if ignoreEdmDP.has_key(line): continue
      line = line.replace("*","\*")
      cmd = "cat %s | grep ' %s ' | awk '{print $1}' | sort | uniq " % (edmpluginFile,line)
      out1 = commands.getoutput (cmd).split('\n')
      print line
      for plugin in out1:
        if plugin:
            print "   **"+plugin+"**"
      print

def searchEdmPluginDump (edmpluginFile, srcDir):
    """ Searches the edmpluginFile to find any duplicate Reflex
    definitions."""
    if not len (edmpluginFile):
        try:
            edmpluginFile = getReleaseBaseDir() + '/lib/' + \
                            os.environ.get('SCRAM_ARCH') + '/.edmplugincache'
        except:
            raise RuntimeError,  \
                  "$CMSSW_RELEASE_BASE or $SCRAM_ARCH not found."
    if not len (srcDir):
        try:
            srcDir = getReleaseBaseDir() + '/src'
        except:
            raise RuntimeError, "$CMSSW_RELEASE_BASE not found."
    try:
        os.chdir (srcDir)
    except:
        raise RuntimeError, "'%s' is not a valid directory." % srcDir
    searchDuplicatePlugins (edmpluginFile)
    packageNames = commands.getoutput ('ls -1').split ('\n')
    global packageREs
    #print "pN", packageNames
    for package in packageNames:
        packageREs.append( re.compile( r'^(' + package + r')(\S+)$') )
    # read the pipe of the grep command
    prevLine = ''
    searchREs = [];
    doSearch = False
    if options.searchFor:
        fixSpacesRE = re.compile (r'\s+');
        doSearch = True
        words = options.searchFor.split('|')
        #print "words:", words
        for word in words:
            word = fixSpacesRE.sub (r'.*', word);
            searchREs.append( re.compile (word) )
    problemSet = set()
    cmd = "grep Reflex %s | awk '{print $2}' | sort" % edmpluginFile
    for line in commands.getoutput (cmd).split('\n'):
        if doSearch:
            for regex in searchREs:
                if regex.search (line):
                    problemSet.add (line)
                    break
        else:
            if line == prevLine:
                if not ignoreEdmDP.has_key(line):
                    problemSet.add (line)
            # print line
            prevLine = line
    # Look up in which libraries the problems are found
    pluginCapRE = re.compile (r'plugin(\S+?)Capabilities.so')
    fixStarsRE  = re.compile (r'\*')
    lcgReflexRE = re.compile (r'^LCGReflex/')
    percentRE   = re.compile (r'%')
    problemList = sorted (list (problemSet))    
    for problem in problemList:
        # Unbackwhacked stars will mess with the grep command.  So
        # let's fix them now and not worry about it
        fixedProblem = fixStarsRE.sub (r'\*', problem)
        cmd = 'grep "%s" %s | awk \'{print $1}\'' % (fixedProblem,
                                                     edmpluginFile)
        # print 'cmd', cmd
        output = commands.getoutput (cmd).split('\n')
        problem = lcgReflexRE.sub (r'', problem)
        problem = percentRE.sub   (r' ', problem)
        print problem
        #if doSearch: continue
        for line in output:
            match = pluginCapRE.match (line)
            if match:                          
                line = match.group(1)
            print "  ", getXmlName (line)
        print

def getXmlName (line):
    """Given a line from EDM plugin dump, try to get XML file name."""
    global packageMatchDict
    retval = packageMatchDict.get (line)
    if retval:
        return retval
    for regex in packageREs:
        match = regex.search (line)
        if match:
            xmlFile = "./%s/%s/src/classes_def.xml" % \
                      (match.group(1), match.group(2))
            if os.path.exists (xmlFile):
                packageMatchDict [line] = xmlFile
                return xmlFile
            #return "**%s/%s**" % (match.group(1), match.group(2)) If
    # we're here, then we haven't been successful yet.  Let's try the
    # brute force approach.
    # Try 1
    cmd = 'find . -name classes_def.xml -print | grep %s' % line
    output = commands.getoutput (cmd).split ('\n')
    if output and len (output) == 1:
        retval = output[0];
        if retval:
            packageMatchDict [line] = retval
            return retval
    # Try 2
    cmd = 'find . -name "BuildFile" -exec grep -q %s {} \; -print' % line
    output = commands.getoutput (cmd).split ('\n')
    if output and len (output) == 1:
        retval = output[0];
        if retval:
            retval = retval + ' (%s)' % line
            packageMatchDict [line] = retval
            return retval
    return "**" +  line + "**"
    


packageREs = [];
packageMatchDict = {}

if __name__ == "__main__":
    # setup options parser
    parser = optparse.OptionParser ("Usage: %prog [options]\n"\
                                    "Searches classes_def.xml for duplicate "\
                                    "definitions")
    xmlGroup  = optparse.OptionGroup (parser, "ClassDef XML options")
    dumpGroup = optparse.OptionGroup (parser, "EdmPluginDump options")
    xmlGroup.add_option ('--dups', dest='dups', action='store_true',
                         default=False,
                         help="Search for duplicate definitions")
    xmlGroup.add_option ('--lostDefs', dest='lostDefs', action='store_true',
                         default=False,
                         help="Looks for definitions in the wrong libraries")
    xmlGroup.add_option ('--lazyLostDefs', dest='lazyLostDefs',
                         action='store_true',
                         default=False,
                         help="Will try to ignore as many lost defs as reasonable")
    xmlGroup.add_option ('--verbose', dest='verbose',
                         action='store_true',
                         default=False,
                         help="Prints out a lot of information")
    xmlGroup.add_option ('--showXMLs', dest='showXMLs', action='store_true',
                         default=False,
                         help="Shows all 'classes_def.xml' files")
    xmlGroup.add_option ('--dir', dest='srcdir', type='string', default='',
                         help="directory to search for 'classes_def.xml'"\
                         " files (default: $CMSSW_RELEASE_BASE/src)")
    dumpGroup.add_option ('--edmPD', dest='edmPD', action='store_true',
                          default=False,
                          help="Searches EDM Plugin Dump for duplicates")
    dumpGroup.add_option ('--edmFile', dest='edmFile', type='string',
                          default='',
                          help="EDM Plugin Dump cache file'"\
                          " (default: $CMSSW_RELEASE_BASE/lib/"\
                          "$SCRAM_ARCH/.edmplugincache)")
    dumpGroup.add_option ('--searchFor', dest='searchFor', type='string',
                          default='',
                          help="Search EPD for given pipe-separated (|) regexs"
                          " instead of duplicates")
    parser.add_option_group (xmlGroup)
    parser.add_option_group (dumpGroup)
    (options, args) = parser.parse_args()

    # Let's go:
    if options.lazyLostDefs:
        options.lostDefs = True
    if options.showXMLs or options.lostDefs or options.dups:
        searchClassDefXml (options.srcdir)
    if options.edmPD:
        searchEdmPluginDump (options.edmFile, options.srcdir)
