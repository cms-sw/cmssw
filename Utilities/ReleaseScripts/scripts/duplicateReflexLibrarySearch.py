#! /usr/bin/env python3

from __future__ import print_function
import optparse
import os
import re
import sys
import pprint
import subprocess
from XML2Python import xml2obj
import six
try:
  from subprocess import getoutput
except:
  from commands import getoutput
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
         {'Associations': ['TTTrackTruthPair', 'edm::Wrapper.+edm::AssociationMap.+TrackingParticle']},
         {'TrajectoryState'         : ['TrajectoryStateOnSurface']},
         {'TrackTriggerAssociation' : ['(TTClusterAssociationMap|TTStubAssociationMap|TTTrackAssociationMap|TrackingParticle).*Phase2TrackerDigi',
                                       '(TTStub|TTCluster|TTTrack).*Phase2TrackerDigi.*TrackingParticle']},
         {'L1TrackTrigger'        : ['(TTStub|TTCluster|TTTrack).*Phase2TrackerDigi']},
         {'L1TCalorimeterPhase2'  : ['l1tp2::CaloTower.*']},
         {'L1TCalorimeter'        : ['l1t::CaloTower.*']},
         {'VertexFinder'          : ['l1tVertexFinder::Vertex']},
         {'GsfTracking'           : ['reco::GsfTrack(Collection|).*(MomentumConstraint|VertexConstraint)', 'Trajectory.*reco::GsfTrack']},
         {'PatCandidates'         : ['pat::PATObject','pat::Lepton', 'reco::RecoCandidate','pat::[A-Za-z]+Ref(Vector|)', 'pat::UserHolder']},
         {'BTauReco'              : ['reco::.*SoftLeptonTagInfo', 'reco::SoftLeptonProperties','reco::SecondaryVertexTagInfo','reco::IPTagInfo','reco::TemplatedSecondaryVertexTagInfo', 'reco::CATopJetProperties','reco::HTTTopJetProperties']},
         {'CastorReco'            : ['reco::CastorJet']},
         {'JetMatching'           : ['reco::JetFlavourInfo', 'reco::JetFlavour','reco::MatchedPartons']},
         {'RecoCandidate'         : ['reco::Candidate']},
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
         {'L1Trigger'             : ['l1extra::L1.+Particle', 'l1t::Vertex']},
         {'TrackInfo'             : ['reco::TrackingRecHitInfo']},
         {'EgammaCandidates'      : ['reco::GsfElectron.*','reco::Photon.*']},
         {'HcalIsolatedTrack'     : ['reco::IsolatedPixelTrackCandidate', 'reco::EcalIsolatedParticleCandidate', 'reco::HcalIsolatedTrackCandidate']},
         {'HcalRecHit'            : ['HFRecHit','HORecHit','ZDCRecHit','HBHERecHit']},
         {'PFRootEvent'           : ['EventColin::']},
         {'CaloTowers'            : ['CaloTower.*']},
         {'GsfTrackReco'          : ['GsfTrack.*']},
         {'METReco'               : ['reco::(Calo|PF|Gen|)MET','reco::PFClusterMET']},
         {'ParticleFlowReco'      : ['reco::RecoPFClusterRefCandidateRef.*']},
         {'ParticleFlowCandidate' : ['reco::PFCandidateRef','reco::PFCandidateFwdRef','reco::PFCandidate']},
         {'PhysicsToolsObjects'   : ['PhysicsTools::Calibration']},
         {'TrackReco'             : ['reco::Track','reco::TrackRef']},
         {'VertexReco'            : ['reco::Vertex']},
         {'TFWLiteSelectorTest'   : ['tfwliteselectortest']},
         {'TauReco'               : ['reco::PFJetRef']},
         {'JetReco'               : ['reco::.*Jet','reco::.*Jet(Collection|Ref)']},
         {'HGCDigi'               : ['HGCSample']},
         {'HGCRecHit'             : ['constHGCRecHit','HGCRecHit']},
         {'SiPixelObjects'        : ['SiPixelQuality.*']},
     ]

ignoreEdmDP = {
  'LCGReflex/__gnu_cxx::__normal_iterator<std::basic_string<char>*,std::vector<std::basic_string<char>%>%>' : 1,
  '' : 1
}

def searchClassDefXml ():
    """ Searches through the requested directory looking at
    'classes_def.xml' files looking for duplicate Reflex definitions."""
    # compile necessary RE statements
    classNameRE    = re.compile (r'class\s+name\s*=\s*"([^"]*)"')
    spacesRE       = re.compile (r'\s+')
    stdRE          = re.compile (r'std::')
    srcClassNameRE = re.compile (r'(\w+)/src/classes_def.*[.]xml')
    ignoreSrcRE    = re.compile (r'.*/FWCore/Skeletons/scripts/mkTemplates/.+')
    braketRE       = re.compile (r'<.+>')
    print("Searching for 'classes_def.xml' in '%s'." % os.path.join(os.environ.get('CMSSW_BASE'),'src'))
    xmlFiles = []
    for srcDir in [os.environ.get('CMSSW_BASE'),os.environ.get('CMSSW_RELEASE_BASE')]:
        if not len(srcDir): continue
        for xml in getoutput ('cd '+os.path.join(srcDir,'src')+'; find . -name "*classes_def*.xml" -follow -print').split ('\n'):
            if xml and (not xml in xmlFiles):
                xmlFiles.append(xml)
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
    classDict = {}
    ncdict = {'class' : 'className', 'function' : 'functionName'}
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
                    print('%s not found in' % exceptName, end=' ')
                    print(regexList[xcount][0])
                    sys.exit()
            else: continue
        if options.verbose:
            print("filename", filename)
        try:
            filepath = os.path.join(os.environ.get('CMSSW_BASE'),'src',filename)
            if not os.path.exists(filepath):
                filepath = os.path.join(os.environ.get('CMSSW_RELEASE_BASE'),'src',filename)
            xmlObj = xml2obj (filename = filepath,
                              filtering = True,
                              nameChangeDict = ncdict)
        except Exception as detail:
            print("File %s is malformed XML.  Please fix." % filename)
            print("  ", detail)
            continue
        try:
            classList = xmlObj.selection.className
        except:
            try:
                classList = xmlObj.className
            except:
                # this isn't a real classes_def.xml file.  Skip it
                print("**** SKIPPING '%s' - Doesn't seem to have proper information." % filename)
                continue
        if not classList:
            classList = xmlObj.functionName
            if not classList:
                print("**** SKIPPING '%s' - Dosen't seem to have proper information(not class/function)." % filename)
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
            for typedef, tdList in six.iteritems(typedefsDict):
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
                #print("searching %s for %s" % (equivRE[1], className))
                if equivRE[0].search (className):
                    foundEquiv = True
                    break
            for simpleRE in simpleObjectREs:
                if simpleRE[0].search (className):
                    foundEquiv = True
                    if options.verbose and simpleRE[1] != className:
                        print("    Using %s to ignore %s" \
                              % (simpleRE[1], className))                    
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
            print('\n%s\n%s\n' % (filename, dupProblems))
    # for filename
    if options.dups:
        for name, fileSet in sorted( six.iteritems(classDict) ):
            if len (fileSet) < 2:
                continue
            print(name)
            fileList = sorted (fileSet)
            for filename in fileList:
                print("  ", filename)
            print()
        # for name, fileSet
    # if not noDups
    #pprint.pprint (classDict)


def searchDuplicatePlugins ():
    """ Searches the edmpluginFile to find any duplicate
    plugins."""
    edmpluginFile = ''
    libenv = 'LD_LIBRARY_PATH'
    if os.environ.get('SCRAM_ARCH').startswith('osx'): libenv = 'DYLD_FALLBACK_LIBRARY_PATH'
    biglib = '/biglib/'+os.environ.get('SCRAM_ARCH')
    for libdir in os.environ.get(libenv).split(':'):
        if libdir.endswith(biglib): continue
        if os.path.exists(libdir+'/.edmplugincache'): edmpluginFile = edmpluginFile + ' ' + libdir+'/.edmplugincache'
    if edmpluginFile == '': edmpluginFile = os.path.join(os.environ.get('CMSSW_BASE'),'lib',os.environ.get('SCRAM_ARCH'),'.edmplugincache')
    cmd = "cat %s | awk '{print $2\" \"$1}' | sort | uniq | awk '{print $1}' | sort | uniq -c | grep '2 ' | awk '{print $2}'" % edmpluginFile
    output = getoutput (cmd).split('\n')
    for line in output:
        if line in ignoreEdmDP: continue
        line = line.replace("*","\*")
        cmd = "cat %s | grep ' %s ' | awk '{print $1}' | sort | uniq " % (edmpluginFile,line)
        out1 = getoutput (cmd).split('\n')
        print(line)
        for plugin in out1:
            if plugin:
                print("   **"+plugin+"**")
        print()

if __name__ == "__main__":
    # setup options parser
    parser = optparse.OptionParser ("Usage: %prog [options]\n"\
                                    "Searches classes_def.xml for wrong/duplicate "\
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
                         help="Obsolete")
    dumpGroup.add_option ('--edmPD', dest='edmPD', action='store_true',
                          default=False,
                          help="Searches EDM Plugin Dump for duplicates")
    dumpGroup.add_option ('--edmFile', dest='edmFile', type='string',
                          default='',
                          help="Obsolete")
    parser.add_option_group (xmlGroup)
    parser.add_option_group (dumpGroup)
    (options, args) = parser.parse_args()

    # Let's go:
    if options.lazyLostDefs:
        options.lostDefs = True
    if options.showXMLs or options.lostDefs or options.dups:
        searchClassDefXml ()
    if options.edmPD:
        searchDuplicatePlugins ()
