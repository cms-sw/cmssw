import os
import re
import sys
import shutil
import subprocess

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True

import plotting

# Mapping from releases to GlobalTags
_globalTags = {
    "CMSSW_6_2_0": {"default": "PRE_ST62_V8"},
    "CMSSW_6_2_0_SLHC15": {"UPG2019withGEM": "DES19_62_V8", "UPG2023SHNoTaper": "DES23_62_V1"},
    "CMSSW_6_2_0_SLHC17": {"UPG2019withGEM": "DES19_62_V8", "UPG2023SHNoTaper": "DES23_62_V1"},
    "CMSSW_6_2_0_SLHC20": {"UPG2019withGEM": "DES19_62_V8", "UPG2023SHNoTaper": "DES23_62_V1"},
    "CMSSW_7_0_0": {"default": "POSTLS170_V3", "fullsim_50ns": "POSTLS170_V4"},
    "CMSSW_7_0_0_AlcaCSA14": {"default": "POSTLS170_V5_AlcaCSA14", "fullsim_50ns": "POSTLS170_V6_AlcaCSA14"},
    "CMSSW_7_0_7_pmx": {"default": "PLS170_V7AN1", "fullsim_50ns": "PLS170_V6AN1"},
    "CMSSW_7_0_9_patch3": {"default": "PLS170_V7AN2", "fullsim_50ns": "PLS170_V6AN2"},
    "CMSSW_7_0_9_patch3_Premix": {"default": "PLS170_V7AN2", "fullsim_50ns": "PLS170_V6AN2"},
    "CMSSW_7_1_0": {"default": "POSTLS171_V15", "fullsim_50ns": "POSTLS171_V16"},
    "CMSSW_7_1_9": {"default": "POSTLS171_V17", "fullsim_50ns": "POSTLS171_V18"},
    "CMSSW_7_1_9_patch2": {"default": "POSTLS171_V17", "fullsim_50ns": "POSTLS171_V18"},
    "CMSSW_7_1_10_patch2": {"default": "MCRUN2_71_V1", "fullsim_50ns": "MCRUN2_71_V0"},
    "CMSSW_7_2_0_pre5": {"default": "POSTLS172_V3", "fullsim_50ns": "POSTLS172_V4"},
    "CMSSW_7_2_0_pre7": {"default": "PRE_LS172_V11", "fullsim_50ns": "PRE_LS172_V12"},
    "CMSSW_7_2_0_pre8": {"default": "PRE_LS172_V15", "fullsim_50ns": "PRE_LS172_V16"},
    "CMSSW_7_2_0": {"default": "PRE_LS172_V15", "fullsim_50ns": "PRE_LS172_V16"},
    "CMSSW_7_2_0_PHYS14": {"default": "PHYS14_25_V1_Phys14"},
    "CMSSW_7_2_2_patch1": {"default": "MCRUN2_72_V1", "fullsim_50ns": "MCRUN2_72_V0"},
    "CMSSW_7_2_2_patch1_Fall14DR": {"default": "MCRUN2_72_V3_71XGENSIM"},
#    "CMSSW_7_3_0_pre1": {"default": "PRE_LS172_V15", "fullsim_25ns": "PRE_LS172_V15_OldPU", "fullsim_50ns": "PRE_LS172_V16_OldPU"},
    "CMSSW_7_3_0_pre1": {"default": "PRE_LS172_V15", "fullsim_50ns": "PRE_LS172_V16"},
#    "CMSSW_7_3_0_pre2": {"default": "MCRUN2_73_V1_OldPU", "fullsim_50ns": "MCRUN2_73_V0_OldPU"},
    "CMSSW_7_3_0_pre2": {"default": "MCRUN2_73_V1", "fullsim_50ns": "MCRUN2_73_V0"},
    "CMSSW_7_3_0_pre3": {"default": "MCRUN2_73_V5", "fullsim_50ns": "MCRUN2_73_V4"},
    "CMSSW_7_3_0": {"default": "MCRUN2_73_V7", "fullsim_50ns": "MCRUN2_73_V6"},
    "CMSSW_7_3_0_71XGENSIM": {"default": "MCRUN2_73_V7_71XGENSIM"},
    "CMSSW_7_3_0_71XGENSIM_FIXGT": {"default": "MCRUN2_73_V9_71XGENSIM_FIXGT"},
    "CMSSW_7_3_1_patch1": {"default": "MCRUN2_73_V9", "fastsim": "MCRUN2_73_V7"},
    "CMSSW_7_3_1_patch1_GenSim_7113": {"default": "MCRUN2_73_V9_GenSim_7113"},
    "CMSSW_7_3_3": {"default": "MCRUN2_73_V11", "fullsim_50ns": "MCRUN2_73_V10", "fastsim": "MCRUN2_73_V13"},
    "CMSSW_7_4_0_pre1": {"default": "MCRUN2_73_V5", "fullsim_50ns": "MCRUN2_73_V4"},
    "CMSSW_7_4_0_pre2": {"default": "MCRUN2_73_V7", "fullsim_50ns": "MCRUN2_73_V6"},
    "CMSSW_7_4_0_pre2_73XGENSIM": {"default": "MCRUN2_73_V7_73XGENSIM_Pythia6", "fullsim_50ns": "MCRUN2_73_V6_73XGENSIM_Pythia6"},
    "CMSSW_7_4_0_pre5": {"default": "MCRUN2_73_V7", "fullsim_50ns": "MCRUN2_73_V6"},
    "CMSSW_7_4_0_pre5_BS": {"default": "MCRUN2_73_V9_postLS1beamspot", "fullsim_50ns": "MCRUN2_73_V8_postLS1beamspot"},
    "CMSSW_7_4_0_pre6": {"default": "MCRUN2_74_V1", "fullsim_50ns": "MCRUN2_74_V0"},
    "CMSSW_7_4_0_pre6_pmx": {"default": "MCRUN2_74_V1", "fullsim_50ns": "MCRUN2_74_V0"},
    "CMSSW_7_4_0_pre8": {"default": "MCRUN2_74_V7", "fullsim_25ns": "MCRUN2_74_V5_AsympMinGT", "fullsim_50ns": "MCRUN2_74_V4_StartupMinGT"},
    "CMSSW_7_4_0_pre8_minimal": {"default": "MCRUN2_74_V5_MinGT", "fullsim_25ns": "MCRUN2_74_V5_AsympMinGT", "fullsim_50ns": "MCRUN2_74_V4_StartupMinGT"},
    "CMSSW_7_4_0_pre8_25ns_asymptotic": {"default": "MCRUN2_74_V7"},
    "CMSSW_7_4_0_pre8_50ns_startup":    {"default": "MCRUN2_74_V6"},
    "CMSSW_7_4_0_pre8_50ns_asympref":   {"default": "MCRUN2_74_V5A_AsympMinGT"}, # for reference of 50ns asymptotic
    "CMSSW_7_4_0_pre8_50ns_asymptotic": {"default": "MCRUN2_74_V7A_AsympGT"},
    "CMSSW_7_4_0_pre8_ROOT6": {"default": "MCRUN2_74_V7"},
    "CMSSW_7_4_0_pre8_pmx": {"default": "MCRUN2_74_V7", "fullsim_50ns": "MCRUN2_74_V6"},
    "CMSSW_7_4_0_pre8_pmx_v2": {"default": "MCRUN2_74_V7_gs_pre7", "fullsim_50ns": "MCRUN2_74_V6_gs_pre7"},
    "CMSSW_7_4_0_pre8_pmx_v3": {"default": "MCRUN2_74_V7_bis", "fullsim_50ns": "MCRUN2_74_V6_bis"},
    "CMSSW_7_4_0_pre9": {"default": "MCRUN2_74_V7", "fullsim_50ns": "MCRUN2_74_V6"},
    "CMSSW_7_4_0_pre9_ROOT6": {"default": "MCRUN2_74_V7", "fullsim_50ns": "MCRUN2_74_V6"},
    "CMSSW_7_4_0_pre9_ROOT6_pmx": {"default": "MCRUN2_74_V7", "fullsim_50ns": "MCRUN2_74_V6"},
    "CMSSW_7_4_0_pre9_extended": {"default": "MCRUN2_74_V7_extended"},
    "CMSSW_7_4_0": {"default": "MCRUN2_74_V7_gensim_740pre7", "fullsim_50ns": "MCRUN2_74_V6_gensim_740pre7", "fastsim": "MCRUN2_74_V7"},
    "CMSSW_7_4_0_71XGENSIM": {"default": "MCRUN2_74_V7_GENSIM_7_1_15", "fullsim_50ns": "MCRUN2_74_V6_GENSIM_7_1_15"},
    "CMSSW_7_4_0_71XGENSIM_PU": {"default": "MCRUN2_74_V7_gs7115_puProd", "fullsim_50ns": "MCRUN2_74_V6_gs7115_puProd"},
    "CMSSW_7_4_0_71XGENSIM_PXworst": {"default": "MCRUN2_74_V7C_pxWorst_gs7115", "fullsim_50ns": "MCRUN2_74_V6A_pxWorst_gs7115"},
    "CMSSW_7_4_0_71XGENSIM_PXbest": {"default": "MCRUN2_74_V7D_pxBest_gs7115", "fullsim_50ns": "MCRUN2_74_V6B_pxBest_gs7115"},
    "CMSSW_7_4_0_pmx": {"default": "MCRUN2_74_V7", "fullsim_50ns": "MCRUN2_74_V6"},
    "CMSSW_7_4_1": {"default": "MCRUN2_74_V9_gensim_740pre7", "fullsim_50ns": "MCRUN2_74_V8_gensim_740pre7", "fastsim": "MCRUN2_74_V9"},
    "CMSSW_7_4_1_71XGENSIM": {"default": "MCRUN2_74_V9_gensim71X", "fullsim_50ns": "MCRUN2_74_V8_gensim71X"},
    "CMSSW_7_4_1_extended": {"default": "MCRUN2_74_V9_extended"},
    "CMSSW_7_4_3": {"default": "MCRUN2_74_V9", "fullsim_50ns": "MCRUN2_74_V8", "fastsim": "MCRUN2_74_V9", "fastsim_25ns": "MCRUN2_74_V9_fixMem"},
    "CMSSW_7_4_3_extended": {"default": "MCRUN2_74_V9_ext","fastsim": "MCRUN2_74_V9_fixMem"},
    "CMSSW_7_4_3_pmx": {"default": "MCRUN2_74_V9_ext", "fullsim_50ns": "MCRUN2_74_V8", "fastsim": "MCRUN2_74_V9_fixMem"},
    "CMSSW_7_4_3_patch1_unsch": {"default": "MCRUN2_74_V9_unsch", "fullsim_50ns": "MCRUN2_74_V8_unsch"},
    "CMSSW_7_4_4": {"default": "MCRUN2_74_V9_38Tbis", "fullsim_50ns": "MCRUN2_74_V8_38Tbis"},
    "CMSSW_7_4_4_0T": {"default": "MCRUN2_740TV1_0Tv2", "fullsim_50ns": "MCRUN2_740TV0_0TV2", "fullsim_25ns": "MCRUN2_740TV1_0TV2"},
    "CMSSW_7_5_0_pre1": {"default": "MCRUN2_74_V7", "fullsim_50ns": "MCRUN2_74_V6"},
    "CMSSW_7_5_0_pre2": {"default": "MCRUN2_74_V7", "fullsim_50ns": "MCRUN2_74_V6"},
    "CMSSW_7_5_0_pre3": {"default": "MCRUN2_74_V7", "fullsim_50ns": "MCRUN2_74_V6"},
    "CMSSW_7_5_0_pre3_pmx": {"default": "MCRUN2_74_V7", "fullsim_50ns": "MCRUN2_74_V6"},
    "CMSSW_7_5_0_pre4": {"default": "MCRUN2_75_V1", "fullsim_50ns": "MCRUN2_75_V0"},
    "CMSSW_7_5_0_pre4_pmx": {"default": "MCRUN2_75_V1", "fullsim_50ns": "MCRUN2_75_V0"},
}

_releasePostfixes = ["_AlcaCSA14", "_PHYS14", "_TEST", "_pmx_v2", "_pmx_v3", "_pmx", "_Fall14DR", "_71XGENSIM_FIXGT", "_71XGENSIM_PU", "_71XGENSIM_PXbest", "_71XGENSIM_PXworst", "_71XGENSIM", "_73XGENSIM", "_BS", "_GenSim_7113", "_extended",
                     "_25ns_asymptotic", "_50ns_startup", "_50ns_asympref", "_50ns_asymptotic", "_minimal", "_0T", "_unsch"]
def _stripRelease(release):
    for pf in _releasePostfixes:
        if pf in release:
            return release.replace(pf, "")
    return release


def _getGlobalTag(sample, release):
    """Get a GlobalTag.

    Arguments:
    sample  -- Sample object
    release -- CMSSW release string
    """
    if not release in _globalTags:
        print "Release %s not found from globaltag map in validation.py" % release
        sys.exit(1)
    gtmap = _globalTags[release]
    if sample.hasOverrideGlobalTag():
        ogt = sample.overrideGlobalTag()
        if release in ogt:
            gtmap = _globalTags[ogt[release]]
    if sample.fullsim():
        if sample.hasScenario():
            return gtmap[sample.scenario()]
        if sample.hasPileup():
            puType = sample.pileupType()
            if "50ns" in puType:
                return gtmap.get("fullsim_50ns", gtmap["default"])
            if "25ns" in puType:
                return gtmap.get("fullsim_25ns", gtmap["default"])
    if sample.fastsim():
        if sample.hasPileup():
            puType = sample.pileupType()
            if "25ns" in puType:
                return gtmap.get("fastsim_25ns", gtmap["default"])
        return gtmap.get("fastsim", gtmap["default"])
    return gtmap["default"]

# Mapping from release series to RelVal download URLs
_relvalUrls = {
    "6_2_X": "https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/RelVal/CMSSW_6_2_x/",
    "7_0_X": "https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/RelVal/CMSSW_7_0_x/",
    "7_1_X": "https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/RelVal/CMSSW_7_1_x/",
    "7_2_X": "https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/RelVal/CMSSW_7_2_x/",
    "7_3_X": "https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/RelVal/CMSSW_7_3_x/",
    "7_4_X": "https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/RelVal/CMSSW_7_4_x/",
    "7_5_X": "https://cmsweb.cern.ch/dqm/relval/data/browse/ROOT/RelVal/CMSSW_7_5_x/",
}

def _getRelValUrl(release):
    """Get RelVal download URL for a given release."""
    version_re = re.compile("CMSSW_(?P<X>\d+)_(?P<Y>\d+)")
    m = version_re.search(release)
    if not m:
        raise Exception("Regex %s does not match to release version %s" % (version_re.pattern, release))
    version = "%s_%s_X" % (m.group("X"), m.group("Y"))
    if not version in _relvalUrls:
        print "No RelVal URL for version %s, please update _relvalUrls" % version
        sys.exit(1)
    return _relvalUrls[version]

class Sample:
    """Represents a RelVal sample."""
    def __init__(self, sample, append=None, midfix=None, putype=None,
                 fastsim=False, fastsimCorrespondingFullsimPileup=None,
                 version="v1", scenario=None, overrideGlobalTag=None):
        """Constructor.

        Arguments:
        sample -- String for name of the sample

        Keyword arguments
        append  -- String for a variable name within the DWM file names, to be directly appended to sample name (e.g. "HS"; default None)
        midfix  -- String for a variable name within the DQM file names, to be appended after underscore to "sample name+append" (e.g. "13", "UP15"; default None)
        putype  -- String for pileup type (e.g. "25ns"/"50ns" for FullSim, "AVE20" for FastSim; default None)
        fastsim -- Bool indicating the FastSim status (default False)
        fastsimCorrespondingFullSimPileup -- String indicating what is the FullSim pileup sample corresponding this FastSim sample. Must be set if fastsim=True and putype!=None (default None)
        version -- String for dataset/DQM file version (default "v1")
        scenario -- Geometry scenario for upgrade samples (default None)
        overrideGlobalTag -- GlobalTag obtained from release information (in the form of {"release": "actualRelease"}; default None)
        """
        self._sample = sample
        self._append = append
        self._midfix = midfix
        self._putype = putype
        self._fastsim = fastsim
        self._fastsimCorrespondingFullsimPileup = fastsimCorrespondingFullsimPileup
        self._version = version
        self._scenario = scenario
        self._overrideGlobalTag = overrideGlobalTag

        if self._fastsim and self.hasPileup() and self._fastsimCorrespondingFullsimPileup is None:
            self._fastsimCorrespondingFullsimPileup = self._putype

    def sample(self):
        """Get the sample name"""
        return self._sample

    def name(self):
        """Get the sample name"""
        return self._sample

    def hasPileup(self):
        """Return True if sample has pileup"""
        return self._putype is not None

    def pileup(self):
        """Return "PU"/"noPU" corresponding the pileup status"""
        if self.hasPileup():
            return "PU"
        else:
            return "noPU"

    def pileupType(self, release=None):
        """Return the pileup type"""
        if isinstance(self._putype, dict):
            return self._putype.get(release, self._putype["default"])
        else:
            return self._putype

    def version(self, release=None):
        if isinstance(self._version, dict):
            return self._version.get(release, self._version["default"])
        else:
            return self._version

    def hasScenario(self):
        return self._scenario is not None

    def scenario(self):
        return self._scenario

    def hasOverrideGlobalTag(self):
        return self._overrideGlobalTag is not None

    def overrideGlobalTag(self):
        return self._overrideGlobalTag

    def fastsim(self):
        """Return True for FastSim sample"""
        return self._fastsim

    def fullsim(self):
        """Return True for FullSim sample"""
        return not self._fastsim

    def fastsimCorrespondingFullsimPileup(self):
        return self._fastsimCorrespondingFullsimPileup

    def dirname(self, newRepository, newRelease, newSelection):
        """Return the output directory name

        Arguments:
        newRepository -- String for base directory for output files
        newRelease    -- String for CMSSW release
        newSelection  -- String for histogram selection
        """
        pileup = ""
        if self.hasPileup() and not self._fastsim:
            pileup = "_"+self._putype
        return "{newRepository}/{newRelease}/{newSelection}{pileup}/{sample}".format(
            newRepository=newRepository, newRelease=newRelease, newSelection=newSelection,
            pileup=pileup, sample=sample)

    def filename(self, newRelease):
        """Return the DQM file name

        Arguments:
        newRelease -- String for CMSSW release
        """
        pileup = ""
        fastsim = ""
        midfix = ""
        scenario = ""
        sample = self._sample
        if self._append is not None:
            midfix += self._append
        if self._midfix is not None:
            midfix += "_"+self._midfix
        if self.hasPileup():
            if self._fastsim:
                sample = sample.replace("RelVal", "RelValFS_")
                # old style
                #pileup = "PU_"
                #midfix += "_"+self.pileupType(newRelease)
                # new style
                pileup = "PU"+self.pileupType(newRelease)+"_"
            else:
                pileup = "PU"+self.pileupType(newRelease)+"_"
        if self._fastsim:
            fastsim = "_FastSim"
        if self._scenario is not None:
            scenario = "_"+self._scenario
            
        globalTag = _getGlobalTag(self, newRelease)

        fname = 'DQM_V0001_R000000001__{sample}{midfix}__{newrelease}-{pileup}{globaltag}{scenario}{fastsim}-{version}__DQMIO.root'.format(
            sample=sample, midfix=midfix, newrelease=_stripRelease(newRelease),
            pileup=pileup, globaltag=globalTag, scenario=scenario, fastsim=fastsim,
            version=self.version(newRelease)
        )

        return fname

    def datasetpattern(self, newRelease):
        """Return the dataset pattern

        Arguments:
        newRelease -- String for CMSSW release
        """
        pileup = ""
        fastsim = ""
        digi = ""
        if self.hasPileup():
            pileup = "-PU_"
        if self._fastsim:
            fastsim = "_FastSim-"
            digi = "DIGI-"
        else:
            fastsim = "*"
        globalTag = _getGlobalTag(self, newRelease)
        return "{sample}/{newrelease}-{pileup}{globaltag}{fastsim}{version}/GEN-SIM-{digi}RECO".format(
            sample=self._sample, newrelease=newRelease,
            pileup=pileup, globaltag=globalTag, fastsim=fastsim, digi=digi,
            version=self.version(newRelease)
            )

class Validation:
    """Base class for Tracking/Vertex validation."""
    def __init__(self, fullsimSamples, fastsimSamples, newRelease, newFileModifier=None, selectionName=""):
        """Constructor.

        Arguments:
        fullsimSamples -- List of Sample objects for FullSim samples (may be empty)
        fastsimSamples -- List of Sample objects for FastSim samples (may be empty)
        newRelease     -- CMSSW release to be validated
        newFileModifier -- If given, a function to modify the names of the new files (function takes a string and returns a string)
        selectionName  -- If given, use this string as the selection name (appended to GlobalTag for directory names)
        """
        try:
            self._newRelease = os.environ["CMSSW_VERSION"]
        except KeyError:
            print >>sys.stderr, 'Error: CMSSW environment variables are not available.'
            print >>sys.stderr, '       Please run cmsenv'
            sys.exit()

        self._fullsimSamples = fullsimSamples
        self._fastsimSamples = fastsimSamples
        if newRelease != "":
            self._newRelease = newRelease
        self._newFileModifier = newFileModifier
        self._selectionName = selectionName

    def _getDirectoryName(self, *args, **kwargs):
        return None

    def _getSelectionName(self, *args, **kwargs):
        return self._selectionName

    def download(self):
        """Download DQM files. Requires grid certificate and asks your password for it."""
        filenames = [s.filename(self._newRelease) for s in self._fullsimSamples+self._fastsimSamples]
        if self._newFileModifier is not None:
            filenames = map(self._newFileModifier, filenames)
        filenames = filter(lambda f: not os.path.exists(f), filenames)
        if len(filenames) == 0:
            print "All files already downloaded"
            return

        relvalUrl = _getRelValUrl(self._newRelease)
        urls = [relvalUrl+f for f in filenames]
        certfile = os.path.join(os.environ["HOME"], ".globus", "usercert.pem")
        if not os.path.exists(certfile):
            print "Certificate file {certfile} does not exist, unable to download RelVal files from {url}".format(certfile=certfile, url=relvalUrl)
            sys.exit(1)
        keyfile = os.path.join(os.environ["HOME"], ".globus", "userkey.pem")
        if not os.path.exists(certfile):
            print "Private key file {keyfile} does not exist, unable to download RelVal files from {url}".format(keyfile=keyfile, url=relvalUrl)
            sys.exit(1)
        
        cmd = ["curl", "--cert-type", "PEM", "--cert", certfile, "--key", keyfile, "-k"]
        for u in urls:
            cmd.extend(["-O", u])
        print "Downloading %d files from RelVal URL %s:" % (len(filenames), relvalUrl)
        print " "+"\n ".join(filenames)
        print "Please provide your private key pass phrase when curl asks it"
        ret = subprocess.call(cmd)
        if ret != 0:
            print "Downloading failed with exit code %d" % ret
            sys.exit(1)

        # verify
        allFine = True
        for f in filenames:
            p = subprocess.Popen(["file", f], stdout=subprocess.PIPE)
            stdout = p.communicate()[0]
            if p.returncode != 0:
                print "file command failed with exit code %d" % p.returncode
                sys.exit(1)
            if not "ROOT" in stdout:
                print "File {f} is not ROOT, please check the correct version, GobalTag etc. from {url}".format(f=f, url=relvalUrl)
                allFine = False
                if os.path.exists(f):
                    os.remove(f)
        if not allFine:
            sys.exit(1)

    def doPlots(self, algos, qualities, refRelease, refRepository, newRepository, plotter, plotterDrawArgs={}):
        """Create validation plots.

        Arguments:
        algos         -- List of strings for algoritms
        qualities     -- List of strings for quality flags (can be None)
        refRelease    -- String for reference CMSSW release
        refRepository -- String for directory where reference root files are
        newRepository -- String for directory whete to put new files
        plotter       -- plotting.Plotter object that does the plotting

        Keyword arguments:
        plotterDrawArgs -- Dictionary for additional arguments to Plotter.draw() (default: {})
        """
        self._refRelease = refRelease
        self._refRepository = refRepository
        self._newRepository = newRepository
        self._plotter = plotter
        self._plotterDrawArgs = plotterDrawArgs

        if qualities is None:
            qualities = [None]
        if algos is None:
            algos = [None]

        # New vs. Ref
        for s in self._fullsimSamples+self._fastsimSamples:
            for q in qualities:
                for a in algos:
                    self._doPlots(a, q, s)
#                    if s.fullsim() and s.hasPileup():
#                        self._doPlotsPileup(a, q, s)

        # Fast vs. Full in New
        for fast in self._fastsimSamples:
            correspondingFull = None
            for full in self._fullsimSamples:
                if fast.name() != full.name():
                    continue
                if fast.hasPileup():
                    if not full.hasPileup():
                        continue
                    if fast.fastsimCorrespondingFullsimPileup() != full.pileupType():
                        continue
                else:
                    if full.hasPileup():
                        continue

                if correspondingFull is None:
                    correspondingFull = full
                else:
                    raise Exception("Got multiple compatible FullSim samples for FastSim sample %s %s" % (fast.name(), fast.pileup()))
            if correspondingFull is None:
                raise Exception("Did not find compatible FullSim sample for FastSim sample %s %s" % (fast.name(), fast.pileup()))
            for q in qualities:
                for a in algos:
                    self._doPlotsFastFull(a, q, fast, correspondingFull)

    def _doPlots(self, algo, quality, sample):
        """Do the real plotting work for a given algorithm, quality flag, and sample."""
        # Get GlobalTags
        refGlobalTag = _getGlobalTag(sample, self._refRelease)
        newGlobalTag = _getGlobalTag(sample, self._newRelease)

        # Construct selection string
        tmp = ""
        if sample.hasScenario():
            tmp += "_"+sample.scenario()
        tmp += "_"+sample.pileup()
        refSelection = refGlobalTag+tmp+self._getSelectionName(quality, algo)
        newSelection = newGlobalTag+tmp+self._getSelectionName(quality, algo)
        if sample.hasPileup():
            refPu = sample.pileupType(self._refRelease)
            if refPu != "":
                refSelection += "_"+refPu
            newPu = sample.pileupType(self._newRelease)
            if newPu != "":
                newSelection += "_"+newPu

        # Check that the new DQM file exists
        harvestedfile = sample.filename(self._newRelease)
        if not os.path.exists(harvestedfile):
            print "Harvested file %s does not exist!" % harvestedfile
            sys.exit(1)

        valname = "val.{sample}.root".format(sample=sample.name())

        # Construct reference directory name
        tmp = [self._refRepository, self._refRelease]
        if sample.fastsim():
            tmp.extend(["fastsim", self._refRelease])
        tmp.extend([refSelection, sample.name()])
        refdir = os.path.join(*tmp)

        # Construct new directory name
        tmp = [self._newRepository, self._newRelease]
        if sample.fastsim():
            tmp.extend(["fastsim", self._newRelease])
        tmp.extend([newSelection, sample.name()])
        newdir = os.path.join(*tmp)

        # Open reference file if it exists
        refValFilePath = os.path.join(refdir, valname)
        if not os.path.exists(refValFilePath):
            print "Reference file %s not found" % refValFilePath
            if not plotting.missingOk:
                print "If this is acceptable, please set 'plotting.missingOk=True'"
                sys.exit(1)
            else:
                refValFile = None
        else:
            refValFile = ROOT.TFile.Open(refValFilePath)

        # Copy the relevant histograms to a new validation root file
        newValFile = _copySubDir(harvestedfile, valname, self._plotter.getPossibleDirectoryNames(), self._getDirectoryName(quality, algo))
        fileList = [valname]

        # Do the plots
        print "Comparing ref and new {sim} {sample} {algo} {quality}".format(
            sim="FullSim" if not sample.fastsim() else "FastSim",
            sample=sample.name(), algo=algo, quality=quality)
        self._plotter.create([refValFile, newValFile], [
            "%s, %s %s" % (sample.name(), _stripRelease(self._refRelease), refSelection),
            "%s, %s %s" % (sample.name(), _stripRelease(self._newRelease), newSelection)
        ],
                             subdir = self._getDirectoryName(quality, algo))
        fileList.extend(self._plotter.draw(algo, **self._plotterDrawArgs))

        newValFile.Close()
        if refValFile is not None:
            refValFile.Close()

        # Move plots to new directory
        print "Moving plots and %s to %s" % (valname, newdir)
        if not os.path.exists(newdir):
            os.makedirs(newdir)
        for f in fileList:
            shutil.move(f, os.path.join(newdir, f))

    def _doPlotsFastFull(self, algo, quality, fastSample, fullSample):
        """Do the real plotting work for FastSim vs. FullSim for a given algorithm, quality flag, and sample."""
        # Get GlobalTags
        fastGlobalTag = _getGlobalTag(fastSample, self._newRelease)
        fullGlobalTag = _getGlobalTag(fullSample, self._newRelease)

        # Construct selection string
        tmp = self._getSelectionName(quality, algo)
        fastSelection = fastGlobalTag+"_"+fastSample.pileup()+tmp
        fullSelection = fullGlobalTag+"_"+fullSample.pileup()+tmp
        if fullSample.hasPileup():
            fullSelection += "_"+fullSample.pileupType(self._newRelease)
            fastSelection += "_"+fastSample.pileupType(self._newRelease)

        # Construct directories for FastSim, FullSim, and for the results
        fastdir = os.path.join(self._newRepository, self._newRelease, "fastsim", self._newRelease, fastSelection, fastSample.name())
        fulldir = os.path.join(self._newRepository, self._newRelease, fullSelection, fullSample.name())
        newdir = os.path.join(self._newRepository, self._newRelease, "fastfull", self._newRelease, fastSelection, fastSample.name())

        # Open input root files
        valname = "val.{sample}.root".format(sample=fastSample.name())
        fastValFilePath = os.path.join(fastdir, valname)
        if not os.path.exists(fastValFilePath):
            print "FastSim file %s not found" % fastValFilePath
        fullValFilePath = os.path.join(fulldir, valname)
        if not os.path.exists(fullValFilePath):
            print "FullSim file %s not found" % fullValFilePath

        fastValFile = ROOT.TFile.Open(fastValFilePath)
        fullValFile = ROOT.TFile.Open(fullValFilePath)

        # Do plots
        print "Comparing FullSim and FastSim {sample} {algo} {quality}".format(
            sample=fastSample.name(), algo=algo, quality=quality)
        self._plotter.create([fullValFile, fastValFile], [
            "FullSim %s, %s %s" % (fullSample.name(), _stripRelease(self._newRelease), fullSelection),
            "FastSim %s, %s %s" % (fastSample.name(), _stripRelease(self._newRelease), fastSelection),
        ],
                             subdir = self._getDirectoryName(quality, algo))
        fileList = self._plotter.draw(algo, **self._plotterDrawArgs)

        fullValFile.Close()
        fastValFile.Close()
        
        # Move plots to new directory
        print "Moving plots to %s" % (newdir)
        if not os.path.exists(newdir):
            os.makedirs(newdir)
        for f in fileList:
            shutil.move(f, os.path.join(newdir, f))

    def _doPlotsPileup(self, algo, quality, sample):
        """Do the real plotting work for Old vs. New pileup scenarios for a given algorithm, quality flag, and sample."""
        # Get GlobalTags
        newGlobalTag = _getGlobalTag(sample, self._newRelease)
        refGlobalTag = newGlobalTag + "_OldPU" 

        # Construct selection string
        tmp = self._getSelectionName(quality, algo)
        refSelection = refGlobalTag+"_"+sample.pileup()+tmp+"_"+sample.pileupType(self._newRelease)
        newSelection = newGlobalTag+"_"+sample.pileup()+tmp+"_"+sample.pileupType(self._newRelease)

        # Construct directories for FastSim, FullSim, and for the results
        refdir = os.path.join(self._newRepository, self._newRelease, refSelection, sample.name())
        newdir = os.path.join(self._newRepository, self._newRelease, newSelection, sample.name())
        resdir = os.path.join(self._newRepository, self._newRelease, "pileup", self._newRelease, newSelection, sample.name())

        # Open input root files
        valname = "val.{sample}.root".format(sample=sample.name())
        refValFilePath = os.path.join(refdir, valname)
        if not os.path.exists(refValFilePath):
            print "Ref pileup file %s not found" % refValFilePath
        newValFilePath = os.path.join(newdir, valname)
        if not os.path.exists(newValFilePath):
            print "New pileup file %s not found" % newValFilePath

        refValFile = ROOT.TFile.Open(refValFilePath)
        newValFile = ROOT.TFile.Open(newValFilePath)

        # Do plots
        print "Comparing Old and New pileup {sample} {algo} {quality}".format(
            sample=sample.name(), algo=algo, quality=quality)
        self._plotter.create([refValFile, newValFile], [
            "%d BX %s, %s %s" % ({"25ns": 10, "50ns": 20}[sample.pileupType(self._newRelease)], sample.name(), _stripRelease(self._newRelease), refSelection),
            "35 BX %s, %s %s" % (sample.name(), _stripRelease(self._newRelease), newSelection),
        ],
                             subdir = self._getDirectoryName(quality, algo))
        fileList = self._plotter.draw(algo, **self._plotterDrawArgs)

        newValFile.Close()
        refValFile.Close()

        # Move plots to new directory
        print "Moving plots to %s" % (resdir)
        if not os.path.exists(resdir):
            os.makedirs(resdir)
        for f in fileList:
            shutil.move(f, os.path.join(resdir, f))

def _copySubDir(oldfile, newfile, basenames, dirname):
    """Copy a subdirectory from oldfile to newfile.

    Arguments:
    oldfile   -- String for source TFile
    newfile   -- String for destination TFile
    basenames -- List of strings for base directories, first existing one is picked
    dirname   -- String for directory name under the base directory
    """
    oldf = ROOT.TFile.Open(oldfile)

    dirold = None
    for basename in basenames:
        dirold = oldf.GetDirectory(basename)
        if dirold:
            break
    if not dirold:
        raise Exception("Did not find any of %s directories from file %s" % (",".join(basenames), oldfile))
    if dirname:
        d = dirold.Get(dirname)
        if not d:
            raise Exception("Did not find directory %s under %s" % (dirname, dirold.GetPath()))
        dirold = d

    newf = ROOT.TFile.Open(newfile, "RECREATE")
    dirnew = newf
    for d in basenames[0].split("/"):
        dirnew = dirnew.mkdir(d)
    if dirname:
        dirnew = dirnew.mkdir(dirname)
    _copyDir(dirold, dirnew)

    oldf.Close()
    return newf

def _copyDir(src, dst):
    """Copy recursively objects from src TDirectory to dst TDirectory."""
    keys = src.GetListOfKeys()
    for key in keys:
        classname = key.GetClassName()
        cl = ROOT.TClass.GetClass(classname)
        if not cl:
            continue
        if cl.InheritsFrom("TDirectory"):
            src2 = src.GetDirectory(key.GetName())
            dst2 = dst.mkdir(key.GetName())
            _copyDir(src2, dst2)
        elif cl.InheritsFrom("TTree"):
            t = key.ReadObj()
            dst.cd()
            newt = t.CloneTree(-1, "fast")
            newt.Write()
            newt.Delete()
        else:
            dst.cd()
            obj = key.ReadObj()
            obj.Write()
            obj.Delete()

class SimpleValidation:
    def __init__(self, files, labels, newdir):
        self._files = files
        self._labels = labels
        self._newdir = newdir

    def doPlots(self, algos, qualities, plotter, algoDirMap=None, newdirFunc=None, plotterDrawArgs={}):
        self._plotter = plotter
        self._algoDirMap = algoDirMap
        self._newdirFunc = newdirFunc
        self._plotterDrawArgs = plotterDrawArgs

        if qualities is None:
            qualities = [None]
        if algos is None:
            algos = [None]

        for q in qualities:
            for a in algos:
                self._doPlots(a, q)

    def _doPlots(self, algo, quality):
        openFiles = []
        for f in self._files:
            if not os.path.exists(f):
                print "File %s not found" % f
                sys.exit(1)
            openFiles.append(ROOT.TFile.Open(f))

        # dirs = []
        # for tf in openFiles:
        #     theDir = None
        #     for pd in self._plotter.getPossibleDirectoryNames():
        #         theDir = tf.GetDirectory(pd)
        #         if theDir:
        #             break
        #     if not theDir:
        #         print "Did not find any of %s directories from file %s" % (",".join(self._plotter.getPossibleDirectoryNames()), tf.GetName())
        #         sys.exit(1)
        #     if self._algoDirMap is not None:
        #         d = theDir.Get(self._algoDirMap[quality][algo])
        #         if not theDir:
        #             print "Did not find dir %s from %s" % (self._algoDirMap[quality][algo], theDir.GetPath())
        #             sys.exit(1)
        #         theDir = d
        #     dirs.append(theDir)

        subdir = None
        if self._algoDirMap is not None:
            subdir = self._algoDirMap[quality][algo]
        self._plotter.create(openFiles, self._labels, subdir=subdir)
        fileList = self._plotter.draw(algo, **self._plotterDrawArgs)

        for tf in openFiles:
            tf.Close()

        newdir = self._newdir
        if self._newdirFunc is not None:
            newdir = os.path.join(newdir, self._newdirFunc(algo, quality))

        print "Moving plots to %s" % newdir
        if not os.path.exists(newdir):
            os.makedirs(newdir)
        for f in fileList:
            shutil.move(f, os.path.join(newdir, f))
