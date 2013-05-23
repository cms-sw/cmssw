##  Note: Please do not use or modify any data or functions with a
##  leading underscore.  If you "mess" with the internal structure,
##  the classes may not function as intended.


from FWCore.Utilities.Enumerate import Enumerate
from DataFormats.FWLite import Events, Handle
import re
import os
import copy
import pprint
import random
import sys
import inspect
import ROOT
ROOT.gROOT.SetBatch()

# regex for reducing 'warn()' filenames
filenameRE = re.compile (r'.+/Validation/Tools/')
# Whether warn() should print anythingg
quietWarn = False

def setQuietWarn (quiet = True):
    global quietWarn
    quietWarn = quiet


def warn (*args, **kwargs):
    """print out warning with line number and rest of arguments"""
    if quietWarn: return
    frame = inspect.stack()[1]
    filename = frame[1]
    lineNum  = frame[2]
    #print filename, filenameRE
    filename = filenameRE.sub ('', filename)
    #print "after '%s'" % filename
    blankLines = kwargs.get('blankLines', 0)
    if blankLines:
        print '\n' * blankLines
    spaces = kwargs.get('spaces', 0)
    if spaces:
        print ' ' * spaces,
    if len (args):
        print "%s (%s): " % (filename, lineNum),
        for arg in args:
            print arg,
        print
    else:
        print "%s (%s):" % (filename, lineNum)


class GenObject (object):
    """Infrastruture to define general objects and their attributes."""

    ########################
    ## Static Member Data ##
    ########################

    types              = Enumerate ("float int long string", "type")
    _objFunc           = Enumerate ("obj func", "of")
    _cppType           = dict ( {types.float  : 'double',
                                 types.int    : 'int',
                                 types.long   : 'long',
                                 types.string : 'std::string' } )
    _basicSet          = set( [types.float, types.int, types.float,
                               types.string] )
    _defaultValue      = dict ( {types.float  : 0.,
                                 types.int    : 0,
                                 types.long   : 0L,
                                 types.string : '""' } )
    _objsDict          = {} # info about GenObjects
    _equivDict         = {} # hold info about 'equivalent' muons
    _ntupleDict        = {} # information about different ntuples
    _tofillDict        = {} # information on how to fill from different ntuples
    _rootObjectDict    = {} # hold objects and stl::vectors to objects
                            # hooked up to a root tree
    _rootClassDict     = {} # holds classes (not instances) associated with
                            # a given GenObject
    _kitchenSinkDict   = {} # dictionary that holds everything else...
    _runEventList      = []
    _runEventListDone  = False
    uselessReturnCode  = 1 << 7 # pick a unique return code
    
    ####################
    ## Compile Regexs ##
    ####################
    _nonSpacesRE      = re.compile (r'\S')
    _colonRE          = re.compile (r'\s*:\s*')
    _singleColonRE    = re.compile (r'(.+?):(.+)')
    _doubleColonRE    = re.compile (r'(.+?):(.+?):(.+)')
    _doublePercentRE  = re.compile (r'%%')    
    _parenRE          = re.compile (r'(.+)\((.*)\)')
    _spacesRE         = re.compile (r'\s+')
    _dotRE            = re.compile (r'\s*\.\s*')
    _commaRE          = re.compile (r'\s*,\s*')
    _singleQuoteRE    = re.compile (r'^\'(.+)\'$')
    _doubleQuoteRE    = re.compile (r'^\"(.+)\"$')
    _bracketRE        = re.compile (r'\[\s*(.+?)\s*\]')
    _commentRE        = re.compile (r'#.+$')
    _aliasRE          = re.compile (r'alias=(\S+)',    re.IGNORECASE)
    _labelRE          = re.compile (r'label=(\S+)',    re.IGNORECASE)
    _typeRE           = re.compile (r'type=(\S+)',     re.IGNORECASE)
    _singletonRE      = re.compile (r'singleton',      re.IGNORECASE)
    _typeRE           = re.compile (r'type=(\S+)',     re.IGNORECASE)
    _defaultRE        = re.compile (r'default=(\S+)',  re.IGNORECASE)
    _shortcutRE       = re.compile (r'shortcut=(\S+)', re.IGNORECASE)
    _precRE           = re.compile (r'prec=(\S+)',     re.IGNORECASE)
    _formRE           = re.compile (r'form=(\S+)',     re.IGNORECASE)
    _nonAlphaRE       = re.compile (r'\W')
    _percentAsciiRE   = re.compile (r'%([0-9a-fA-F]{2})')
    
    #############################
    ## Static Member Functions ##
    #############################

    @staticmethod
    def char2ascii (match):
        return "%%%02x" % ord (match.group(0))

    @staticmethod
    def ascii2char (match):
        return chr( int( match.group(1), 16 ) )

    @staticmethod
    def encodeNonAlphanumerics (line):
        """Use a web like encoding of characters that are non-alphanumeric"""
        return GenObject._nonAlphaRE.sub( GenObject.char2ascii, line )

    @staticmethod
    def decodeNonAlphanumerics (line):
        """Decode lines encoded with encodeNonAlphanumerics()"""
        return GenObject._percentAsciiRE.sub( GenObject.ascii2char, line )
        

    @staticmethod
    def addObjectVariable (obj, var, **optionsDict):
        """ User passes in in object and variable names."""
        if not optionsDict.has_key ('varType'):
            optionsDict['varType'] = GenObject.types.float
        varType = optionsDict['varType']
        if not GenObject.types.isValidValue (varType):
            print "Type '%s' not valid.  Skipping (%s, %s, %s)." % \
                  (varType, obj, var, varType)
            return
        if not optionsDict.has_key ('default'):
            optionsDict['default'] = GenObject._defaultValue[varType]
        if obj.startswith ("_") or var.startswith ("_"):
            print "Skipping (%s, %s, %s) because of leading underscore." % \
                  (obj, var, varType)
            return
        GenObject._objsDict.setdefault (obj, {}).setdefault (var, optionsDict)


    @staticmethod
    def getVariableProperty (obj, var, key):
        """Returns property assoicated with 'key' for variable 'var'
        of object 'obj'.  Returns 'None' if any of the above are not
        defined."""
        return GenObject._objsDict.get (obj, {}).get (var, {}). get (key, None)


    @staticmethod
    def setEquivExpression (obj, variable, precision):
        """Adds an equivalence constraint.  Must have at least one to
        compare GO objects."""
        if obj.startswith ("_"):
            print "Skipping (%s, %s) because of leading underscore." % \
                  (obj, expression)
            return
        GenObject._equivDict.setdefault (obj,[]).append ( (variable,
                                                           precision) )
        
    
    @staticmethod
    def printGlobal():
        """Meant for debugging, but ok if called by user"""
        print "objs: "
        pprint.pprint (GenObject._objsDict,        indent=4)
        print "equiv: "                            
        pprint.pprint (GenObject._equivDict,       indent=4)
        print "ntuple: "                           
        pprint.pprint (GenObject._ntupleDict,      indent=4)
        print "tofill: "                           
        pprint.pprint (GenObject._tofillDict,      indent=4)
        print "kitchenSink: "
        pprint.pprint (GenObject._kitchenSinkDict, indent=4)
        print "rootClassDict"
        pprint.pprint (GenObject._rootClassDict,   indent=4)


    @staticmethod
    def checksum (str):
        """Returns a string of hex value of a checksum of input
        string."""
        return hex( reduce( lambda x, y : x + y, map(ord, str) ) )[2:]


    @staticmethod
    def rootClassName (objName):
        """Returns the name of the equivalent Root object"""
        return "go_" + objName


    @staticmethod
    def rootDiffClassName (objName):
        """Returns the name of the equivalent Root diff object"""
        return "goDiff_" + objName


    @staticmethod
    def rootDiffContClassName (objName):
        """Returns the name of the equivalent Root diff container
        object"""
        return "goDiffCont_" + objName


    @staticmethod
    def _setupClassHeader (className, noColon = False):
        """Returns a string with the class header for a class
        'classname'"""
        retval  = "\nclass %s\n{\n  public:\n" % className
        retval += "      typedef std::vector< %s > Collection;\n\n" % className
        # constructor
        if noColon:
            retval += "      %s()" % className
        else:
            retval += "      %s() :\n" % className
        return retval


    @staticmethod
    def _finishClassHeader (className, datadec):
        """Returns a stringg with the end of a class definition"""
        retval  = "\n      {}\n" + datadec + "};\n"
        retval += "#ifdef __MAKECINT__\n#pragma link C++ class " + \
                 "vector< %s >+;\n#endif\n\n" % className
        return retval


    @staticmethod
    def _createCppClass (objName):
        """Returns a string containing the '.C' file necessary to
        generate a shared object library with dictionary."""
        if not GenObject._objsDict.has_key (objName):
            # not good
            print "Error: GenObject does not know about object '%s'." % objName
            raise RuntimeError, "Failed to create C++ class."
        className   = GenObject.rootClassName (objName)
        diffName    = GenObject.rootDiffClassName (objName)
        contName    = GenObject.rootDiffContClassName (objName)
        goClass     = GenObject._setupClassHeader (className)
        diffClass   = GenObject._setupClassHeader (diffName)
        contClass   = GenObject._setupClassHeader (contName, noColon = True)
        goDataDec   = diffDataDec = contDataDec = "\n      // data members\n"
        first = True
        for key in sorted( GenObject._objsDict[objName].keys() ):
            if key.startswith ("_"): continue
            varTypeList = GenObject._objsDict[objName][key]
            cppType = GenObject._cppType[ varTypeList['varType'] ]
            default = varTypeList['default']
            if first:
                first = False
            else:
                goClass   += ",\n"
                diffClass += ',\n'
            goClass   += "         %s (%s)" % (key, default)
            goDataDec += "      %s %s;\n" % (cppType, key)
            # is this a basic class?
            goType = varTypeList['varType']
            if goType in GenObject._basicSet:
                # basic type
                diffClass   += "         %s (%s),\n" % (key, default)
                diffDataDec += "      %s %s;\n" % (cppType, key)
                if goType == GenObject.types.string:
                    # string
                    otherKey = 'other_' + key
                    diffClass   += "         %s (%s)" % (otherKey, default)
                    diffDataDec += "      %s %s;\n" % (cppType, otherKey)
                else:
                    # float, long, or int
                    deltaKey = 'delta_' + key
                    diffClass   += "         %s (%s)" % (deltaKey, default)
                    diffDataDec += "      %s %s;\n" % (cppType, deltaKey)
            else:
                raise RuntimeError, "Shouldn't be here yet."
            # definition
        # do contClass
        if GenObject.isSingleton (objName):
            # singleton
            contDataDec += "         %s diff\n" % diffName
            contDataDec += "      void setDiff (const %s &rhs)" % diffName + \
                           " { diff = rhs; }\n"
        else:
            # vector of objects
            contDataDec += "      void clear() {firstOnly.clear(); secondOnly.clear(); diff.clear(); }\n"
            contDataDec += "      %s::Collection firstOnly;\n"  % className
            contDataDec += "      %s::Collection secondOnly;\n" % className
            contDataDec += "      %s::Collection diff;\n"       % diffName
            # give me a way to clear them all at once
        # Finish off the classes
        goClass   += GenObject._finishClassHeader (className, goDataDec)
        diffClass += GenObject._finishClassHeader (diffName,  diffDataDec)
        contClass += GenObject._finishClassHeader (contName,  contDataDec)
        if objName == 'runevent':
            # we don't want a diff class for this
            diffClass = ''
            contClass = ''
        return goClass + diffClass + contClass


    @staticmethod
    def _loadGoRootLibrary ():
        """Loads Root shared object library associated with all
        defined GenObjects. Will create library if necessary."""
        print "Loading GO Root Library"
        key = "_loadedLibrary"        
        if GenObject._kitchenSinkDict.get (key):
            # Already done, don't do it again:
            return
        # Mark it as done
        GenObject._kitchenSinkDict[key] = True
        # Generate source code
        sourceCode = "#include <string>\n#include <vector>\n" \
                     + "using namespace std;\n"
        for objClassName in sorted( GenObject._objsDict.keys() ):
            sourceCode = sourceCode + GenObject._createCppClass (objClassName)
        GenObjectRootLibDir = "genobjectrootlibs"
        if not os.path.exists (GenObjectRootLibDir):
            os.mkdir (GenObjectRootLibDir)
        key = GenObject.checksum( sourceCode )
        basename = "%s_%s" % ("GenObject", key)
        SO = "%s/%s_C" % (GenObjectRootLibDir, basename)
        linuxSO = "%s.so" % SO
        windowsSO = "%s.dll" % SO
        if not os.path.exists (linuxSO) and not os.path.exists (windowsSO):
            print "creating SO"
            filename = "%s/%s.C" % (GenObjectRootLibDir, basename)
            if not os.path.exists (filename):
                print "creating .C file"
                target = open (filename, "w")
                target.write (sourceCode)
                target.close()
            else:
                print "%s exists" % filename
            ## command = "echo .L %s+ | root.exe -b" % filename
            ## os.system (command)
            ROOT.gSystem.CompileMacro (filename,"k")
        else:
            print "loading %s" % SO
            ROOT.gSystem.Load(SO)
        return


    @staticmethod
    def _tofillGenObject():
        """Makes sure that I know how to read root files I made myself"""
        genObject = "GenObject"
        ntupleDict = GenObject._ntupleDict.setdefault (genObject, {})
        ntupleDict['_useChain'] = True
        ntupleDict['_tree'] = "goTree"
        for objName in GenObject._objsDict.keys():
            rootObjName = GenObject.rootClassName (objName)
            if GenObject.isSingleton (objName):
                ntupleDict[objName] = objName
            else:
                ntupleDict[objName] = objName + "s"
            tofillDict = GenObject._tofillDict.\
                         setdefault (genObject, {}).\
                         setdefault (objName, {})
            for varName in GenObject._objsDict [objName].keys():
                # if the key starts with an '_', then it is not a
                # variable, so don't treat it as one.
                if varName.startswith ("_"):
                    continue
                tofillDict[varName] = [ [(varName, GenObject._objFunc.obj)],
                                        {}]


    @staticmethod
    def prepareToLoadGenObject():
        """Makes all necessary preparations to load root files created
        by GenObject."""
        GenObject._tofillGenObject()
        GenObject._loadGoRootLibrary()


    @staticmethod
    def parseVariableTofill (fillname):
        """Returns tofill tuple made from string"""
        parts = GenObject._dotRE.split (fillname)
        partsList = []
        for part in parts:
            parenMatch = GenObject._parenRE.search (part)
            mode   = GenObject._objFunc.obj
            parens = []
            if parenMatch:
                part   = parenMatch.group (1)
                mode   = GenObject._objFunc.func
                parens = \
                       GenObject._convertStringToParameters \
                       (parenMatch.group (2))
            partsList.append(  (part, mode, parens) )
        return partsList

    @staticmethod
    def _fixLostGreaterThans (handle):
        offset = handle.count ('<') - handle.count('>')
        if not offset:
            return handle
        if offset < 0:
            print "Huh?  Too few '<' for each '>' in handle '%'" % handle
            return handle
        return handle + ' >' * offset


    @staticmethod
    def loadConfigFile (configFile):
        """Loads configuration file"""
        objName    = ""
        tupleName  = ""
        tofillName = ""
        modeEnum   = Enumerate ("none define tofill ntuple", "mode")
        mode       = modeEnum.none
        try:
            config = open (configFile, 'r')
        except:
            raise RuntimeError, "Can't open configuration '%s'" % configFile
        for lineNum, fullLine in enumerate (config):
            fullLine = fullLine.strip()
            # get rid of comments
            line = GenObject._commentRE.sub ('', fullLine)
            # Is there anything on this line?
            if not GenObject._nonSpacesRE.search (line):
                # Nothing to see here folks.  Keep moving....
                continue
            ###################
            ## Bracket Match ##
            ###################
            bracketMatch = GenObject._bracketRE.search (line)
            if bracketMatch:
                # a header
                section = bracketMatch.group(1)
                words = GenObject._spacesRE.split( section )
                if len (words) < 1:
                    raise RuntimeError, "Don't understand line '%s'(%d)" \
                          % (fullLine, lineNum)
                # The first word is the object name
                # reset the rest of the list
                objName = words[0]
                words = words[1:]
                colonWords = GenObject._colonRE.split (objName)
                if len (colonWords) > 3:
                    raise RuntimeError, "Don't understand line '%s'(%d)" \
                          % (fullLine, lineNum)
                if len (colonWords) == 1:
                    ##########################
                    ## GenObject Definition ##
                    ##########################
                    mode = modeEnum.define
                    for word in words:
                        if GenObject._singletonRE.match (word):
                            #GenObject._singletonSet.add (objName)
                            objsDict = GenObject._objsDict.\
                                       setdefault (objName, {})
                            objsDict['_singleton'] = True
                            continue
                        # If we're still here, then we didn't have a valid
                        # option.  Complain vociferously
                        print "I don't understand '%s' in section '%s' : %s" \
                              % (word, section, mode)
                        raise RuntimeError, \
                              "Config file parser error '%s'(%d)" \
                              % (fullLine, lineNum)
                elif len (colonWords) == 2:
                    #######################
                    ## Ntuple Definition ##
                    #######################
                    mode = modeEnum.ntuple
                    ntupleDict = GenObject._ntupleDict.\
                                setdefault (colonWords[0], {})
                    ntupleDict['_tree'] = colonWords[1]
                else:
                    ##########################
                    ## Object 'tofill' Info ##
                    ##########################
                    mode = modeEnum.tofill
                    objName    = colonWords [0]
                    tupleName  = colonWords [1]
                    tofillName = colonWords [2]
                    ntupleDict = GenObject._ntupleDict.\
                                 setdefault (tupleName, {})
                    ntupleDict[objName] = tofillName
                    for word in words:
                        # label
                        labelMatch = GenObject._labelRE.search (word)
                        if labelMatch:
                            label = tuple( GenObject._commaRE.\
                                           split(  labelMatch.group (1) ) )
                            ntupleDict.setdefault ('_label', {}).\
                                                  setdefault (tofillName,
                                                              label)
                            continue
                        # shortcut
                        shortcutMatch = GenObject._shortcutRE.search (word)
                        if shortcutMatch:
                            shortcutFill = \
                                         GenObject.\
                                         parseVariableTofill ( shortcutMatch.\
                                                               group(1) )
                            ntupleDict.setdefault ('_shortcut', {}).\
                                                  setdefault (tofillName,
                                                              shortcutFill)
                            continue
                        # type/handle
                        typeMatch = GenObject._typeRE.search (word)
                        if typeMatch:
                            handleString = \
                                         GenObject.\
                                         _fixLostGreaterThans (typeMatch.group(1))
                            handle = Handle( handleString )
                            ntupleDict.setdefault ('_handle', {}).\
                                                  setdefault (tofillName,
                                                              handle)
                            continue
                        # alias 
                        aliasMatch = GenObject._aliasRE.search (word)
                        if aliasMatch:
                            ntupleDict.setdefault ('_alias', {}).\
                                                  setdefault (tofillName,
                                                              aliasMatch.\
                                                              group(1))
                            continue
                        # is this a lost '<'
                        if word == '>':
                            continue
                        # If we're still here, then we didn't have a valid
                        # option.  Complain vociferously
                        print "I don't understand '%s' in section '%s' : %s" \
                              % (word, section, mode)
                        raise RuntimeError, \
                              "Config file parser error '%s'(%d)" \
                              % (fullLine, lineNum)
            ##############
            ## Variable ##
            ##############
            else:
                # a variable
                if modeEnum.none == mode:
                    # Poorly formatted 'section' tag
                    print "I don't understand line '%s'." % fullLine
                    raise RuntimeError, \
                          "Config file parser error '%s'(%d)" \
                          % (fullLine, lineNum)
                colonWords = GenObject._colonRE.split (line, 1)
                if len (colonWords) < 2:
                    # Poorly formatted 'section' tag
                    print "I don't understand line '%s'." % fullLine
                    raise RuntimeError, \
                          "Config file parser error '%s'(%d)" \
                          % (fullLine, lineNum)
                varName = colonWords[0]
                option  = colonWords[1]
                if option:
                    pieces = GenObject._spacesRE.split (option)
                else:
                    pieces = []
                if modeEnum.define == mode:
                    #########################
                    ## Variable Definition ##
                    #########################
                    # is this a variable or an option?
                    if varName.startswith("-"):
                        # this is an option
                        if "-equiv" == varName.lower():
                            for part in pieces:
                                halves = part.split (",")
                                if 2 != len (halves):
                                    print "Problem with -equiv '%s' in '%s'" % \
                                          (part, section)
                                    raise RuntimeError, \
                                          "Config file parser error '%s'(%d)" \
                                          % (fullLine, lineNum)
                                if halves[1]:
                                    halves[1] = float (halves[1])
                                    if not halves[1] >= 0:
                                        print "Problem with -equiv ",\
                                              "'%s' in '%s'" % \
                                              (part, section)
                                        raise RuntimeError, \
                                              "Config file parser error '%s'(%d)" \
                                              % (fullLine, lineNum)
                                GenObject.setEquivExpression (section,
                                                              halves[0],
                                                              halves[1])
                        continue
                    # If we're here, then this is a variable
                    optionsDict = {}
                    for word in pieces:
                        typeMatch = GenObject._typeRE.search (word)
                        if typeMatch and \
                               GenObject.types.isValidKey (typeMatch.group(1)):
                            varType = typeMatch.group(1).lower()
                            optionsDict['varType'] = GenObject.types (varType)
                            continue
                        defaultMatch = GenObject._defaultRE.search (word)
                        if defaultMatch:
                            optionsDict['default'] = defaultMatch.group(1)
                            continue
                        precMatch = GenObject._precRE.search (word)
                        if precMatch:
                            optionsDict['prec'] = float (precMatch.group (1))
                            continue
                        formMatch = GenObject._formRE.search (word)
                        if formMatch:
                            form = GenObject._doublePercentRE.\
                                   sub ('%', formMatch.group (1))
                            optionsDict['form'] = form
                            continue
                        # If we're still here, then we didn't have a valid
                        # option.  Complain vociferously
                        print "I don't understand '%s' in section '%s'." \
                              % (word, option)
                        raise RuntimeError, \
                              "Config file parser error '%s'(%d)" \
                              % (fullLine, lineNum)
                    GenObject.addObjectVariable (objName, varName, \
                                                 **optionsDict)
                else: # if modeEnum.define != mode
                    ############################
                    ## Variable 'tofill' Info ##
                    ############################
                    if len (pieces) < 1:
                        continue
                    fillname, pieces = pieces[0], pieces[1:]
                    ## parts = GenObject._dotRE.split (fillname)
                    ## partsList = []
                    ## for part in parts:
                    ##     parenMatch = GenObject._parenRE.search (part)
                    ##     mode   = GenObject._objFunc.obj
                    ##     parens = []
                    ##     if parenMatch:
                    ##         part   = parenMatch.group (1)
                    ##         mode   = GenObject._objFunc.func
                    ##         parens = \
                    ##                GenObject._convertStringToParameters \
                    ##                (parenMatch.group (2))
                    ##     partsList.append(  (part, mode, parens) )
                    # I don't yet have any options available here, but
                    # I'm keeping the code here for when I add them.
                    optionsDict = {}
                    for word in pieces:
                        # If we're still here, then we didn't have a valid
                        # option.  Complain vociferously
                        print "I don't understand '%s' in section '%s'." \
                              % (word, option)
                        raise RuntimeError, \
                              "Config file parser error '%s'(%d)" \
                              % (fullLine, lineNum)
                    tofillDict = GenObject._tofillDict.\
                                 setdefault (tupleName, {}).\
                                 setdefault (objName, {})
                    partsList = GenObject.parseVariableTofill (fillname)
                    tofillDict[varName] = [partsList, optionsDict]
        # for line
        for objName in GenObject._objsDict:
            # if this isn't a singleton, add 'index' as a variable
            if not GenObject.isSingleton (objName):
                GenObject.addObjectVariable (objName, 'index',
                                             varType = GenObject.types.int,
                                             form = '%3d')


    @staticmethod
    def changeVariable (tupleName, objName, varName, fillname):
        """Updates the definition used to go from a Root object to a
        GenObject.  'tupleName' and 'objName' must already be defined."""
        parts = GenObject._dotRE.split (fillname)
        partsList = []
        for part in parts:
            parenMatch = GenObject._parenRE.search (part)
            mode   = GenObject._objFunc.obj
            parens = []
            if parenMatch:
                part   = parenMatch.group (1)
                mode   = GenObject._objFunc.func
                parens = \
                       GenObject._convertStringToParameters \
                       (parenMatch.group (2))
            partsList.append(  (part, mode, parens) )
        GenObject._tofillDict[tupleName][objName][varName] = [partsList, {}]



    @staticmethod
    def evaluateFunction (obj, partsList, debug=False):
        """Evaluates function described in partsList on obj"""
        for part in partsList:
            if debug: warn (part, spaces=15)
            obj = getattr (obj, part[0])
            if debug: warn (obj, spaces=15)
            # if this is a function instead of a data member, make
            # sure you call it with its arguments:
            if GenObject._objFunc.func == part[1]:
                # Arguments are stored as a list in part[2]
                obj = obj (*part[2])
                if debug: warn (obj, spaces=18)
        return obj


    @staticmethod
    def _genObjectClone (objName, tupleName, obj, index = -1):
        """Creates a GenObject copy of Root object"""
        debug = GenObject._kitchenSinkDict.get ('debug', False)
        if objName == 'runevent':
            debug = False
        tofillObjDict = GenObject._tofillDict.get(tupleName, {})\
                        .get(objName, {})
        genObj = GenObject (objName)
        origObj = obj
        if debug: warn (objName, spaces = 9)
        for genVar, ntDict in tofillObjDict.iteritems():
            if debug: warn (genVar, spaces = 12)
            # lets work our way down the list
            partsList = ntDict[0]
            # start off with the original object
            obj = GenObject.evaluateFunction (origObj, partsList, debug)
            ## for part in partsList:
            ##     if debug: warn (part, spaces=15)
            ##     obj = getattr (obj, part[0])
            ##     if debug: warn (obj, spaces=15)
            ##     # if this is a function instead of a data member, make
            ##     # sure you call it with its arguments:
            ##     if GenObject._objFunc.func == part[1]:
            ##         # Arguments are stored as a list in part[2]
            ##         obj = obj (*part[2])
            ##         if debug: warn (obj, spaces=18)
            if debug: warn (obj, spaces=12)
            setattr (genObj, genVar, obj)
        # Do I need to store the index of this object?
        if index >= 0:
            setattr (genObj, 'index', index)
        return genObj


    @staticmethod
    def _rootObjectCopy (goSource, rootTarget):
        """Copies information from goSourse into Root Object"""
        objName = goSource._objName
        for varName in GenObject._objsDict [objName].keys():
            # if the key starts with an '_', then it is not a
            # variable, so don't treat it as one.
            if varName.startswith ("_"):
                continue
            setattr( rootTarget, varName, goSource (varName) )
        
        
    @staticmethod
    def _rootObjectClone (obj):
        """Creates the approprite type of Root object and copies the
        information into it from the GO object."""
        objName = obj._objName
        classObj = GenObject._rootClassDict.get (objName)
        if not classObj:
            goName = GenObject.rootClassName (objName)
            classObj = GenObject._rootClassDict[ goName ]
        rootObj = classObj()
        for varName in GenObject._objsDict [objName].keys():
            setattr( rootObj, varName, obj (varName) )
        return rootObj


    @staticmethod
    def _rootDiffObject (obj1, obj2, rootObj = 0):
        """Given to GOs, it will create and fill the corresponding
        root diff object"""
        objName = obj1._objName
        # if we don't already have a root object, create one
        if not rootObj:
            diffName = GenObject.rootDiffClassName( objName )
            rootObj = GenObject._rootClassDict[diffName]()
        for varName in GenObject._objsDict [objName].keys():
            if varName.startswith ("_"): continue
            goType = GenObject._objsDict[objName][varName]['varType']
            if not goType in GenObject._basicSet:
                # not yet
                continue
            setattr( rootObj, varName, obj1 (varName) )            
            if  goType == GenObject.types.string:
                # string
                otherName = 'other_' + varName
                if obj1 (varName) != obj2 (varName):
                    # don't agree
                    setattr( rootObj, otherName, obj2 (varName) )
                else:
                    # agree
                    setattr( rootObj, otherName, '' )
            else:
                # float, long, or int
                deltaName = 'delta_' + varName
                setattr( rootObj, deltaName,
                         obj2 (varName) - obj1 (varName) )
        return rootObj


    @staticmethod
    def setupOutputTree (outputFile, treeName, treeDescription = "",
                         otherNtupleName = ""):
        """Opens the output file, loads all of the necessary shared
        object libraries, and returns the output file and tree.  If
        'otherNtupleName' is given, it will check and make sure that
        only objects that are defined in it are written out."""
        rootfile = ROOT.TFile.Open (outputFile, "recreate")
        tree = ROOT.TTree (treeName, treeDescription)
        GenObject._loadGoRootLibrary()
        for objName in sorted (GenObject._objsDict.keys()):
            classname = GenObject.rootClassName (objName)
            rootObj = \
                    GenObject._rootClassDict[objName] = \
                    getattr (ROOT, classname)
            if GenObject.isSingleton (objName):
                # singleton object
                obj = GenObject._rootObjectDict[objName] = rootObj()
                tree.Branch (objName, classname, obj)
            else:
                # vector of objects - PLEASE don't forget the '()' on
                # the end of the declaration.  Without this, you are
                # defining a type, not instantiating an object.
                vec = \
                    GenObject._rootObjectDict[objName] = \
                    ROOT.std.vector( rootObj )()
                branchName = objName + "s"
                vecName = "vector<%s>" % classname
                tree.Branch( branchName, vecName, vec)
            # end else if isSingleton
        # end for objName
        return rootfile, tree


    @staticmethod
    def setupDiffOutputTree (outputFile, diffName, missingName,
                             diffDescription = '', missingDescription = ''):
        """Opens the diff output file, loads all of the necessary
        shared object libraries, and returns the output file and tree.b"""
        rootfile = ROOT.TFile.Open (outputFile, "recreate")
        GenObject._loadGoRootLibrary()
        # diff tree
        diffTree = ROOT.TTree (diffName, diffDescription)
        runEventObject = getattr (ROOT, 'go_runevent')()
        diffTree.Branch ('runevent', 'go_runevent', runEventObject)
        GenObject._rootClassDict['runevent'] = runEventObject
        for objName in sorted (GenObject._objsDict.keys()):
            if objName == 'runevent': continue
            classname = GenObject.rootClassName (objName)
            GenObject._rootClassDict[classname] = getattr (ROOT, classname)
            contName = GenObject.rootDiffContClassName (objName)
            diffName = GenObject.rootDiffClassName (objName)
            rootObj = GenObject._rootClassDict[contName] = \
                    getattr (ROOT, contName)
            GenObject._rootClassDict[diffName] = getattr (ROOT, diffName)
            obj = GenObject._rootObjectDict[objName] = rootObj()
            diffTree.Branch (objName, contName, obj)
        # missing tree
        missingTree = ROOT.TTree (missingName, missingDescription)
        rootRunEventClass = getattr (ROOT, 'go_runevent')
        firstOnly  = GenObject._rootClassDict['firstOnly'] = \
                     ROOT.std.vector( rootRunEventClass ) ()
        secondOnly = GenObject._rootClassDict['secondOnly'] = \
                     ROOT.std.vector( rootRunEventClass ) ()
        missingTree.Branch ('firstOnly',  'vector<go_runevent>', firstOnly) 
        missingTree.Branch ('secondOnly', 'vector<go_runevent>', secondOnly) 
        return rootfile, diffTree, missingTree


    @staticmethod
    def _fillRootObjects (event):
        """Fills root objects from GenObject 'event'"""
        for objName, obj in sorted (event.iteritems()):
            if GenObject.isSingleton (objName):
                # Just one
                GenObject._rootObjectCopy (obj,
                                           GenObject._rootObjectDict[objName])
            else:
                # a vector
                vec = GenObject._rootObjectDict[objName]
                vec.clear()
                for goObj in obj:
                    vec.push_back( GenObject._rootObjectClone (goObj) )


    @staticmethod
    def _fillRootDiffs (event1, event2):
        """Fills root diff containers from two GenObject 'event's"""
        


    @staticmethod
    def isSingleton (objName):
        """Returns true if object is a singleton"""
        return GenObject._objsDict[objName].get('_singleton')


    @staticmethod
    def loadEventFromTree (eventTree, eventIndex,
                           onlyRunEvent  = False):
        """Loads event information from Root file (as interfaced by
        'cmstools.EventTree' or 'ROOT.TChain').  Returns a dictionary
        'event' containing lists of objects or singleton object.  If
        'onlyRunEvent' is et to True, then only run and event number
        is read in from the tree."""
        debug     = GenObject._kitchenSinkDict.get ('debug', False)
        tupleName = GenObject._kitchenSinkDict[eventTree]['tupleName']
        event = {}
        # is this a cint tree
        isChain = eventTree.__class__.__name__ == 'TChain'
        if isChain:
            # This means that 'eventTree' is a ROOT.TChain
            eventTree.GetEntry (eventIndex)
        else:
            # This means that 'eventTree' is a FWLite Events
            eventTree.to(eventIndex)
        tofillDict = GenObject._tofillDict.get (tupleName)
        ntupleDict = GenObject._ntupleDict.get (tupleName)
        if not tofillDict:
            print "Don't know how to fill from '%s' ntuple." % tupleName
            return
        eventBranchName = ntupleDict['runevent']
        for objName in tofillDict:
            branchName = ntupleDict[objName]
            if onlyRunEvent and branchName != eventBranchName:
                # not now
                continue
            # Have we been given 'tofill' info for this object?
            if not branchName:
                # guess not
                continue
            if isChain:
                # Root TChain
                objects = getattr (eventTree, branchName)
            else:
                # FWLite
                shortcut = ntupleDict.get('_shortcut', {}).get(branchName)
                if shortcut:
                    objects = GenObject.evaluateFunction (eventTree, shortcut)
                else:
                    eventTree.toBegin()
                    handle = ntupleDict.get('_handle', {}).get(branchName)
                    label  = ntupleDict.get('_label' , {}).get(branchName)
                    if not handle or not label:
                        raise RuntimeError, "Missing handle or label for '%s'"\
                              % branchName
                    if not eventTree.getByLabel (label, handle):
                        raise RuntimeError, "not able to get %s for %s" \
                              % (label, branchName)
                    objects = handle.product()
            # is this a singleton?
            if GenObject.isSingleton (objName):
                event[objName] = GenObject.\
                                 _genObjectClone (objName,
                                                  tupleName,
                                                  objects)
                continue
            # if we're here then we have a vector of items
            if debug: warn (objName, spaces = 3)
            event[objName] = []
            for index, obj in enumerate (objects):
                event[objName].append( GenObject.\
                                       _genObjectClone (objName,
                                                        tupleName,
                                                        obj,
                                                        index) )
            del objects
            # end for obj        
        ## if not isChain:
        ##     del rootEvent
        # end for objName
        return event


    @staticmethod
    def printEvent (event):
        """Prints out event dictionary.  Mostly for debugging"""
        # Print out all singletons first
        for objName, obj in sorted (event.iteritems()):
            #obj = event[objName]
            # is this a singleton?
            if GenObject.isSingleton (objName):
                print "%s: %s" % (objName, obj)
        # Now print out all vectors
        for objName, obj in sorted (event.iteritems()):
            #obj = event[objName]
            # is this a singleton?
            if not GenObject.isSingleton (objName):
                # o.k. obj is a vector
                print "%s:" % objName
                for single in obj:
                    print "  ", single
        print


    @staticmethod
    def setAliases (eventTree, tupleName):
        """runs SetAlias on all saved aliases"""
        aliases = GenObject._ntupleDict[tupleName].get('_alias', {})
        for name, alias in aliases.iteritems():
            eventTree.SetAlias (name, alias)


    @staticmethod
    def changeAlias (tupleName, name, alias):
        """Updates an alias for an object for a given tuple"""
        aliasDict = GenObject._ntupleDict[tupleName]['_alias']
        if not aliasDict.has_key (name):
            raise RuntimeError, "unknown name '%s' in tuple '%s'" % \
                  (name, tupleName)
        aliasDict[name] = alias


    @staticmethod
    def changeLabel (tupleName, objectName, label):
        """Updates an label for an object for a given tuple"""
        labelDict = GenObject._ntupleDict[tupleName]['_label']
        if not labelDict.has_key (objectName):
            raise RuntimeError, "unknown name '%s' in tuple '%s'" % \
                  (objectName, tupleName)
        label = tuple( GenObject._commaRE.split( label ) )
        labelDict[objectName] = label


    @staticmethod
    def prepareTuple (tupleName, files, numEventsWanted = 0):
        """Given the tuple name and list of files, returns either a
        TChain or EventTree, and number of entries"""
        if "GenObject" == tupleName:
            GenObject.prepareToLoadGenObject()            
        if not isinstance (files, list):
            # If this isn't a list, make it one
            files = [files]
            ntupleDict = GenObject._ntupleDict[tupleName]
        treeName = ntupleDict["_tree"]
        if ntupleDict.get('_useChain'):
            chain = ROOT.TChain (treeName)
            for filename in files:
                chain.AddFile (filename)
                numEntries = chain.GetEntries()
        # Are we using a chain or EventTree here?
        else:
            chain = Events (files, forceEvent=True)
            numEntries = chain.size()
        chainDict = GenObject._kitchenSinkDict.setdefault (chain, {})
        if numEventsWanted and numEventsWanted < numEntries:
            numEntries = numEventsWanted
        chainDict['numEntries'] = numEntries
        chainDict['tupleName' ] = tupleName
        return chain


    @staticmethod
    def getRunEventEntryDict (chain, tupleName, numEntries):
        """Returns a dictionary of run, event tuples to entryIndicies"""
        reeDict = {}
        for entryIndex in xrange (numEntries):
            event = GenObject.loadEventFromTree (chain,
                                                 entryIndex,
                                                 onlyRunEvent = True)
            runevent = event['runevent']
            reeDict[ GenObject._re2key (runevent) ] = entryIndex
            #reeDict[ "one two three" ] = entryIndex
            del event
        return reeDict


    @staticmethod
    def _re2key (runevent):
        """Given a GO 'runevent' object, returns a sortable key"""
        # if we don't know how to make this object yet, let's figure
        # it out
        if not GenObject._runEventListDone:
            GenObject._runEventListDone = True
            ignoreSet = set( ['run', 'event'] )
            for varName in sorted (runevent.__dict__.keys()):
                if varName.startswith ('_') or varName in ignoreSet:
                    continue
                form = runevent.getVariableProperty (varName, "form")
                if not form:
                    form = '%s'                
                GenObject._runEventList.append ((varName, form))
        key = 'run:%d event:%d' % (runevent.run, runevent.event)
        for items in GenObject._runEventList:
            varName = items[0]
            form    = ' %s:%s' % (varName, items[1])
            key += form % runevent.getVariableProperty (varName)
        return key


    @staticmethod
    def _key2re (key, runevent=None):
        """Given a key, returns a GO 'runevent' object"""
        if not runevent:
            runevent = GenObject ('runevent')
        words = GenObject._spacesRE.split (key)
        for word in words:
            match = GenObject._singleColonRE.search (word)
            if match:
                # for now, I'm assuming everything in the runevent
                # tuple is an integer.  If this isn't the case, I'll
                # have to come back and be more clever here.
                runevent.__setattr__ (match.group(1), int( match.group(2) ))
        return runevent
                                     

    @staticmethod
    def compareRunEventDicts (firstDict, secondDict):
        """Compares the keys of the two dicts and returns three sets:
        the overlap, first but not second, and second but not first."""
        overlap    = set()
        firstOnly  = set()
        secondOnly = set()
        # loop over the keys of the first dict and compare to second dict
        for key in firstDict.keys():
            if secondDict.has_key (key):
                overlap.add (key)
            else:
                firstOnly.add (key)
        # now loop over keys of second dict and only check for missing
        # entries in first dict
        for key in secondDict.keys():
            if not firstDict.has_key (key):
                secondOnly.add (key)
        # All done
        return overlap, firstOnly, secondOnly


    @staticmethod
    def pairEquivalentObjects (vec1, vec2):
        """Finds the equivalent objects in the two vectors"""
        len1, len2 = len (vec1), len (vec2)
        debug = GenObject._kitchenSinkDict.get ('debug', False)        
        if not len1 or not len2:
            # Nothing to see here folks.  Keep moving.
            if len1:
                noMatch1Set = set( xrange(len1) )
            else:
                noMatch1Set = set ()
            if len2:
                noMatch2Set = set( xrange(len2) )
            else:
                noMatch2Set = set ()
            if debug: warn ("Nothing found", sapces=6)
            return set(), noMatch1Set, noMatch2Set
        objName = vec1[0]._objName
        equivList = GenObject._equivDict[objName]
        firstDict = {}
        secondDict = {}
        # First, look for vec2 objects that are equivalent to a
        # given vec1 object.
        for index1 in xrange (len1):
            objList = []
            obj1 = vec1[index1]
            for index2 in xrange (len2):
                total = 0.
                obj2 = vec2[index2]
                ok = True
                for equiv in equivList:
                    var, precision = equiv[0], equiv[1]
                    val1 = obj1 (var)
                    val2 = obj2 (var)
                    # Do we check equality or a precision
                    if precision:
                        value = abs (val1 - val2) / precision
                        if value >= 1.:
                            ok = False
                            break
                        total += value ** 2
                    elif val1 != val2:
                        ok = False
                        break
                if ok:
                    objList.append( (total, index2) )
            objList.sort()
            firstDict[index1] = objList
        # Now do the same thing, but this time look for vec1 objects
        # that are equivalent to a given vec2 object
        for index2 in xrange (len2):
            objList = []
            obj2 = vec2[index2]
            for index1 in xrange (len1):
                total = 0.
                obj1 = vec1[index1]
                ok = True
                for equiv in equivList:
                    var, precision = equiv[0], equiv[1]
                    val2 = obj2 (var)
                    val1 = obj1 (var)
                    # Do we check equality or a precision
                    if precision:
                        value = abs (val2 - val1) / precision
                        if value > 1.:
                            ok = False
                            break
                        total += value ** 2
                    elif val2 != val1:
                        ok = False
                        break
                if ok:
                    objList.append( (total, index1) )
            objList.sort()
            secondDict[index2] = objList
        # O.k. Now that we have the candidate matches, lets see who is
        # really matched.
        matchedSet = set()
        noMatch1Set = set()
        firstDictKeys = sorted (firstDict.keys())
        for index1 in firstDictKeys:
            list1 = firstDict[index1]
            # do I have a match?
            if not len (list1):
                # no match
                noMatch1Set.add (index1)
                continue
            # we've got at least one match
            best1 = list1[0]
            index2 = best1[1]
            # Does this one match me?
            list2 = secondDict.get (index2, [])
            if len(list2) and list2[0][1] == index1:
                matchedSet.add( (index1, index2) )
                # get rid of the 2nd key hash
                del firstDict[index1]
                del secondDict[index2]
            else:
                # no match
                noMatch1Set.add (index1)
        noMatch2Set = set( secondDict.keys() )
        return matchedSet, noMatch1Set, noMatch2Set


    @staticmethod
    def compareTwoItems (item1, item2):
        """Compares all of the variables making sure they are the same
        on the two objects."""
        objName = item1._objName
        problems = {}
        relative = GenObject._kitchenSinkDict.get ('relative', False)
        for varName in GenObject._objsDict[objName].keys():
            prec = item1.getVariableProperty (varName, 'prec')
            if prec:
                # we want to check within a precision
                if relative:
                    val1 = item1(varName)
                    val2 = item2(varName)
                    numerator = 2 * abs (val1 - val2)
                    denominator = abs(val1) + abs(val2)
                    if not denominator:
                        # both are exactly zero, so there's no
                        # disagreement here.
                        continue
                    value = numerator / denominator
                    if value > prec:
                        # we've got a problem
                        problems[varName] = value                    
                else:
                    value = abs( item1(varName) - item2(varName) )
                    if value > prec:
                        # we've got a problem
                        problems[varName] = value
            else:
                # we want to check equality
                if item1(varName) != item2(varName):
                    # we have a problem.  sort the values
                    val1, val2 = item1(varName), item2(varName)
                    if val1 > val2:
                        val1, val2 = val2, val1
                    problems[varName] = "%s != %s" % (val1, val2)
        # end for
        return problems


    @staticmethod
    def blurEvent (event, value, where = ""):
        """For debugging purposes only.  Will deliberately change
        values of first tree to verify that script is correctly
        finding problems."""
        for objName in sorted (event.keys()):
            if "runevent" == objName:
                # runevent is a special case.  We don't compare these
                continue
            if GenObject.isSingleton (objName):
                # I'll add this in later.  For now, just skip it
                continue
            count = 0
            for obj in event[objName]:
                count += 1
                for varName in GenObject._objsDict[objName].keys():
                    if isinstance (obj.__dict__[varName], str):
                        # don't bother
                        continue
                    randNumber = random.random()
                    #print "rN", randNumber
                    if randNumber < GenObject._kitchenSinkDict['blurRate']:
                        print "  %s: changing '%s' of '%s:%d'" \
                              % (where, varName, obj._objName, count)
                        ## print "objdict", obj.__dict__.get(varName), ':',\
                        ##       value
                        obj.__dict__[varName] += value


    @staticmethod
    def compareTwoTrees (chain1, chain2, **kwargs):
        """Given all of the necessary information, this routine will
        go through and compare two trees making sure they are
        'identical' within requested precision.  If 'diffOutputName'
        is passed in, a root file with a diffTree and missingTree will
        be produced."""
        print "Comparing Two Trees"
        diffOutputName = kwargs.get ('diffOutputName')
        tupleName1  = GenObject._kitchenSinkDict[chain1]['tupleName']
        numEntries1 = GenObject._kitchenSinkDict[chain1]['numEntries']
        tupleName2  = GenObject._kitchenSinkDict[chain2]['tupleName']
        numEntries2 = GenObject._kitchenSinkDict[chain2]['numEntries']
        debug       = GenObject._kitchenSinkDict.get ('debug', False)
        ree1 = GenObject.getRunEventEntryDict (chain1, tupleName1, numEntries1)
        ree2 = GenObject.getRunEventEntryDict (chain2, tupleName2, numEntries2)
        overlap, firstOnly, secondOnly = \
                 GenObject.compareRunEventDicts (ree1, ree2)
        if diffOutputName:
            rootfile, diffTree, missingTree = \
                      GenObject.setupDiffOutputTree (diffOutputName,
                                                     'diffTree',
                                                     'missingTree')
            if firstOnly:
                vec = GenObject._rootClassDict['firstOnly']
                for key in firstOnly:
                    runevent = GenObject._key2re (key)
                    vec.push_back( GenObject._rootObjectClone( runevent ) )
            if secondOnly:
                vec = GenObject._rootClassDict['secondOnly']
                for key in secondOnly:
                    runevent = GenObject._key2re (key)
                    vec.push_back( GenObject._rootObjectClone( runevent ) )
            missingTree.Fill()
        resultsDict = {}
        if firstOnly:
            resultsDict.setdefault ('_runevent', {})['firstOnly'] = \
                                   len (firstOnly)
        if secondOnly:
            resultsDict.setdefault ('_runevent', {})['secondOnly'] = \
                                   len (secondOnly)
        resultsDict['eventsCompared'] = len (overlap)
        for reTuple in sorted(overlap):
            # if we are filling the diff tree, then save the run and
            # event information.
            if diffOutputName:
                GenObject._key2re (reTuple,
                                   GenObject._rootClassDict['runevent'])
            #print 'retuple', reTuple
            if debug: warn ('event1', blankLines = 3)
            event1 = GenObject.loadEventFromTree (chain1, ree1 [reTuple])
            if debug: warn ('event2', blankLines = 3)
            event2 = GenObject.loadEventFromTree (chain2, ree2 [reTuple])
            if GenObject._kitchenSinkDict.get('printEvent'):
                print "event1:"
                GenObject.printEvent (event1)
                print "event2:"
                GenObject.printEvent (event2)
            if GenObject._kitchenSinkDict.get('blur'):
                where = reTuple
                GenObject.blurEvent (event1,
                                     GenObject._kitchenSinkDict['blur'],
                                     where)
            for objName in sorted (event1.keys()):
                if "runevent" == objName:
                    # runevent is a special case.  We don't compare these
                    continue
                if not GenObject._equivDict.get (objName):
                    # we don't know how to compare these objects, so
                    # skip them.
                    continue
                if GenObject.isSingleton (objName):
                    # I'll add this in later.  For now, just skip it
                    continue
                # Get ready to calculate root diff object if necessary
                rootObj = 0
                if diffOutputName:
                    rootObj = GenObject._rootObjectDict[objName]
                    rootObj.clear()
                vec1 = event1[objName]
                vec2 = event2[objName]
                matchedSet, noMatch1Set, noMatch2Set = \
                            GenObject.pairEquivalentObjects (vec1, vec2)
                if noMatch1Set or noMatch2Set:
                    ## print "No match 1", noMatch1Set
                    ## print "No match 2", noMatch2Set
                    count1 = len (noMatch1Set)
                    count2 = len (noMatch2Set)
                    key = (count1, count2)
                    countDict = resultsDict.\
                                setdefault (objName, {}).\
                                setdefault ('_missing', {})
                    if countDict.has_key (key):
                        countDict[key] += 1
                    else:
                        countDict[key] = 1
                    # should be calculating root diff objects
                    if diffOutputName:
                        # first set
                        for index in sorted(list(noMatch1Set)):
                            goObj = vec1 [index]
                            rootObj.firstOnly.push_back ( GenObject.\
                                                          _rootObjectClone \
                                                          (goObj) )
                        # second set
                        for index in sorted(list(noMatch2Set)):
                            goObj = vec2 [index]
                            rootObj.secondOnly.push_back ( GenObject.\
                                                          _rootObjectClone \
                                                           (goObj) )
                # o.k.  Now that we have them matched, let's compare
                # the proper items:                
                for pair in sorted(list(matchedSet)):
                    if diffOutputName:
                        rootObj.diff.push_back ( GenObject._rootDiffObject \
                                                 ( vec1[ pair[1 - 1] ],
                                                   vec2[ pair[2 - 1] ] ) )
                    problems = GenObject.\
                               compareTwoItems (vec1[ pair[1 - 1] ],
                                                vec2[ pair[2 - 1] ])
                    if problems.keys():
                        # pprint.pprint (problems)
                        for varName in problems.keys():
                            countDict = resultsDict.\
                                        setdefault (objName, {}).\
                                        setdefault ('_var', {})
                            if countDict.has_key (varName):
                                countDict[varName] += 1
                            else:
                                countDict[varName] = 1
                key = 'count_%s' % objName
                if not resultsDict.has_key (key):
                    resultsDict[key] = 0
                resultsDict[key] += len (matchedSet)
                # try cleaning up
                del vec1
                del vec2
            # end for objName        
            if diffOutputName:
                diffTree.Fill()
            del event1
            del event2
        # end for overlap
        if diffOutputName:
            diffTree.Write()
            missingTree.Write()
            rootfile.Close()
        return resultsDict


    @staticmethod
    def saveTupleAs (chain, rootFile):
        """Saves a chain as a GO tree"""
        print "saveTupleAs"
        rootfile, tree = GenObject.setupOutputTree (rootFile, "goTree")
        numEntries = GenObject._kitchenSinkDict[chain]['numEntries']        
        for entryIndex in xrange (numEntries):
            event = GenObject.loadEventFromTree (chain, entryIndex)            
            if GenObject._kitchenSinkDict.get('blur'):
                where = "run %d event %d" % (event['runevent'].run,
                                             event['runevent'].event)
                if random.random() < GenObject._kitchenSinkDict.get('blur'):
                    # dropping event
                    print "Dropping", where
                    continue
                GenObject.blurEvent (event,
                                     GenObject._kitchenSinkDict['blur'],
                                     where)
                # check to see if we should drop the event
            if GenObject._kitchenSinkDict.get('printEvent'):
                GenObject.printEvent (event)
            GenObject._fillRootObjects (event)
            tree.Fill()
        tree.Write()
        rootfile.Close()


    @staticmethod
    def setGlobalFlag (key, value):
        """Sets a global flag in _kitchenSinkDict"""
        GenObject._kitchenSinkDict [key] = value


    @staticmethod
    def printTuple (chain):
        """Prints out all events to stdout"""
        numEntries = GenObject._kitchenSinkDict[chain]['numEntries']
        debug = GenObject._kitchenSinkDict.get ('debug', False)
        if debug: warn (numEntries)
        for entryIndex in xrange (numEntries):
            if debug: warn (entryIndex, spaces=3)
            event = GenObject.loadEventFromTree (chain, entryIndex)            
            GenObject.printEvent (event)
            if debug: warn(spaces=3)

    @staticmethod
    def _convertStringToParameters (string):
        """Convert comma-separated string into a python list of
        parameters.  Currently only understands strings, floats,  and
        integers."""
        retval = []        
        words = GenObject._commaRE.split (string)
        for word in words:
            if not len (word):
                continue
            match = GenObject._singleQuoteRE.search (word)
            if match:
                retval.append (match.group (1))
                continue
            match = GenObject._doubleQuoteRE.search (word)
            if match:
                retval.append (match.group (1))
                continue
            try:
                val = int (word)
                retval.append (val)
                continue
            except:
                pass
            try:
                val = float (word)
                retval.append (val)
                continue
            except:
                pass
            # if we're still here, we've got a problem
            raise RuntimeError, "Unknown parameter '%s'." % word
        return retval

        
    ######################
    ## Member Functions ##
    ######################


    def __init__ (self, objName):
        """Class initializer"""
        if not GenObject._objsDict.has_key (objName):# or \
            #not GenObject._equivDict.has_key (objName) :
            # not good
            print "Error: GenObject does not know about object '%s'." % objName
            raise RuntimeError, "Failed to create GenObject object."
        self._localObjsDict = GenObject._objsDict [objName]
        self._objName = objName;
        for key, varDict in self._localObjsDict.iteritems():
            # if the key starts with an '_', then it is not a
            # variable, so don't treat it as one.
            if key.startswith ("_"):
                continue
            self.setValue (key, varDict['default'])
            

    def setValue (self, name, value):
        """Wrapper for __setattr___"""
        self.__setattr__ (name, value)

    
    def getVariableProperty (self, var, key):
        """ Returns property assoicated with 'key' for variable 'var'
        of object of the same type as 'self'.  Returns 'None' if 'var'
        or 'key' is not defined."""
        return GenObject._objsDict.get (self._objName,
                                        {}).get (var, {}). get (key, None)


    def __setattr__ (self, name, value):
        """Controls setting of values."""
        if name.startswith ("_"):
            # The internal version. Set anything you want.
            object.__setattr__ (self, name, value)
        else:
            # user version - Make sure this variable has already been

            # defined for this type:
            if not self._localObjsDict.has_key (name):
                # this variable has not been defined
                print "Warning: '%s' for class '%s' not setup. Skipping." % \
                      (name, self._objName)
                return
            varType = self.getVariableProperty (name, 'varType')
            # if this is an int, make sure it stays an int
            if GenObject.types.int == varType:
                try:
                    # This will work with integers, floats, and string
                    # representations of integers.
                    value = int (value)
                except:
                    # This works with string representations of floats
                    value = int( float( value ) )
            elif GenObject.types.long == varType:
                try:
                    # This will work with integers, floats, and string
                    # representations of integers.
                    value = long (value)
                except:
                    # This works with string representations of floats
                    value = long( float( value ) )
            elif GenObject.types.float == varType:
                # Make sure it's a float
                value = float (value)
            elif GenObject.types.string == varType:
                # make sure it's a string
                value = str (value)
            # if we're still here, set it
            object.__setattr__ (self, name, value)


    def __call__ (self, key):
        """Makes object callable"""
        return object.__getattribute__ (self, key)


    def __str__ (self):
        """String representation"""
        retval = ""
        for varName, value in sorted (self.__dict__.iteritems()):
            if varName.startswith ('_'): continue
            form = self.getVariableProperty (varName, "form")
            if form:
                format = "%s:%s  " % (varName, form)
                retval = retval + format % value
            else:
                retval = retval + "%s:%s  " % (varName, value)
        return retval

