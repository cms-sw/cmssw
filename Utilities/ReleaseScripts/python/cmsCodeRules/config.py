__author__="Aurelija"
__date__ ="$2010-08-06 14.27.51$"

import os
from Utilities.ReleaseScripts.commentSkipper.commentSkipper import filter as comment
from Utilities.ReleaseScripts.cmsCodeRules.cppFunctionSkipper import filterFiles as function
ordering = ['1', '2', '3', '4', '5', '6']

# default values for directories

checkPath = os.getcwd()
picklePath = os.getcwd()
txtPath = os.getcwd()
htmlPath = os.getcwd()

# exception for directories and files 

exceptPaths = []

# --------------------------------------------------------------------------------
# configuration info for each rule ...

rulesNames = []
Configuration = {}

# --------------------------------------------------------------------------------

# configuration for rule 1

ruleName = '1'
rulesNames.append(ruleName)
Configuration[ruleName] = {}

Configuration[ruleName]['description'] = 'Search for "using namespace" or "using std::" in header files'
Configuration[ruleName]['filesToMatch'] = ['*.h']
Configuration[ruleName]['exceptPaths'] = []
Configuration[ruleName]['skip']  = [comment, function]
Configuration[ruleName]['filter'] = '(\susing|\Ausing)\s+(namespace|std::)' #should be regular expression
Configuration[ruleName]['exceptFilter'] = []

# --------------------------------------------------------------------------------

# configuration for rule 2

ruleName = '2'
rulesNames.append(ruleName)
Configuration[ruleName] = {}

Configuration[ruleName]['description'] = 'Search for CXXFLAGS flags that are set to -g or -O0 in BuildFile'
Configuration[ruleName]['filesToMatch'] = ['BuildFile', 'BuildFile.xml']
Configuration[ruleName]['exceptPaths'] = []
Configuration[ruleName]['skip']  = [comment]
Configuration[ruleName]['filter'] = '\s(CXXFLAGS|CPPFLAGS)(\+|=|\w|\"|\'|-|\s)*(-g|-O0)(\s|\'|\")' #should be regular expression
Configuration[ruleName]['exceptFilter'] = []

# --------------------------------------------------------------------------------

# configuration for rule 3

ruleName = '3'
rulesNames.append(ruleName)
Configuration[ruleName] = {}

Configuration[ruleName]['description'] = 'Search for "catch(...)" statements in *.cc, *.cxx files'
Configuration[ruleName]['filesToMatch'] = ['*.cc', '*.cxx']
Configuration[ruleName]['exceptPaths'] = ['FWCore/*', 'EventFilter/*', '*/*/test/*', '*/*/bin/*']
Configuration[ruleName]['skip']  = [comment]
Configuration[ruleName]['filter'] = 'catch\s*\(\s*\.\.\.\s*\)' #should be regular expression
Configuration[ruleName]['exceptFilter'] = []
# --------------------------------------------------------------------------------

# configuration for rule 4

ruleName = '4'
rulesNames.append(ruleName)
Configuration[ruleName] = {}

Configuration[ruleName]['description'] = 'Search for "copyright" declaration in *.c, *.cc, *.cxx, *.h files'
Configuration[ruleName]['filesToMatch'] = ['*.h', '*.c', '*.cc', '*.cxx']
Configuration[ruleName]['exceptPaths'] = [
                                           'FWCore/Utilities/interface/math_private.h',
                                           'FWCore/Utilities/interface/md5.h',
                                           'FWCore/Utilities/interface/tinyxml.h',
                                           'FWCore/Utilities/src/md5.c',
                                           'FWCore/Utilities/src/tinyxml.cc',
                                           'FWCore/Utilities/src/tinyxmlerror.cc',
                                           'FWCore/Utilities/src/tinyxmlparser.cc',
                                           'DataFormats/Common/interface/Trie.h',
                                           'DataFormats/Math/interface/sse_mathfun.h',
                                           'EventFilter/Processor/src/procUtils.cc',
					   'CondCore/ORA/src/GenMD5.cc',
                                           'PhysicsTools/JetMCUtils/interface/combination.h',
                                         ] #could be file name, dir, fileName:line. But path should be only from that directory in which we are searching
Configuration[ruleName]['skip']  = []
Configuration[ruleName]['filter'] = '(\A|\W)(c|C)(o|O)(p|P)(y|Y)(r|R)(i|I)(g|G)(h|H)(t|T)\W(\+|=|\w|\"|\'|-|\s)*(\((c|C)\)|\d{4})' #should be regular expression
Configuration[ruleName]['exceptFilter'] = []

# --------------------------------------------------------------------------------

# configuration for rule 5

ruleName = '5'
rulesNames.append(ruleName)
Configuration[ruleName] = {}

Configuration[ruleName]['description'] = 'Search for "pragma" statement in *.c, *.cc, *.cxx, *.h files'
Configuration[ruleName]['filesToMatch'] = ['*.h', '*.c', '*.cc', '*.cxx']
Configuration[ruleName]['exceptPaths'] = ['*/*LinkDef.h',
                                          'FWCore/Utilities/interface/tinyxml.h',
                                          'Utilities/StorageFactory/src/LocalFileSystem.cc:.*:#\s*pragma\s+GCC\s+diagnostic\s+ignored',
                                          'RecoVertex/BeamSpotProducer/test/scripts/BSVectorDict.h',
                                          'FWCore/Utilities/*/*:.*:#\s*pragma\s+GCC\s+visibility\s+(push\\(default\\)|pop)\s*$',
                                          'CondFormats/JetMETObjects/interface/Linkdef.h',
                                          'DataFormats/GeometryVector/interface/Basic3DVectorLD.h',
                                          'DataFormats/Scalers/interface/ScalersRaw.h',
                                          'FWCore/MessageService/plugins/Module.cc',
                                          'GeneratorInterface/RivetInterface/plugins/tinyxml.h',
                                          'PerfTools/Callgrind/plugins/CallgrindAnalyzer.cc',
                                          'PerfTools/Callgrind/src/ProfilerService.cc',
                                          'RecoTracker/TkDetLayers/src/BladeShapeBuilderFromDet.h',
                                          'RecoTracker/TkDetLayers/src/BoundDiskSector.h',
                                          'RecoTracker/TkDetLayers/src/CompatibleDetToGroupAdder.h',
                                          'RecoTracker/TkDetLayers/src/CompositeTECPetal.h',
                                          'RecoTracker/TkDetLayers/src/CompositeTECWedge.h',
                                          'RecoTracker/TkDetLayers/src/DetGroupMerger.h',
                                          'RecoTracker/TkDetLayers/src/DiskSectorBounds.h',
                                          'RecoTracker/TkDetLayers/src/ForwardDiskSectorBuilderFromDet.h',
                                          'RecoTracker/TkDetLayers/src/ForwardDiskSectorBuilderFromWedges.h',
                                          'RecoTracker/TkDetLayers/src/GlobalDetRodRangeZPhi.h',
                                          'RecoTracker/TkDetLayers/src/LayerCrossingSide.h',
                                          'RecoTracker/TkDetLayers/src/PixelBarrelLayerBuilder.h',
                                          'RecoTracker/TkDetLayers/src/PixelBarrelLayer.h',
                                          'RecoTracker/TkDetLayers/src/PixelBladeBuilder.h',
                                          'RecoTracker/TkDetLayers/src/PixelBlade.h',
                                          'RecoTracker/TkDetLayers/src/PixelForwardLayerBuilder.h',
                                          'RecoTracker/TkDetLayers/src/PixelForwardLayer.h',
                                          'RecoTracker/TkDetLayers/src/PixelForwardLayerPhase1.h',
                                          'RecoTracker/TkDetLayers/src/PixelRodBuilder.h',
                                          'RecoTracker/TkDetLayers/src/PixelRod.h',
                                          'RecoTracker/TkDetLayers/src/SimpleTECWedge.h',
                                          'RecoTracker/TkDetLayers/src/SubLayerCrossings.h',
                                          'RecoTracker/TkDetLayers/src/TECLayerBuilder.h',
                                          'RecoTracker/TkDetLayers/src/TECLayer.h',
                                          'RecoTracker/TkDetLayers/src/TECPetalBuilder.h',
                                          'RecoTracker/TkDetLayers/src/TECPetal.h',
                                          'RecoTracker/TkDetLayers/src/TECWedgeBuilder.h',
                                          'RecoTracker/TkDetLayers/src/TECWedge.h',
                                          'RecoTracker/TkDetLayers/src/TIBLayerBuilder.h',
                                          'RecoTracker/TkDetLayers/src/TIBLayer.h',
                                          'RecoTracker/TkDetLayers/src/TIBRingBuilder.h',
                                          'RecoTracker/TkDetLayers/src/TIBRing.h',
                                          'RecoTracker/TkDetLayers/src/TIDLayerBuilder.h',
                                          'RecoTracker/TkDetLayers/src/TIDLayer.h',
                                          'RecoTracker/TkDetLayers/src/TIDRingBuilder.h',
                                          'RecoTracker/TkDetLayers/src/TkDetUtil.h',
                                          'RecoTracker/TkDetLayers/src/TkGeomDetCompatibilityChecker.h',
                                          'RecoTracker/TkDetLayers/src/TOBLayerBuilder.h',
                                          'RecoTracker/TkDetLayers/src/TOBLayer.h',
                                          'RecoTracker/TkDetLayers/src/TOBRodBuilder.h',
                                          'RecoTracker/TkDetLayers/src/TOBRod.h',
                                          'RecoTracker/TkDetLayers/src/TIDRing.h',
                                         ]#could be file name, dir, fileName:line. Path should be only from that directory in which we are searching
Configuration[ruleName]['skip']  = [comment]
Configuration[ruleName]['filter'] = '#\s*pragma\s' #should be regular expression
Configuration[ruleName]['exceptFilter'] = []
# --------------------------------------------------------------------------------
# configuration for rule 6

ruleName = '6'
rulesNames.append(ruleName)
Configuration[ruleName] = {}

Configuration[ruleName]['description'] = 'Search for "flags" statements in BuildFile'
Configuration[ruleName]['filesToMatch'] = ['BuildFile', 'BuildFile.xml']
Configuration[ruleName]['exceptPaths'] = ['Utilities/RFIOAdaptor/*BuildFile.xml:.*:NO_LIB_CHECKING=',  #could be file name, dir, fileName:line, fileName:lineNORegEx:LineRegEx:. Path should be only from that directory in  which we are searching
					  'Utilities/RFIOAdaptor/*BuildFile.xml:.*:_FILE_OFFSET_BITS=',
					  'Utilities/DCacheAdaptor/*BuildFile.xml:.*:_FILE_OFFSET_BITS=',
					  'Utilities/XrdAdaptor/*BuildFile.xml:.*:_FILE_OFFSET_BITS=',
					  'Utilities/StorageFactory/*BuildFile.xml:.*:_FILE_OFFSET_BITS=',
                                          'Utilities/LStoreAdaptor/BuildFile.xml:.*:_FILE_OFFSET_BITS=',
					  'DQM/CSCMonitorModule/*BuildFile.xml:.*:="DQMGLOBAL"',
					  'DQMServices/Core/*BuildFile.xml:.*:="-DWITHOUT_CMS_FRAMEWORK=0"',
                                          'L1Trigger/CSCTrackFinder/BuildFile.xml:.*:ADD_SUBDIR=',
                                          'MagneticField/Interpolation/BuildFile.xml:.*:="-Wno-format"',
                                          'MagneticField/Interpolation/test/BuildFile.xml:.*:="-Wno-format"',
                                          'GeneratorInterface/Pythia6Interface/plugins/BuildFile.xml:.*:LDFLAGS="\$\(PYTHIA6_BASE\)/lib/pydata.o"',
					 ]
Configuration[ruleName]['skip']  = [comment]
Configuration[ruleName]['filter'] = '<\s*(f|F)(l|L)(a|A)(g|G)(s|S)\s+' #should be regular expression
Configuration[ruleName]['exceptFilter'] = ['EDM_PLUGIN','RIVET_PLUGIN', 'GENREFLEX_ARGS', 'TEST_RUNNER_ARGS', 'INSTALL_SCRIPTS', 'NO_TESTRUN', 'NO_EXPORT']
# --------------------------------------------------------------------------------

rulesDescription  = "Rule number    Description\n"
rulesDescription += "----------------------------------------------------------------------------------------\n"
for key, value in Configuration.items():
    rulesDescription += "     %s         %s\n" %(key, value['description'])

# --------------------------------------------------------------------------------
helpMsg  = "-----------------------------------------------------------HELP-----------------------------------------------------------\n"
helpMsg += "cmsCodeRulesChecker.py [-h] [-html] [-s [DIRECTORY]] [-S [DIRECTORY]] [-p] [-r ruleNumber[,ruleNumber[, ...]]] [-d DIRECTORY]\n\n"
helpMsg += "-r     Specifies rule or rules to be checked. After this parameter should\n       be at least one rule given.\n"
helpMsg += "-d     Specifies that rules should be checked in DIRECTORY. Default \n       directory - current directory\n"
helpMsg += "-S     Specifies to save results in python pickle files. Directory specifies\n       where to store these files. Default directory - current directory\n"
helpMsg += "-s     Specifies to save results in .txt files. Directory specifies where to\n       store these files. Default directory - current directory\n"
helpMsg += "-p     Specifies to print results into a screen\n"
helpMsg += "-h     Prints help message\n"
helpMsg += "-html  Reads pickle files and creates cmsCRPage.html\n\n"
helpMsg += "By default cmsCodeRulesChecker.py checks all rules in current directory and prints results into screen.\n\n"
helpMsg += rulesDescription
