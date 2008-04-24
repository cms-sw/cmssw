Traceback (most recent call last):
  File "/afs/cern.ch/cms/sw/ReleaseCandidates/slc4_ia32_gcc345/mon/2.1-mon-16/CMSSW_2_1_X_2008-04-21-1600/src/FWCore/ParameterSet/python/cfg2py.py", line 10, in ?
    print cmsParse.dumpCff(fileInPath)
  File "/afs/cern.ch/cms/sw/ReleaseCandidates/slc4_ia32_gcc345/mon/2.1-mon-16/CMSSW_2_1_X_2008-04-21-1600/python/FWCore/ParameterSet/parseConfig.py", line 1600, in dumpCff
    compressedValues = _getCompressedNodes(fileName, 0, values)
  File "/afs/cern.ch/cms/sw/ReleaseCandidates/slc4_ia32_gcc345/mon/2.1-mon-16/CMSSW_2_1_X_2008-04-21-1600/python/FWCore/ParameterSet/parseConfig.py", line 1359, in _getCompressedNodes
    raise pp.ParseFatalException(s,loc,"the process contains the error \n"+str(e))
FWCore.ParameterSet.parsecf.pyparsing.ParseFatalException: the process contains the error 
include file RecoTBCalo/EcalTBTDCReconstructor/data/Ecal2004TBTDCRanges.cff had the parsing error 
Expected "source" (at char 5), (line:1, col:6) (at char 0), (line:1, col:1)
