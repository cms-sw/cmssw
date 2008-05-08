Traceback (most recent call last):
  File "/afs/cern.ch/cms/sw/ReleaseCandidates/slc4_ia32_gcc345/thu/2.1-thu-02/CMSSW_2_1_X_2008-05-08-0200/src/FWCore/ParameterSet/python/cfg2py.py", line 8, in ?
    print cmsParse.dumpCfg(fileInPath)
  File "/afs/cern.ch/cms/sw/ReleaseCandidates/slc4_ia32_gcc345/thu/2.1-thu-02/CMSSW_2_1_X_2008-05-08-0200/python/FWCore/ParameterSet/parseConfig.py", line 1600, in dumpCfg
    return cfgDumper.parseFile(_fileFactory(fileName))[0]
  File "/afs/cern.ch/cms/sw/ReleaseCandidates/slc4_ia32_gcc345/thu/2.1-thu-02/CMSSW_2_1_X_2008-05-08-0200/python/FWCore/ParameterSet/parsecf/pyparsing.py", line 990, in parseFile
    return self.parseString(file_contents)
  File "/afs/cern.ch/cms/sw/ReleaseCandidates/slc4_ia32_gcc345/thu/2.1-thu-02/CMSSW_2_1_X_2008-05-08-0200/python/FWCore/ParameterSet/parsecf/pyparsing.py", line 770, in parseString
    loc, tokens = self._parse( instring.expandtabs(), 0 )
  File "/afs/cern.ch/cms/sw/ReleaseCandidates/slc4_ia32_gcc345/thu/2.1-thu-02/CMSSW_2_1_X_2008-05-08-0200/python/FWCore/ParameterSet/parsecf/pyparsing.py", line 663, in _parseNoCache
    loc,tokens = self.parseImpl( instring, preloc, doActions )
  File "/afs/cern.ch/cms/sw/ReleaseCandidates/slc4_ia32_gcc345/thu/2.1-thu-02/CMSSW_2_1_X_2008-05-08-0200/python/FWCore/ParameterSet/parsecf/pyparsing.py", line 1810, in parseImpl
    loc, resultlist = self.exprs[0]._parse( instring, loc, doActions )
  File "/afs/cern.ch/cms/sw/ReleaseCandidates/slc4_ia32_gcc345/thu/2.1-thu-02/CMSSW_2_1_X_2008-05-08-0200/python/FWCore/ParameterSet/parsecf/pyparsing.py", line 689, in _parseNoCache
    tokens = fn( instring, tokensStart, retTokens )
  File "/afs/cern.ch/cms/sw/ReleaseCandidates/slc4_ia32_gcc345/thu/2.1-thu-02/CMSSW_2_1_X_2008-05-08-0200/python/FWCore/ParameterSet/parseConfig.py", line 1374, in _dumpCfg
    values = _getCompressedNodes(s, loc, list(iter(toks[0][1])) )
  File "/afs/cern.ch/cms/sw/ReleaseCandidates/slc4_ia32_gcc345/thu/2.1-thu-02/CMSSW_2_1_X_2008-05-08-0200/python/FWCore/ParameterSet/parseConfig.py", line 1366, in _getCompressedNodes
    raise pp.ParseFatalException(s,loc,"the process contains the error \n"+str(e))
FWCore.ParameterSet.parsecf.pyparsing.ParseFatalException: the process contains the error 
multiple items found with label:XMLIdealGeometryESSource (at char 0), (line:1, col:1)
