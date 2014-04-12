# -*- coding: utf-8 -*-

import FWCore.ParameterSet.Config as cms
import os

def parseCommandLineOptions(process):

  import FWCore.ParameterSet.VarParsing as VarParsing
  options = VarParsing.VarParsing('analysis')
  options.register('ZmumuCollection',
                   'goldenZmumuCandidatesGe2IsoMuons', # default value
                   VarParsing.VarParsing.multiplicity.singleton,
                   VarParsing.VarParsing.varType.int,         
                   "collection of selected Z->mumu candidates")
  
  options.register('mdtau',
                   0, # default value
                   VarParsing.VarParsing.multiplicity.singleton,
                   VarParsing.VarParsing.varType.int,         
                   "mdtau value passed to TAUOLA for selecting tau decay modes")

  options.register('transformationMode',
                   1, #default value
                   VarParsing.VarParsing.multiplicity.singleton,
                   VarParsing.VarParsing.varType.int,
                   "transformation mode: 0=mumu->mumu, 1=mumu->tautau")

  options.register('embeddingMode',
                   0, #default value
                   VarParsing.VarParsing.multiplicity.singleton,
                   VarParsing.VarParsing.varType.int,
                   "embedding mode: 0=PF, 1=RH")

  options.register('minVisibleTransverseMomentum',
                   "", #default value
                   VarParsing.VarParsing.multiplicity.singleton,
                   VarParsing.VarParsing.varType.string,
                   "generator level cut on visible transverse momentum (typeN:pT,[...];[...])")
  
  options.register('useJson',
                   0, # default value, false
                   VarParsing.VarParsing.multiplicity.singleton,
                   VarParsing.VarParsing.varType.int,         
                   "should I enable event selection by JSON file ?")

  options.register('overrideBeamSpot',
                   0, # default value, false
                   VarParsing.VarParsing.multiplicity.singleton,
                   VarParsing.VarParsing.varType.int,         
                   "should I override beamspot in globaltag ?")

  options.register('doNotSkimEvents',
                   0, # default value, false
                   VarParsing.VarParsing.multiplicity.singleton,
                   VarParsing.VarParsing.varType.int,         
                   "should I disable all event selection cuts ?")

  setFromCL = False
  if not hasattr(process, "doNotParse"):
    import sys
    if hasattr(sys, "argv") == True:
      if not sys.argv[0].endswith('cmsDriver.py'):
        options.parseArguments()
        setFromCL = True
  if not setFromCL:
    print "CL parsing disabled!"

  if setFromCL:
    process.options['mdtau'] = options.mdtau
    process.options['transformationMode'] = options.transformationMode
    process.options['embeddingMode'] = options.embeddingMode
    process.options['minVisibleTransverseMomentum'] = options.minVisibleTransverseMomentum
    if options.useJson != 0:
      process.options['useJson'] = True
    else:
      process.options['useJson'] = False
    if options.overrideBeamSpot != 0:
      process.options['overrideBeamSpot'] = True
    else:
      process.options['overrideBeamSpot'] = False
    if options.doNotSkimEvents != 0:
      process.options['doNotSkimEvents'] = True
    else:
      process.options['doNotSkimEvents'] = False
