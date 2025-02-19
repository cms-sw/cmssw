# -*- coding: utf-8 -*-
import FWCore.ParameterSet.Config as cms


def customise(process):


  print "Add "
  print "process.local = cms.PSet() # for local running "
  if hasattr(process,"local"):
    print "#########################################################"
    print "  local run!"
    print "#########################################################"
    print
    print
    print
    
    base="file:/scratch/scratch0/tfruboes/DATA_tmp/RelValMinBias/CMSSW_4_2_0_pre4-MC_42_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG/"
    process.mix.input.fileNames = cms.untracked.vstring(
                                       base+'4C824492-2639-E011-9506-001A928116F0.root', 
                                       base+'9ED6309B-C238-E011-A1D6-003048678ADA.root', 
                                       base+'487A3591-CA38-E011-92A8-00248C0BE013.root', 
                                       base+'0E5A4D31-BD38-E011-93B7-0026189437F2.root')



  process.mix.input.nbPileupEvents.probFunctionVariable = cms.vint32()
  process.mix.input.nbPileupEvents.probValue = cms.vdouble()
  #print dir(process.mix.input.nbPileupEvents.probValue)
  for i in range(0,51):
    process.mix.input.nbPileupEvents.probFunctionVariable.append(i)
    val=0.25
    if i == 0 or i==9 or i==19 or i==29:
    #val=1
    #if i==0:
      process.mix.input.nbPileupEvents.probValue.append(val)
    else:
      process.mix.input.nbPileupEvents.probValue.append(0.)

  return(process)
