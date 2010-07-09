# -*- coding: utf-8 -*-

from FWCore.ParameterSet.Modules import _Module


def customise(process):
  
  
  process._Process__name="HLT2"
  process.TFileService = cms.Service("TFileService",  fileName = cms.string("histo.root")          )

  print "Changing eventcontent to RAW+AODSIM + misc. "
  process.output.outputCommands = cms.untracked.vstring("drop *")
  process.output.outputCommands.extend(process.RAWEventContent.outputCommands )
  process.output.outputCommands.extend(process.AODSIMEventContent.outputCommands )

  keepMC = cms.untracked.vstring("keep *_*_zMusExtracted_*",
                                 "keep *_dimuonsGlobal_*_*",
                                 'keep *_generator_*_*'
  )
  process.output.outputCommands.extend(keepMC)

  # getRid of second "drop *"
  index = 0
  for item in process.output.outputCommands[:]:
    if item == "drop *" and index != 0:
      #print index," ",process.output.outputCommands[index]
      del process.output.outputCommands[index]
      index -= 1
    index += 1  


  return(process)
