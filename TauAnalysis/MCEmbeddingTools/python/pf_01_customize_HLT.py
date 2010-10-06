# -*- coding: utf-8 -*-

from FWCore.ParameterSet.Modules import _Module


def customise(process):
 
   
  
  process._Process__name="HLT2"
  process.TFileService = cms.Service("TFileService",  fileName = cms.string("histo_simulation.root")          )

  try:
	  outputModule = process.output
  except:
    pass
  try:
	  outputModule = getattr(process,str(getattr(process,list(process.endpaths)[-1])))
  except:
    pass

  print "Changing eventcontent to RAW+AODSIM + misc. "
  outputModule.outputCommands = cms.untracked.vstring("drop *")
  outputModule.outputCommands.extend(process.RAWEventContent.outputCommands )
  outputModule.outputCommands.extend(process.AODSIMEventContent.outputCommands )

  keepMC = cms.untracked.vstring("keep *_*_zMusExtracted_*",
                                 "keep *_*_zmmCands_*",
                                 "keep *_dimuonsGlobal_*_*",
                                 'keep *_generator_*_*'
  )
  outputModule.outputCommands.extend(keepMC)

  # getRid of second "drop *"
  index = 0
  for item in outputModule.outputCommands[:]:
    if item == "drop *" and index != 0:
      #print index," ",outputModule.outputCommands[index]
      del outputModule.outputCommands[index]
      index -= 1
    index += 1  



  process.VtxSmeared = cms.EDProducer("FlatEvtVtxGenerator", 
    MaxZ = cms.double(0.0),
    MaxX = cms.double(0.0),
    MaxY = cms.double(0.0),
    MinX = cms.double(0.0),
    MinY = cms.double(0.0),
    MinZ = cms.double(0.0),
    TimeOffset = cms.double(0.0),
    src = cms.InputTag("generator")
  )


  return(process)
