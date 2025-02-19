# -*- coding: utf-8 -*-
import FWCore.ParameterSet.Config as cms
import os

def customise(process):
 
  process._Process__name="EmbeddedINPUT"

  try:
	  outputModule = process.output
  except:
    pass
  try:
	  outputModule = getattr(process,str(getattr(process,list(process.endpaths)[-1])))
  except:
    pass

  outputModule.outputCommands.extend(
    cms.untracked.vstring("keep *_goldenZmumuCandidatesGe0IsoMuons_*_*",
                          "keep *_goldenZmumuCandidatesGe1IsoMuons_*_*",
                          "keep *_goldenZmumuCandidatesGe2IsoMuons_*_*"))

  process.load('Configuration.StandardSequences.GeometryDB_cff')
  process.load('Configuration.StandardSequences.MagneticField_38T_cff')
  process.load("TrackingTools/TransientTrack/TransientTrackBuilder_cfi")
  process.load("TauAnalysis/MCEmbeddingTools/ZmumuStandalonSelection_cff")

  # Add mumu selection to schedule
  process.goldenZmumuSkimPath = cms.Path(process.goldenZmumuSelectionSequence)
  process.schedule.insert(0, process.goldenZmumuSkimPath)

  # Only write out events which have at least one muon pair
  outputModule.SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring('goldenZmumuSkimPath'))

  return(process)
