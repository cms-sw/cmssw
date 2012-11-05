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
    outputModule = getattr(process, str(getattr(process, list(process.endpaths)[-1])))
  except:
    pass

  outputModule.outputCommands.extend(
    cms.untracked.vstring("keep *_goldenZmumuCandidatesGe0IsoMuons_*_*",
                          "keep *_goldenZmumuCandidatesGe1IsoMuons_*_*",
                          "keep *_goldenZmumuCandidatesGe2IsoMuons_*_*",
                          "keep *_goldenZmumuPreFilterHistos_*_*",
                          "keep *_goldenZmumuPostFilterHistos_*_*"))

  process.load('Configuration.StandardSequences.GeometryDB_cff')
  process.load('Configuration.StandardSequences.MagneticField_38T_cff')
  process.load("TrackingTools/TransientTrack/TransientTrackBuilder_cfi")
  process.load("TauAnalysis/MCEmbeddingTools/ZmumuStandaloneSelection_cff")

  # output file for muon acceptance histograms
  process.TFileService = cms.Service("TFileService",
    fileName = cms.string("histo_skimmed.root")
  )

  # Define configuration parameter default values
  from TauAnalysis.MCEmbeddingTools.setDefaults import setDefaults
  setDefaults(process)
  
  # Read configuration parameter values by command-line parsing
  from TauAnalysis.MCEmbeddingTools.embeddingCommandLineOptions import parseCommandLineOptions
  if process.options['parseCommandLine']:
    parseCommandLineOptions(process)

  # Add mumu selection to schedule
  process.goldenZmumuSkimPath = cms.Path(process.goldenZmumuSelectionSequence)
  process.goldenZmumuFilter.src = process.customization_options.ZmumuCollection
  process.schedule.insert(0, process.goldenZmumuSkimPath)

  # Only write out events which have at least one muon pair
  outputModule.SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring('goldenZmumuSkimPath'))

  return process
