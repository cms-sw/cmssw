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
    cms.untracked.vstring("keep *_goodMuons_*_*",
                          "keep *_goodMuonsPFIso_*_*",
                          "keep *_highestPtMuPlus_*_*",
                          "keep *_highestPtMuMinus_*_*",
                          "keep *_highestPtMuPlusPFIso_*_*",
                          "keep *_highestPtMuMinusPFIso_*_*",
                          "keep *_goldenZmumuCandidatesGe0IsoMuons_*_*",
                          "keep *_goldenZmumuCandidatesGe1IsoMuons_*_*",
                          "keep *_goldenZmumuCandidatesGe2IsoMuons_*_*",
                          "keep TH2DMEtoEDM_MEtoEDMConverter_*_*",
                          "drop *_TriggerResults_*_EmbeddedINPUT"))

  process.load('Configuration.StandardSequences.GeometryDB_cff')
  process.load('Configuration.StandardSequences.MagneticField_38T_cff')
  process.load("TrackingTools/TransientTrack/TransientTrackBuilder_cfi")
  process.load("TauAnalysis/MCEmbeddingTools/ZmumuStandaloneSelection_cff")

  # DQM store output for muon acceptance histograms
  process.load('DQMServices.Core.DQMStore_cfg')

  # Define configuration parameter default values
  from TauAnalysis.MCEmbeddingTools.setDefaults import setDefaults
  setDefaults(process)
  
  # Read configuration parameter values by command-line parsing
  #from TauAnalysis.MCEmbeddingTools.embeddingCommandLineOptions import parseCommandLineOptions
  #if process.options['parseCommandLine']:
  #  parseCommandLineOptions(process)

  process.load('HLTrigger.HLTfilters.triggerResultsFilter_cfi')
  process.embedTriggerFilter = process.triggerResultsFilter.clone(
    hltResults = cms.InputTag('TriggerResults', '', 'HLT'),
    l1tResults = cms.InputTag(''),
    triggerConditions = cms.vstring('HLT_Mu17_Mu8_v*')
  )

  # Add mumu selection to schedule
  if process.customization_options.isMC.value():
    process.goldenZmumuFilterPath = cms.Path(process.embedTriggerFilter*process.goldenZmumuFilterSequence)
  else:
    process.goldenZmumuFilterPath = cms.Path(process.embedTriggerFilter*process.goldenZmumuFilterSequenceData)
  process.goldenZmumuFilter.src = process.customization_options.ZmumuCollection
  process.schedule.insert(0, process.goldenZmumuFilterPath)

  # Only write out events which have at least one muon pair
  outputModule.SelectEvents = cms.untracked.PSet(
    SelectEvents = cms.vstring('goldenZmumuFilterPath'))

  return process
