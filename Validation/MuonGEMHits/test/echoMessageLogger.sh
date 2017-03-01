#!/bin/bash
echo """
process.load(\"FWCore.MessageLogger.MessageLogger_cfi\")
process.MessageLogger = cms.Service(
    \"MessageLogger\",
    destinations = cms.untracked.vstring(
        'detailedInfo',
         'critical'
         ),
    detailedInfo = cms.untracked.PSet(
        threshold = cms.untracked.string('DEBUG')
         ),
    debugModules = cms.untracked.vstring(
        'gemSimHitValidation',
        'gemStripValidation',

        )
    )
"""
