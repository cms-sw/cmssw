#!/usr/bin/env python

import FWCore.ParameterSet.Config as cms
import Validation.RecoTau.DBSApi_cff as mydbs

dummy = cms.Source ("PoolSource",fileNames = cms.untracked.vstring())
dataDict = {}
for key in mydbs.datasetDict.keys():
    mydbs.FillSource(key,dummy)
    dataDict[key]= [{'file':fname} for fname in dummy.fileNames]
    dummy.fileNames = cms.untracked.vstring()
    

toxml = {'dataFiles' : dataDict}

outFile = open('SourcesDatabase.xml','w')
outFile.write(mydbs.DictToXML(toxml))
