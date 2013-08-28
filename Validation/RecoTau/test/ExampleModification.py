#TestModify.py
import FWCore.ParameterSet.Config as cms
from RecoTauTag.Configuration.RecoPFTauTag_cff import shrinkingConePFTauDiscriminationByLeadingPionPtCut

print "Modifying lead pion requirement."
shrinkingConePFTauDiscriminationByLeadingPionPtCut.MinPtLeadingPion = cms.double(20.)
