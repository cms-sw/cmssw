import FWCore.ParameterSet.Config as cms

HepPDTESSource = cms.ESSource("HepPDTESSource",
    pdtFileName = cms.FileInPath('SimGeneral/HepPDTESSource/data/particle.tbl')
)


