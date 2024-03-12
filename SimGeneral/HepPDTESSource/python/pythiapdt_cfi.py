import FWCore.ParameterSet.Config as cms

HepPDTESSource = cms.ESSource("HepPDTESSource",
    pdtFileName = cms.FileInPath('SimGeneral/HepPDTESSource/data/pythiaparticle.tbl')
)


# foo bar baz
# P2kue033wzw7t
