import FWCore.ParameterSet.Config as cms

HepPDTESSource = cms.ESSource("HepPDTESSource",
    pdtFileName = cms.FileInPath('SimG4CMS/Calo/data/pythiaparticle.tbl')
)


