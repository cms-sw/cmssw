import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml', 
                               'SimG4Core/CheckSecondary/data/AlTarget.xml'),
    rootNodeName = cms.string('AlTarget:OCMS')
)


# foo bar baz
# K3JfAsG4ZZ1a9
# XU7FSKtDtJ9IN
