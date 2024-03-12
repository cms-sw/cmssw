import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml', 
                               'SimG4Core/CheckSecondary/data/CuTarget.xml'),
    rootNodeName = cms.string('CuTarget:OCMS')
)


# foo bar baz
# 0dHeraLLwrt6n
