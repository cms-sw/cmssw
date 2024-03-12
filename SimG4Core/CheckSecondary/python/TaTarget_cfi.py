import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml', 
                               'SimG4Core/CheckSecondary/data/TaTarget.xml'),
    rootNodeName = cms.string('TaTarget:OCMS')
)


# foo bar baz
# tkTn2ymgVVL0O
