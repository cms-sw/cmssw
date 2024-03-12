import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml', 
                               'SimG4Core/CheckSecondary/data/NbTarget.xml'),
    rootNodeName = cms.string('NbTarget:OCMS')
)


# foo bar baz
# 68SIN4ygl4zze
# RYooolNQ7BNab
