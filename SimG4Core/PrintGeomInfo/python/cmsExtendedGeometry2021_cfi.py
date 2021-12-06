import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring(
         'SimG4Core/PrintGeomInfo/data/dddDBBigFile.xml',
         'SimG4Core/PrintGeomInfo/data/hcalSimNumbering.xml',
         'SimG4Core/PrintGeomInfo/data/hcalRecNumbering.xml'),
    rootNodeName = cms.string('cms:OCMS')
)
