import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/HGCalTBCommonData/data/TB170/cms.xml',
        'Geometry/HGCalTBCommonData/data/TB170/hgcal.xml',
        'Geometry/HGCalTBCommonData/data/TB170/hgcalEE.xml',
        'Geometry/HGCalTBCommonData/data/TB170/hgcalHE.xml',
        'Geometry/HGCalTBCommonData/data/TB170/ahcal.xml',
        'Geometry/HGCalTBCommonData/data/hgcalwafer/v7/hgcalwafer.xml',
        'Geometry/HGCalTBCommonData/data/TB170/hgcalBeam.xml',
        'Geometry/HGCalTBCommonData/data/TB170/hgcalsense.xml',
        'Geometry/HGCalTBCommonData/data/TB170/hgcProdCuts.xml',
        'Geometry/HGCalTBCommonData/data/TB170/hgcalCons.xml'),
    rootNodeName = cms.string('cms:OCMS')
)


