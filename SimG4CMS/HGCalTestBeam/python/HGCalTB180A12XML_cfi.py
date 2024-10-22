import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/HGCalTBCommonData/data/TB180/cms.xml',
        'Geometry/HGCalTBCommonData/data/TB180/hgcal.xml',
        'Geometry/HGCalTBCommonData/data/TB180/hgcalEE.xml',
        'Geometry/HGCalTBCommonData/data/TB180/Absorb12/hgcalAbsorber.xml',
        'Geometry/HGCalTBCommonData/data/hgcalwafer/v7/hgcalwafer.xml',
        'Geometry/HGCalTBCommonData/data/TB180/hgcalsense.xml',
        'Geometry/HGCalTBCommonData/data/TB180/hgcProdCuts.xml',
        'Geometry/HGCalTBCommonData/data/TB180/hgcalCons.xml'),
    rootNodeName = cms.string('cms:OCMS')
)


