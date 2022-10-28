import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/HGCalCommonData/data/TB180/cms.xml',
        'Geometry/HGCalCommonData/data/TB180/hgcal.xml',
        'Geometry/HGCalCommonData/data/TB180/hgcalEE.xml',
        'Geometry/HGCalCommonData/data/TB180/Absorb22/hgcalAbsorber.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/v7/hgcalwafer.xml',
        'Geometry/HGCalCommonData/data/TB180/hgcalsense.xml',
        'Geometry/HGCalCommonData/data/TB180/hgcProdCuts.xml',
        'Geometry/HGCalCommonData/data/TB180/hgcalCons.xml'),
    rootNodeName = cms.string('cms:OCMS')
)


