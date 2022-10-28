import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/HGCalCommonData/data/TB161/cms.xml',
        'Geometry/HGCalCommonData/data/TB161/hgcal.xml',
        'Geometry/HGCalCommonData/data/TB161/TimingLayer/hgcalEE.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/v7/hgcalwafer.xml',
        'Geometry/HGCalCommonData/data/TB161/TimingLayer/hgcalBeam.xml',
        'Geometry/HGCalCommonData/data/TB161/hgcalsense.xml',
        'Geometry/HGCalCommonData/data/TB161/hgcProdCuts.xml',
        'Geometry/HGCalCommonData/data/TB161/TimingLayer/hgcalCons.xml'),
    rootNodeName = cms.string('cms:OCMS')
)



