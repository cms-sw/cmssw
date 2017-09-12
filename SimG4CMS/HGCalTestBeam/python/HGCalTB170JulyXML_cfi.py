import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
        'Geometry/CMSCommonData/data/rotations.xml',
        'Geometry/HGCalCommonData/data/TB170/cms.xml',
        'Geometry/HGCalCommonData/data/TB170/July17/hgcal.xml',
        'Geometry/HGCalCommonData/data/TB170/July17/hgcalEE.xml',
        'Geometry/HGCalCommonData/data/TB170/July17/hgcalHE.xml',
        'Geometry/HGCalCommonData/data/TB170/July17/ahcal.xml',
        'Geometry/HGCalCommonData/data/hgcalwafer/v7/hgcalwafer.xml',
        'Geometry/HGCalCommonData/data/TB170/July17/hgcalBeam.xml',
        'Geometry/HGCalCommonData/data/TB170/hgcalsense.xml',
        'Geometry/HGCalCommonData/data/TB170/hgcProdCuts.xml',
        'Geometry/HGCalCommonData/data/TB170/hgcalCons.xml'),
    rootNodeName = cms.string('cms:OCMS')
)


