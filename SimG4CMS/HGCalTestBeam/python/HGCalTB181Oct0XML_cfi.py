import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
                               'Geometry/CMSCommonData/data/rotations.xml',
                               'Geometry/HGCalCommonData/data/hgcalMaterial/v2/hgcalMaterial.xml',
                               'Geometry/HGCalCommonData/data/TB181/cms.xml',
                               'Geometry/HGCalCommonData/data/TB181/Oct181/hgcal.xml',
                               'Geometry/HGCalCommonData/data/TB181/Oct181/hgcalEE.xml',
                               'Geometry/HGCalCommonData/data/TB181/Oct181/hgcalHE.xml',
                               'Geometry/HGCalCommonData/data/TB181/ahcal.xml',
                               'Geometry/HGCalCommonData/data/TB181/Oct180/hgcalBeam.xml',
                               'Geometry/HGCalCommonData/data/hgcalwafer/v7/hgcalwafer.xml',
                               'Geometry/HGCalCommonData/data/TB181/Oct181/hgcalsense.xml',
                               'Geometry/HGCalCommonData/data/TB181/hgcProdCuts.xml',
                               'Geometry/HGCalCommonData/data/TB181/Oct181/hgcalCons.xml'
                               ),
    rootNodeName = cms.string('cms:OCMS')
)


