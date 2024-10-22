import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
                               'Geometry/CMSCommonData/data/rotations.xml',
                               'Geometry/HGCalTBCommonData/data/hgcalMaterial/v1/hgcalMaterial.xml',
                               'Geometry/HGCalTBCommonData/data/TB181/Oct181/cms.xml',
                               'Geometry/HGCalTBCommonData/data/TB181/Oct182/hgcal.xml',
                               'Geometry/HGCalTBCommonData/data/TB181/Oct182/hgcalEE.xml',
                               'Geometry/HGCalTBCommonData/data/TB181/Oct182/hgcalHE.xml',
                               'Geometry/HGCalTBCommonData/data/TB181/ahcal.xml',
                               'Geometry/HGCalTBCommonData/data/TB181/Oct182/hgcalBeam.xml',
                               'Geometry/HGCalTBCommonData/data/hgcalwafer/v7/hgcalwafer.xml',
                               'Geometry/HGCalTBCommonData/data/TB181/Oct181/hgcalsense.xml',
                               'Geometry/HGCalTBCommonData/data/TB181/hgcProdCuts.xml',
                               'Geometry/HGCalTBCommonData/data/TB181/Oct181/hgcalCons.xml'
                               ),
    rootNodeName = cms.string('cms:OCMS')
)


