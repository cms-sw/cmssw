
import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml',
                               'Geometry/CMSCommonData/data/rotations.xml',
                               'Geometry/HGCalCommonData/data/hgcalMaterial/v1/hgcalMaterial.xml',
                               'Geometry/HGCalCommonData/data/TB181/cms.xml',
                               'Geometry/HGCalCommonData/data/TB181/June18/hgcal.xml',
                               'Geometry/HGCalCommonData/data/TB181/June18/hgcalEE.xml',
                               'Geometry/HGCalCommonData/data/TB181/June18/hgcalBeam.xml',
                               'Geometry/HGCalCommonData/data/hgcalwafer/v7/hgcalwafer.xml',
                               'Geometry/HGCalCommonData/data/TB181/June18/hgcalsense.xml',
                               'Geometry/HGCalCommonData/data/TB181/June18/hgcProdCuts.xml',
                               'Geometry/HGCalCommonData/data/TB181/June18/hgcalCons.xml'
                               ),
    rootNodeName = cms.string('cms:OCMS')
)


