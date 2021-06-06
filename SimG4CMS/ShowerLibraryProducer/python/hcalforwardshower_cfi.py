import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml', 
        'Geometry/HcalCommonData/data/hcalforwardmaterial.xml', 
        'Geometry/HcalCommonData/data/hcalforwardshower/v2/hcalforwardshower.xml',
        'Geometry/HcalCommonData/data/hcalSimNumbering/hfshower/v1/hcalSimNumbering.xml',
        'Geometry/HcalCommonData/data/hcalRecNumbering/hfshower/v1/hcalRecNumbering.xml'),
    rootNodeName = cms.string('hcalforwardshower:HFWorld')
)


