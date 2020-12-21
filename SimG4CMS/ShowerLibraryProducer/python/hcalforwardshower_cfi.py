import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml', 
        'Geometry/HcalCommonData/data/hcalforwardmaterial.xml', 
        'Geometry/HcalCommonData/data/hcalforwardshower/v2/hcalforwardshower.xml',
        'Geometry/HcalCommonData/data/hcalSimNumbering/hfshower/hcalSimNumbering.xml',
        'Geometry/HcalCommonData/data/hcalRecNumbering/hfshower/hcalRecNumbering.xml'),
    rootNodeName = cms.string('hcalforwardshower:HFWorld')
)


