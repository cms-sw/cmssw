import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml', 
        'Geometry/HcalCommonData/data/hcalrotations.xml', 
        'SimG4CMS/HcalTestBeam/test/data/TBHcal.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal04BeamLine.xml', 
        'Geometry/HcalTestBeamData/data/TBHcalXtal.xml', 
        'Geometry/HcalTestBeamData/data/TBHcalCable.xml', 
        'Geometry/HcalTestBeamData/data/TBHcalBarrel.xml', 
        'Geometry/HcalTestBeamData/data/TBHcalEndcap.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal03HcalOuter.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal03Sens.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal04Util.xml', 
        'Geometry/HcalTestBeamData/data/CaloUtil.xml',
        'Geometry/HcalTestBeamData/data/TBHcal04ProdCuts.xml'),
    rootNodeName = cms.string('TBHcal:OTBHCal')
)


