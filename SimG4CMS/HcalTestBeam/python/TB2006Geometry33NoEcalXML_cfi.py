import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml', 
        'Geometry/CMSCommonData/data/rotations.xml', 
        'Geometry/HcalCommonData/data/hcalrotations.xml', 
        'SimG4CMS/HcalTestBeam/test/33/TBHcal.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal06BeamLine.xml', 
        'Geometry/HcalTestBeamData/data/TBHcalCable.xml', 
        'Geometry/HcalTestBeamData/data/TBHcalBarrel.xml', 
        'Geometry/HcalTestBeamData/data/TBHcalEndcap.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal06HcalOuter.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal06Sens.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal06ProdCuts.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal06Util.xml'),
    rootNodeName = cms.string('TBHcal:OTBHCal')
)


