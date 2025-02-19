import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml', 
        'Geometry/CMSCommonData/data/rotations.xml', 
        'Geometry/HcalCommonData/data/hcalrotations.xml', 
        'SimG4CMS/HcalTestBeam/test/2010/TBHcal.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal10BeamLine.xml', 
        'Geometry/HcalTestBeamData/data/TBHcalCable.xml', 
        'Geometry/HcalTestBeamData/data/2010/TBHcalBarrel.xml', 
        'Geometry/HcalTestBeamData/data/TBHcalEndcap.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal10HcalOuter.xml', 
        'Geometry/HcalTestBeamData/data/ebcon.xml', 
        'Geometry/HcalTestBeamData/data/eregalgo.xml', 
        'Geometry/EcalCommonData/data/ebrot.xml', 
        'Geometry/EcalCommonData/data/ebalgo.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal10Sens.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal10ebsens.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal10ProdCuts.xml', 
        'Geometry/EcalSimData/data/EBProdCuts.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal10Util.xml'),
    rootNodeName = cms.string('TBHcal:OTBHCal')
)


