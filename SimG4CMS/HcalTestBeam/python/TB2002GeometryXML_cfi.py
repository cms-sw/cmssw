import FWCore.ParameterSet.Config as cms

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/HcalTestBeamData/data/TBHcal02Materials.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal02Rotations.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal02.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal02BeamLine.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal02Xtal.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal02HcalBarrel.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal02HcalOuter.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal02HcalSens.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal02XtalSens.xml', 
        'Geometry/HcalTestBeamData/data/TBHcal02Util.xml'),
    rootNodeName = cms.string('TBHcal02:TestBeamHCal02')
)


