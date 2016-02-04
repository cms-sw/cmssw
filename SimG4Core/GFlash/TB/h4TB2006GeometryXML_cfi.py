import FWCore.ParameterSet.Config as cms

MuonNumberingInitialization = cms.ESProducer("MuonNumberingInitialization")

XMLIdealGeometryESSource = cms.ESSource("XMLIdealGeometryESSource",
    geomXMLFiles = cms.vstring('Geometry/CMSCommonData/data/materials.xml', 
        'Geometry/CMSCommonData/data/rotations.xml', 
        'Geometry/EcalTestBeam/data/ebcon.xml', 
        'Geometry/EcalCommonData/data/ebrot.xml', 
        'Geometry/EcalTestBeam/data/eregalgo.xml', 
        'Geometry/EcalCommonData/data/ebalgo.xml', 
        'SimG4Core/GFlash/TB/tbrot.xml', 
        'Geometry/EcalTestBeam/data/TBH4.xml', 
        'Geometry/EcalTestBeam/data/TBH4ecalsens.xml', 
        'Geometry/HcalSimData/data/CaloUtil.xml', 
        'SimG4Core/GFlash/TB/gflashTBH4ProdCuts.xml', 
        'Geometry/CMSCommonData/data/FieldParameters.xml'),
    rootNodeName = cms.string('TBH4:OCMS')
)



