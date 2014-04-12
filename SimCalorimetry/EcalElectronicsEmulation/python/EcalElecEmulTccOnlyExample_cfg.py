import FWCore.ParameterSet.Config as cms

process = cms.Process("EcalSimRawData")
#simulation of raw data. Defines the ecalSimRawData module:
process.load("SimCalorimetry.EcalElectronicsEmulation.EcalSimRawData_cfi")

# Geometry
#
process.load("Geometry.CMSCommonData.cmsSimIdealGeometryXML_cfi")

# Calo geometry service model
process.load("Geometry.CaloEventSetup.CaloGeometry_cff")

# Description of EE trigger tower map
process.load("Geometry.CaloEventSetup.EcalTrigTowerConstituents_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
    # number of events to generate:
    input = cms.untracked.int32(2)
)

process.ecalSimpleProducer = cms.EDProducer("EcalSimpleProducer",
    #      string formula = "200+(4<=isample0)*(isample0<=6)*16*(1.)+1<<12"
    formula = cms.string(''),
    #TT samples:
    #  TT sample format:
    #        |11 | 10 |    9 - 0    |
    #        |gap|fgvb|      Et     |
    # energy set to TT id in first event and then incremented at each event:
    tpFormula = cms.string('itt0+ievt0'),
    verbose = cms.untracked.bool(False)
)

process.p = cms.Path(process.ecalSimpleProducer*process.ecalSimRawData)
process.ecalSimRawData.trigPrimProducer = 'ecalSimpleProducer'
process.ecalSimRawData.tcpDigiCollection = ''
process.ecalSimRawData.tcc2dccData = False
process.ecalSimRawData.srp2dccData = False
process.ecalSimRawData.fe2dccData = False
process.ecalSimRawData.tpVerbose = False


