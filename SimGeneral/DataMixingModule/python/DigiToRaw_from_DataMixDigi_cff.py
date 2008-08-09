import FWCore.ParameterSet.Config as cms

from EventFilter.CSCTFRawToDigi.csctfpacker_cfi import *
from EventFilter.DTTFRawToDigi.dttfpacker_cfi import *
from EventFilter.GctRawToDigi.gctDigiToRaw_cfi import *
from EventFilter.L1GlobalTriggerRawToDigi.l1GtPack_cfi import *
from EventFilter.SiPixelRawToDigi.SiPixelDigiToRaw_cfi import *
from EventFilter.SiStripRawToDigi.SiStripDigiToRaw_cfi import *
from SimCalorimetry.EcalTrigPrimProducers.ecalTriggerPrimitiveDigis_cff import *
import EventFilter.EcalDigiToRaw.ecalDigiToRaw_cfi
ecalPacker = EventFilter.EcalDigiToRaw.ecalDigiToRaw_cfi.ecaldigitorawzerosup.clone()
from EventFilter.ESDigiToRaw.esDigiToRaw_cfi import *
from EventFilter.HcalRawToDigi.HcalDigiToRaw_cfi import *
from EventFilter.CSCRawToDigi.cscPacker_cfi import *
from EventFilter.DTRawToDigi.dtPacker_cfi import *
from EventFilter.RPCRawToDigi.rpcPacker_cfi import *
from EventFilter.RawDataCollector.rawDataCollector_cfi import *
DigiToRaw = cms.Sequence(csctfpacker*dttfpacker*gctDigiToRaw*l1GtPack*siPixelRawData*SiStripDigiToRaw*ecalPacker*esDigiToRaw*hcalRawData*cscpacker*dtpacker*rpcpacker*rawDataCollector)

#
# put replacements here:
#
# Have to read merged Digis or RecHits from DataMixer
#
# start with muons:
csc2DRecHits.CSCDigiTag = 'mix'
csc2DRecHits.CSCStripDigiTag = 'MuonCSCStripDigisDM'
csc2DRecHits.CSCWireDigiTag = 'MuonCSCWireDigisDM'
dt1DRecHits.dtDigiLabel = 'mix:muonDTDigisDM'
dt1DRecHits.rpcDigiLabel = 'mix:muonRPCDigisDM'
# Calo - using RecHits here
islandBasicClusters.barrelHitCollection = 'EcalRecHitsEBDM'
islandBasicClusters.endcapHitCollection = 'EcalRecHitsEEDM'
towerMaker.hbheInput = 'mix:HBHERecHitCollectionDM'
towerMaker.hoInput = 'mix:HORecHitCollectionDM'
towerMaker.hfInput = 'mix:HFRecHitCollectionDM'
towerMaker.ecalInputs = { 'mix:EcalRecHitsEBDM', 'mix:EcalRecHitsEEDM' }
# Tracker
siStripClusters.DigiProducersList???  { 'mix','SiStripDigisDM'}
siPixelClusters.src = 'mix:siPixelDigisDM'
#

# packer replacements
csctfpacker.lctProducer = "simCscTriggerPrimitiveDigis:MPCSORTED"
csctfpacker.trackProducer = 'simCsctfTrackDigis'
cscpacker.wireDigiTag = cms.InputTag("mix","MuonCSCWireDigiDM"),
cscpacker.stripDigiTag = cms.InputTag("mix","MuonCSCStripDigiDM")
#
dttfpacker.DTDigi_Source = 'simDtTriggerPrimitiveDigis'
dttfpacker.DTTracks_Source = "simDttfDigis:DTTF"



gctDigiToRaw.rctInputLabel = 'simRctDigis'
gctDigiToRaw.gctInputLabel = 'simGctDigis'
l1GtPack.DaqGtInputTag = 'simGtDigis'
l1GtPack.MuGmtInputTag = 'simGmtDigis'
ecalPacker.Label = 'simEcalDigis'
ecalPacker.InstanceEB = 'ebDigis'
ecalPacker.InstanceEE = 'eeDigis'
ecalPacker.labelEBSRFlags = "simEcalDigis:ebSrFlags"
ecalPacker.labelEESRFlags = "simEcalDigis:eeSrFlags"

