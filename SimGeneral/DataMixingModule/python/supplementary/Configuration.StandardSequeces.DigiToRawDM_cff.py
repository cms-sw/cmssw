import FWCore.ParameterSet.Config as cms

from EventFilter.CSCTFRawToDigi.csctfpacker_cfi import *
from EventFilter.DTTFRawToDigi.dttfpacker_cfi import *
from EventFilter.GctRawToDigi.gctDigiToRaw_cfi import *
from EventFilter.L1GlobalTriggerRawToDigi.l1GtPack_cfi import *
from EventFilter.L1GlobalTriggerRawToDigi.l1GtEvmPack_cfi import *
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
from EventFilter.CastorRawToDigi.CastorDigiToRaw_cfi import *
from EventFilter.RawDataCollector.rawDataCollector_cfi import *
DigiToRaw = cms.Sequence(csctfpacker*dttfpacker*gctDigiToRaw*l1GtPack*l1GtEvmPack*siPixelRawData*SiStripDigiToRaw*ecalPacker*esDigiToRaw*hcalRawData*cscpacker*dtpacker*rpcpacker*rawDataCollector)
#DigiToRaw = cms.Sequence(csctfpacker*dttfpacker*gctDigiToRaw*l1GtPack*l1GtEvmPack*siPixelRawData*SiStripDigiToRaw*ecalPacker*esDigiToRaw*hcalRawData*cscpacker*dtpacker*rpcpacker*castorRawData*rawDataCollector)
#
# Re-define inputs to look at the DataMixer output
#
siPixelRawData.InputLabel = cms.InputTag("mixData:siPixelDigisDM")
SiStripDigiToRaw.InputModuleLabel = cms.string('mixData')
SiStripDigiToRaw.InputDigiLabel = cms.string('siStripDigisDM')
#
ecalPacker.Label = 'mixData'
ecalPacker.InstanceEB = 'EBDigiCollectionDM'
ecalPacker.InstanceEE = 'EEDigiCollectionDM'
esDigiToRaw.InstanceES = cms.string('ESDigiCollectionDM')
esDigiToRaw.Label = cms.string('mixData')
#
hcalRawData.HBHE = cms.untracked.InputTag("mixData")
hcalRawData.HF = cms.untracked.InputTag("mixData")
hcalRawData.HO = cms.untracked.InputTag("mixData")
hcalRawData.ZDC = cms.untracked.InputTag("mixData")
#
cscpacker.wireDigiTag = cms.InputTag("mixData","MuonCSCWireDigisDM")
cscpacker.stripDigiTag = cms.InputTag("mixData","MuonCSCStripDigisDM")
cscpacker.comparatorDigiTag = cms.InputTag("mixData","MuonCSCComparatorDigisDM")
dtpacker.digibytype =  cms.untracked.bool(False)
#dtpacker.digiColl = cms.untracked.string('mixData')
dtpacker.digiColl = cms.untracked.string('simMuonDTDigis')
rpcpacker.InputLabel = cms.InputTag("mixData")
#
csctfpacker.lctProducer = "simCscTriggerPrimitiveDigis:MPCSORTED"
csctfpacker.trackProducer = 'simCsctfTrackDigis'
dttfpacker.DTDigi_Source = 'simDtTriggerPrimitiveDigis'
dttfpacker.DTTracks_Source = "simDttfDigis:DTTF"
gctDigiToRaw.rctInputLabel = 'simRctDigis'
gctDigiToRaw.gctInputLabel = 'simGctDigis'
l1GtPack.DaqGtInputTag = 'simGtDigis'
l1GtPack.MuGmtInputTag = 'simGmtDigis'
l1GtEvmPack.EvmGtInputTag = 'simGtDigis'
ecalPacker.labelEBSRFlags = "simEcalDigis:ebSrFlags"
ecalPacker.labelEESRFlags = "simEcalDigis:eeSrFlags"

