import FWCore.ParameterSet.Config as cms

# Module to create simulated ME0 Pre Reco digis.
simMuonME0ReDigis = cms.EDProducer("ME0ReDigiProducer",
    inputCollection    =cms.string('simMuonME0Digis'),
    useBuiltinGeo      =cms.bool(True),   #Use CMSSW defined geometry for digitization, not custom strips and paritions
    numberOfStrips     =cms.uint32(384), # If use custom: number of strips per partition                                             
    numberOfPartitions =cms.uint32(8),   # If use custom:  number of partitions per chamber                                           
    neutronAcceptance  =cms.double(2.0),   # fraction of neutron events to keep in event (>= 1 means no filtering)      
    timeResolution     =cms.double(5),   # smear time by gaussian with this sigma (in ns)....negative for no smearing 
    minBXReadout       =cms.int32(-1),  # Minimum BX to readout                                                      
    maxBXReadout       =cms.int32(1), # Maximum BX to readout
    layerReadout       =cms.vint32(1,1,1,1,1,1), # Don't readout layer if entry is 0 (Layer number 1 (near IP) in the numbering scheme is idx 0)                                                                                                       
    mergeDigis         =cms.bool(True),   # Keep only one digi at the same chamber, strip, partition, and BX           
)