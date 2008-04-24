import FWCore.ParameterSet.Config as cms

simMuonDTDigis = cms.EDFilter("DTDigitizer",
    # Velocity of signal propagation along the wire (cm/ns)
    # For the default value
    # cfr. CMS-IN 2000-021:   (2.56+-0.17)x1e8 m/s
    #      CMS NOTE 2003-17:  (0.244)  m/ns
    vPropWire = cms.double(24.4),
    # The Synchronization Algo Name. Algos Type= [DTDigiSyncTOFCorr,DTDigiSyncFromTable]
    SyncName = cms.string('DTDigiSyncTOFCorr'),
    # configure the creation of Digi-Sim links:	
    #   one-to-one correspondence between digis and SimHits (MultipleLinks=false)
    #   or association of SimHits within a given time window (of the order of the resolution, in ns)
    MultipleLinks = cms.bool(True),
    # constant drift velocity for the IdealModel (cm/ns)
    IdealModelConstantDriftVelocity = cms.double(0.0055),
    interpolate = cms.bool(True),
    # Its parameters
    pset = cms.PSet(
        TOFCorrection = cms.int32(2),
        offset = cms.double(500.0)
    ),
    # Dead time for signals on the same wire (number from M. Pegoraro)  
    deadTime = cms.double(150.0),
    debug = cms.untracked.bool(False),
    # further configurable smearing
    Smearing = cms.double(2.4),
    # ideal model. Used for debug.
    # It uses constant drift velocity and it hasn't any external delay 	
    IdealModel = cms.bool(False),
    LinksTimeWindow = cms.double(10.0),
    onlyMuHits = cms.bool(False)
)



