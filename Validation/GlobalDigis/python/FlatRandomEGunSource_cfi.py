import FWCore.ParameterSet.Config as cms

# this cfi included the stuff needed for a single particle gun
#
# flat random E-gun, single muon
# 
# if you want another particle type, replace the PartID
# (standard STDHEP numbering scheme)
#
# to run it along with CMS detector simulation
# (OscarProducer) make sure to select QGSP physics
# list, instead DummyPhysics ("Dummy" has only EM 
# process and wont know to model interactions of
# hadrons with matter)
#
source = cms.Source("FlatRandomEGunSource",
    PGunParameters = cms.untracked.PSet(
        #you can request more than 1 particle
        #vint32  PartID = {211,11}
        PartID = cms.untracked.vint32(13, 11, 211, 22),
        MaxEta = cms.untracked.double(3.5),
        MaxPhi = cms.untracked.double(3.14159265359),
        MinEta = cms.untracked.double(-3.5),
        MinE = cms.untracked.double(9.99),
        MinPhi = cms.untracked.double(-3.14159265359), ## must be in radians

        MaxE = cms.untracked.double(10.01)
    ),
    Verbosity = cms.untracked.int32(0), ## set to 1 (or greater)  for printouts

    AddAntiParticle = cms.untracked.bool(False), ## if you turn it ON, for each 

    #untracked int32 maxEvents = 5
    firstRun = cms.untracked.uint32(1)
)


