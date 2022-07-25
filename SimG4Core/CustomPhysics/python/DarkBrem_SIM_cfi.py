#To use, add the following to the python configuration:
#process.load('SimG4Core.CustomPhysics.DarkBrem_SIM_cfi')
#process.g4SimHits.Physics.type = 'SimG4Core/Physics/CustomPhysics' 
#process.g4SimHits.Physics = cms.PSet(
#process.g4SimHits.Physics,
#process.customPhysicsSetup
#)
#process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
#        DBremWatcher = cms.PSet(
#                PDGCodes = cms.untracked.vint32([9994]),
#                DBremBiasFactor = process.customPhysicsSetup.DBremBiasFactor
#        ),
#        type = cms.string('DBremWatcher') 
#) )

import FWCore.ParameterSet.Config as cms

customPhysicsSetup = cms.PSet(
    DBrem = cms.untracked.bool(True),
    DBremMass = cms.untracked.double(1000.0),  #Mass in MeV
    DBremScaleFile = cms.untracked.string("root://cmseos.fnal.gov//store/user/revering/MuDBrem_Cu_mA1p0.root"),
    DBremBiasFactor = cms.untracked.double(100) 
)
