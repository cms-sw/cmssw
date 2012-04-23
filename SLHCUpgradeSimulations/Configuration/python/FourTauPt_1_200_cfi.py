import FWCore.ParameterSet.Config as cms

# Modified from FastSimulation/Configuration/python/PythiaTauGun_cfi.py
# Altered to have similar characteristics to our other particle guns
source = cms.Source("EmptySource")

generator = cms.EDProducer("Pythia6PtGun",
    pythiaVerbosity = cms.untracked.bool(False),
    PGunParameters = cms.PSet(
        ParticleID = cms.vint32(-15,-15),
        AddAntiParticle = cms.bool(True),
        MinPt = cms.double(0.9),
        MaxPt = cms.double(200.0),
        MinEta = cms.double(-2.5),
        MaxEta = cms.double(2.5),
        MinPhi = cms.double(-3.1415926535897931),
        MaxPhi = cms.double(3.1415926535897931)
    ),
    PythiaParameters = cms.PSet(
        # This is a vector of ParameterSet names to be read, in this order
        parameterSets = cms.vstring(
            'pythiaTauJets'
        ),

        # Tau jets only
        pythiaTauJets = cms.vstring(
            'MDME(89,1)=0      ! no tau->electron', 
            'MDME(90,1)=0      ! no tau->muon'
        )
    )
)

ProductionFilterSequence = cms.Sequence(generator)
