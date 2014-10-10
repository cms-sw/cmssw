import FWCore.ParameterSet.Config as cms

from GeneratorInterface.Pythia6Interface.pythiaDefault_cff import *
generator = cms.EDProducer("Pythia6PtYDistGun",
                           maxEventsToPrint = cms.untracked.int32(0),
                           pythiaHepMCVerbosity = cms.untracked.bool(False),
                           pythiaPylistVerbosity = cms.untracked.int32(0),
                           
                           PGunParameters = cms.PSet(
        ParticleID = cms.vint32(443),
        kinematicsFile = cms.FileInPath('HeavyIonsAnalysis/Configuration/data/jpsipbpbFlat.root'),
        PtBinning = cms.int32(100000),
        YBinning = cms.int32(500), 
        MinPt = cms.double(0.0),
        MaxPt = cms.double(20.0),
        MinY = cms.double(-2.4),
        MaxY = cms.double(2.4),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        AddAntiParticle = cms.bool(False)
        ),
                           PythiaParameters = cms.PSet(
        pythiaDefaultBlock,
        jpsiDecay = cms.vstring('BRAT(858) = 0 ! switch off',
                                'BRAT(859) = 1 ! switch on',
                                'BRAT(860) = 0 ! switch off',
                                'MDME(858,1) = 0 ! switch off',
                                'MDME(859,1) = 1 ! switch on',
                                'MDME(860,1) = 0 ! switch off'
                                ),
        parameterSets = cms.vstring('pythiaDefault',
                                    'jpsiDecay')
        )
                           )
