import FWCore.ParameterSet.Config as cms

from GeneratorInterface.Pythia6Interface.pythiaDefault_cff import *
generator = cms.EDProducer("Pythia6PtYDistGun",
                           maxEventsToPrint = cms.untracked.int32(0),
                           pythiaHepMCVerbosity = cms.untracked.bool(False),
                           pythiaPylistVerbosity = cms.untracked.int32(0),
                           
                           PGunParameters = cms.PSet(
        ParticleID = cms.vint32(553),
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
        upsilonDecay = cms.vstring('BRAT(1034) = 0 ! switch off',
                                   'BRAT(1035) = 1 ! switch on',
                                   'BRAT(1036) = 0 ! switch off',
                                   'BRAT(1037) = 0 ! switch off',
                                   'BRAT(1038) = 0 ! switch off',
                                   'BRAT(1039) = 0 ! switch off',
                                   'BRAT(1040) = 0 ! switch off',
                                   'BRAT(1041) = 0 ! switch off',
                                   'BRAT(1042) = 0 ! switch off',
                                   'MDME(1034,1) = 0 ! switch off',
                                   'MDME(1035,1) = 1 ! switch on',
                                   'MDME(1036,1) = 0 ! switch off',
                                   'MDME(1037,1) = 0 ! switch off',
                                   'MDME(1038,1) = 0 ! switch off',
                                   'MDME(1039,1) = 0 ! switch off',
                                   'MDME(1040,1) = 0 ! switch off',
                                   'MDME(1041,1) = 0 ! switch off',
                                   'MDME(1042,1) = 0 ! switch off'
                                   ),
        parameterSets = cms.vstring('pythiaDefault',
                                    'upsilonDecay')
        )
                           )

