import FWCore.ParameterSet.Config as cms

from GeneratorInterface.Pythia6Interface.pythiaDefault_cff import *
generator = cms.EDProducer("Pythia6PtYDistGun",
                           maxEventsToPrint = cms.untracked.int32(0),
                           pythiaHepMCVerbosity = cms.untracked.bool(False),
                           pythiaPylistVerbosity = cms.untracked.int32(0),
                           
                           PGunParameters = cms.PSet(
        ParticleID = cms.vint32(23),
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
        zDecay = cms.vstring("MDME(174,1)=0",          # !Z decay into d dbar,
                             "MDME(175,1)=0",          # !Z decay into u ubar,
                             "MDME(176,1)=0",          # !Z decay into s sbar,
                             "MDME(177,1)=0",          # !Z decay into c cbar,
                             "MDME(178,1)=0",          # !Z decay into b bbar,
                             "MDME(179,1)=0",          # !Z decay into t tbar,
                             "MDME(182,1)=0",          # !Z decay into e- e+,
                             "MDME(183,1)=0",          # !Z decay into nu_e nu_ebar,
                             "MDME(184,1)=1",          # !Z decay into mu- mu+,
                             "MDME(185,1)=0",          # !Z decay into nu_mu nu_mubar,
                             "MDME(186,1)=0",          # !Z decay into tau- tau+,
                             "MDME(187,1)=0"           # !Z decay into nu_tau nu_taubar
                             ),
        parameterSets = cms.vstring('pythiaDefault',
                                    'zDecay')
        )
                           )
