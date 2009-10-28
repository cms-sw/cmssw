import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PythiaUESettings_cfi import *
from GeneratorInterface.ExternalDecays.TauolaSettings_cff import *

# Note: currently this is just a sketch and should not be used

newSource = cms.EDProducer("MCParticleReplacer",
    src                = cms.InputTag("selectedMuons"),
    beamSpotSrc        = cms.InputTag("dummy"),
    primaryVertexLabel = cms.InputTag("dummy"),
    hepMcSrc           = cms.InputTag("generator"),

    algorithm = cms.string("ParticleGun"), # "ParticleGun", "ZTauTau"
    hepMcMode = cms.string("new"),         # "new" for new HepMCProduct with taus and decay products,
                                           # "replace" for replacing muons in the existing HepMCProcuct
    verbose = cms.untracked.bool(False),

    ParticleGun = cms.PSet(
        gunParticle	      = cms.int32(15),
        particleOrigin        = cms.string("muonReferencePoint"), # "primaryVertex", "muonReferencePoint"
        forceTauPolarization  = cms.string("W"), # "W", "H+", "h", "H", "A"
        forceTauDecay	      = cms.string("none"), # "none", "hadrons", "1prong", "3prong"
        forceTauPlusHelicity  = cms.int32(0),
        forceTauMinusHelicity = cms.int32(0),
        generatorMode = cms.string("Tauola"),  # "Tauola", "Pythia" (not implemented yet)
        ExternalDecays = cms.PSet(
            Tauola = cms.PSet(
                TauolaPolar,
                TauolaDefaultInputCards
            ),
            parameterSets = cms.vstring('Tauola')
        ),
        PythiaParameters = cms.PSet(
            pythiaUESettingsBlock,
            pgunTauolaParameters = cms.vstring(["MDME(%d,1)=0" % x for x in range(89, 143)]),
            parameterSets = cms.vstring("pythiaUESettings")
        ),
    ),

    ZTauTau = cms.PSet(
        filterEfficiency = cms.untracked.double(1.0),
        pythiaHepMCVerbosity = cms.untracked.bool(False),
        generatorMode = cms.string("Tauola"),  # "Tauola", "Pythia" (not implemented yet)
        ExternalDecays = cms.PSet(
            Tauola = cms.PSet(
                TauolaPolar,
                cms.PSet(
                    InputCards = cms.vstring('TAUOLA = 0 0 102 ! TAUOLA ')      # 114=l+jet, 102=only muons
                )
            ),
            parameterSets = cms.vstring('Tauola')
        ),
        PythiaParameters = cms.PSet(
            pythiaUESettingsBlock,
            ZtautauParameters = cms.vstring('MSEL         = 11 ', 
                'MDME( 174,1) = 0    !Z decay into d dbar', 
                'MDME( 175,1) = 0    !Z decay into u ubar', 
                'MDME( 176,1) = 0    !Z decay into s sbar', 
                'MDME( 177,1) = 0    !Z decay into c cbar', 
                'MDME( 178,1) = 0    !Z decay into b bbar', 
                'MDME( 179,1) = 0    !Z decay into t tbar', 
                'MDME( 182,1) = 0    !Z decay into e- e+', 
                'MDME( 183,1) = 0    !Z decay into nu_e nu_ebar', 
                'MDME( 184,1) = 0    !Z decay into mu- mu+', 
                'MDME( 185,1) = 0    !Z decay into nu_mu nu_mubar', 
                'MDME( 186,1) = 1    !Z decay into tau- tau+', 
                'MDME( 187,1) = 0    !Z decay into nu_tau nu_taubar', 
                'CKIN( 1)     = 40.  !(D=2. GeV)', 
                'CKIN( 2)     = -1.  !(D=-1. GeV)',
                'MSTJ(28) = 1          ! no tau decays in pythia, use tauola instead'),
            parameterSets = cms.vstring("pythiaUESettings", "ZtautauParameters")
        ),
    )
)

# Disable tau decays in Pythia for particle gun
def customise(process):
    if process.newSource.generatorMode.value() != "Pythia" and abs(process.newSource.ParticleGun.gunParticle.value()) == 15:
        process.newSource.ParticleGun.PythiaParameters.parameterSets.append("pgunTauolaParameters")
