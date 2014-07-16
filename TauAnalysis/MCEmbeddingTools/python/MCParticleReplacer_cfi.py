import FWCore.ParameterSet.Config as cms

from Configuration.Generator.PythiaUESettings_cfi import *
from GeneratorInterface.ExternalDecays.TauolaSettings_cff import *

generator = cms.EDProducer("MCParticleReplacer",
    src                = cms.InputTag("selectedMuons"),
    beamSpotSrc        = cms.InputTag("dummy"),
    primaryVertexLabel = cms.InputTag("dummy"),
    hepMcSrc           = cms.InputTag("generator"),

    algorithm          = cms.string("ParticleGun"), # "ParticleGun", "Ztautau"
    pluginType         = cms.string("ParticleReplacerParticleGun"), # "ParticleReplacerParticleGun", "ParticleReplacerZtautau"                       
    hepMcMode          = cms.string("new"),         # "new" for new HepMCProduct with taus and decay products,
                                                    # "replace" for replacing muons in the existing HepMCProcuct                           
    verbose            = cms.bool(False),

    ParticleGun = cms.PSet(
        gunParticle	      = cms.int32(15),
        particleOrigin        = cms.string("muonReferencePoint"), # "primaryVertex", "muonReferencePoint"
        forceTauPolarization  = cms.string("W"), # "W", "H+", "h", "H", "A"
        forceTauDecay	      = cms.string("none"), # "none", "hadrons", "1prong", "3prong"
        forceTauPlusHelicity  = cms.int32(0),
        forceTauMinusHelicity = cms.int32(0),
        generatorMode = cms.string("Tauola"),  # "Tauola", "Pythia" (not implemented yet)
        enablePhotosFSR = cms.bool(False),                           
        ExternalDecays = cms.PSet(
            Tauola = cms.PSet(
                TauolaNoPolar,
                TauolaDefaultInputCards
            ),
            parameterSets = cms.vstring('Tauola')
        ),
        PythiaParameters = cms.PSet(
            pythiaUESettingsBlock,
            pgunTauolaParameters = cms.vstring(["MDME(%d,1)=0" % x for x in range(89, 143)]),
            parameterSets = cms.vstring("pythiaUESettings")
        )
    ),

    Ztautau = cms.PSet(
        TauolaOptions = cms.PSet(
            TauolaPolar,
            InputCards = cms.PSet(
                pjak1 = cms.int32(0),
                pjak2 = cms.int32(0),
                mdtau = cms.int32(0)
            )
        ),
        PythiaParameters = cms.PSet(
            pythiaUESettings = cms.vstring(
                'MSTJ(11)=3     ! Choice of the fragmentation function',
                'MSTJ(22)=2     ! Decay those unstable particles',
                'PARJ(71)=10 .  ! for which ctau  10 mm',
                'MSTP(2)=1      ! which order running alphaS',
                'MSTP(33)=0     ! no K factors in hard cross sections',
                'MSTP(51)=10042 ! structure function chosen (external PDF CTEQ6L1)',
                'MSTP(52)=2     ! work with LHAPDF',
                'MSTP(81)=1     ! multiple parton interactions 1 is Pythia default',
                'MSTP(82)=4     ! Defines the multi-parton model',
                'MSTU(21)=1     ! Check on possible errors during program execution',
                'PARP(82)=1.8387   ! pt cutoff for multiparton interactions',
                'PARP(89)=1960. ! sqrts for which PARP82 is set',
                'PARP(83)=0.5   ! Multiple interactions: matter distrbn parameter',
                'PARP(84)=0.4   ! Multiple interactions: matter distribution parameter',
                'PARP(90)=0.16  ! Multiple interactions: rescaling power',
                'PARP(67)=2.5    ! amount of initial-state radiation',
                'PARP(85)=1.0  ! gluon prod. mechanism in MI',
                'PARP(86)=1.0  ! gluon prod. mechanism in MI',
                'PARP(62)=1.25   ! ',
                'PARP(64)=0.2    ! ',
                'MSTP(91)=1      !',
                'PARP(91)=2.1   ! kt distribution',
                'PARP(93)=15.0  ! '
            ),
            parameterSets = cms.vstring('pythiaUESettings')
        ),
        PhotosOptions = cms.PSet(),
        filterEfficiency = cms.untracked.double(1.0),
        beamEnergy = cms.double(4000.), # GeV
        rfRotationAngle = cms.double(0.),
	rfMirror = cms.bool(True),
        applyMuonRadiationCorrection = cms.string(""),
        enablePhotosFSR = cms.bool(False),
        pythiaHepMCVerbosity = cms.untracked.bool(False),
        generatorMode = cms.string("Tauola"),  # "Tauola", "Pythia" (not implemented yet)
        verbosity = cms.int32(0)
    )
)

# Disable tau decays in Pythia for particle gun
def customise(process):
    if process.generator.generatorMode.value() != "Pythia" and abs(process.generator.ParticleGun.gunParticle.value()) == 15:
        process.generator.ParticleGun.PythiaParameters.parameterSets.append("pgunTauolaParameters")
