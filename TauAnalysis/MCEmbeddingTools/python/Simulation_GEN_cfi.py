"""
This config fragment is needed by the GEN step to modify pythia, which is used to simulate the hadronization of the tau leptons (or electrons/muons). The LHE and cleaning step must be carried out beforehand.
With `--procModifiers` one can specify the final states of the tau decays or if muons (`tau_embedding_mu_to_mu`) or electrons (`tau_embedding_mu_to_e`) are simulated/embedded instead of taus.
To use this config fragment, a cmsDriver command like the following can be used:
```
cmsDriver.py TauAnalysis/MCEmbeddingTools/python/Simulation_GEN_cfi.py \
	--step GEN,SIM,DIGI,L1,DIGI2RAW \
	--processName SIMembeddingpreHLT \
	--mc \
	--beamspot DBrealistic \
	--geometry DB:Extended \
	--eventcontent TauEmbeddingSimGen \
	--datatier RAWSIM \
	--procModifiers tau_embedding_sim,tau_embedding_mu_to_mu \
    --era ... \
    --conditions ... \
    --filein ... \
    --fileout ...
```
"""

import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_run2_common_cff import run2_common
from Configuration.Eras.Modifier_run3_common_cff import run3_common
from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.Pythia8CUEP8M1Settings_cfi import *
from Configuration.ProcessModifiers.tau_embedding_emu_cff import tau_embedding_emu
from Configuration.ProcessModifiers.tau_embedding_etauh_cff import tau_embedding_etauh
from Configuration.ProcessModifiers.tau_embedding_mu_to_e_cff import (
    tau_embedding_mu_to_e,
)
from Configuration.ProcessModifiers.tau_embedding_mu_to_mu_cff import (
    tau_embedding_mu_to_mu,
)
from Configuration.ProcessModifiers.tau_embedding_mutauh_cff import (
    tau_embedding_mutauh,
)
from Configuration.ProcessModifiers.tau_embedding_tauhtauh_cff import (
    tau_embedding_tauhtauh,
)
from GeneratorInterface.ExternalDecays.TauolaSettings_cff import *

# Set the the vertex used for the simulation to the measured vertex.
# replaced in IOMC/EventVertexGenerators/python/VtxSmearedRealistic_cfi.py
tau_embedding_vtx_corrected_to_input = cms.EDProducer(
    "EmbeddingVertexCorrector", src=cms.InputTag("generator", "unsmeared")
)

# As we only want to simulate the the taus, we have to disable the noise simulation.
# Those settings can be set in the digitizers of the mixing module (SimGeneral/MixingModule/python/mixNoPU_cfi.py, because we don't have PU)
# Because the settings there are also imported, this is done by process modifiers in the following files:
# SimGeneral/MixingModule/python/SiPixelSimParameters_cfi.py
# SimGeneral/MixingModule/python/SiStripSimParameters_cfi.py
# SimGeneral/MixingModule/python/ecalDigitizer_cfi.py
# SimGeneral/MixingModule/python/hcalDigitizer_cfi.py
# SimGeneral/MixingModule/python/castorDigitizer_cfi.py # castor was removed in run3, so this is only needed for run2

# The generator module is expected by the GEN step to specify the event generator
generator = cms.EDFilter(
    "Pythia8HadronizerFilter",
    maxEventsToPrint=cms.untracked.int32(1),
    nAttempts=cms.uint32(1000),
    HepMCFilter=cms.PSet(
        filterName=cms.string("EmbeddingHepMCFilter"),
        filterParameters=cms.PSet(
            ElElCut=cms.string("El1.Pt > 22 && El2.Pt > 10"),
            ElHadCut=cms.string("El.Pt > 28 && Had.Pt > 25"),
            ElMuCut=cms.string(
                "(El.Pt > 21 && Mu.Pt > 10) || (El.Pt > 10 && Mu.Pt > 21)"
            ),
            HadHadCut=cms.string("Had1.Pt > 35 && Had2.Pt > 30"),
            MuHadCut=cms.string("Mu.Pt > 18 && Had.Pt > 25 && Mu.Eta < 2.1"),
            MuMuCut=cms.string("Mu1.Pt > 17 && Mu2.Pt > 8"),
            Final_States=cms.vstring(
                "ElEl", "ElHad", "ElMu", "HadHad", "MuHad", "MuMu"
            ),
            BosonPDGID=cms.int32(23),
            IncludeDY=cms.bool(False),
        ),
    ),
    pythiaPylistVerbosity=cms.untracked.int32(1),
    filterEfficiency=cms.untracked.double(1.0),
    pythiaHepMCVerbosity=cms.untracked.bool(False),
    comEnergy=cms.double(13600.0),
    crossSection=cms.untracked.double(1.0),
    PythiaParameters=cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CUEP8M1SettingsBlock,
        processParameters=cms.vstring(
            "JetMatching:merge = off",
            "Init:showChangedSettings = off",
            "Init:showChangedParticleData = off",
            "ProcessLevel:all = off",
        ),
        parameterSets=cms.vstring(
            "pythia8CommonSettings", "pythia8CUEP8M1Settings", "processParameters"
        ),
    ),
)
# Set the correct center of mass energy for run2
(run2_common & ~run3_common).toModify(generator, comEnergy=cms.double(13000.0))

## This modifier sets the correct cuts for mu->mu embedding
tau_embedding_mu_to_mu.toModify(
    generator,
    HepMCFilter={
        "filterParameters": {
            "MuMuCut": cms.string(
                "(Mu1.Pt > 17 && Mu2.Pt > 8 && Mu1.Eta < 2.5 && Mu2.Eta < 2.5)"
            ),
            "Final_States": cms.vstring("MuMu"),
        }
    },
    # disable final state radiation for mu->mu embedding
    PythiaParameters={
        "processParameters": cms.vstring(
            "JetMatching:merge = off",
            "Init:showChangedSettings = off",
            "Init:showChangedParticleData = off",
            "ProcessLevel:all = off",
            "PartonLevel:FSR = off",
        )
    },
)
# only one simulation needed, as the muons don't decay like taus and no wights need to be calculated.
tau_embedding_mu_to_mu.toModify(generator, nAttempts=cms.uint32(1))

# This modifier sets the correct cuts for mu->e embedding
tau_embedding_mu_to_e.toModify(
    generator,
    HepMCFilter={
        "filterParameters": {
            "ElElCut": cms.string(
                "(El1.Pt > 22 && El2.Pt > 10 && El1.Eta < 2.5 && El2.Eta < 2.5)"
            ),
            "Final_States": cms.vstring("ElEl"),
        }
    },
    # disable final state radiation for mu->e embedding
    PythiaParameters={
        "processParameters": cms.vstring(
            "JetMatching:merge = off",
            "Init:showChangedSettings = off",
            "Init:showChangedParticleData = off",
            "ProcessLevel:all = off",
            "PartonLevel:FSR = off",
        )
    },
)
# only one simulation needed, as the electrons don't decay like taus and no wights need to be calculated.
tau_embedding_mu_to_e.toModify(generator, nAttempts=cms.uint32(1))

# This modifier sets the correct cuts for the taus decaying into one jet and one muon
tau_embedding_mutauh.toModify(
    generator,
    HepMCFilter={
        "filterParameters": {
            "MuHadCut": cms.string(
                "(Mu.Pt > 18 && Had.Pt > 18 && Mu.Eta < 2.2 && Had.Eta < 2.4)"
            ),
            "Final_States": cms.vstring("MuHad"),
        }
    },
)

# This modifier sets the correct cuts for the taus decaying into one jet and one electron
tau_embedding_etauh.toModify(
    generator,
    HepMCFilter={
        "filterParameters": {
            "ElHadCut": cms.string(
                "(El.Pt > 18 && Had.Pt > 18 && El.Eta < 2.2 && Had.Eta < 2.4)"
            ),
            "Final_States": cms.vstring("ElHad"),
        }
    },
)

# This modifier sets the correct cuts for the taus decaying into one electron and one muon
tau_embedding_emu.toModify(
    generator,
    HepMCFilter={
        "filterParameters": {
            "ElMuCut": cms.string(
                "(El.Pt > 9 && Mu.Pt > 19 && El.Eta < 2.5 && Mu.Eta < 2.5) || El.Pt > 19 && Mu.Pt > 9 && El.Eta < 2.5 && Mu.Eta < 2.5)"
            ),
            "Final_States": cms.vstring("ElMu"),
        }
    },
)

# This modifier sets the correct cuts for the taus decaying into two jets
tau_embedding_tauhtauh.toModify(
    generator,
    HepMCFilter={
        "filterParameters": {
            "HadHadCut": cms.string(
                "(Had1.Pt > 20 && Had2.Pt > 20 && Had1.Eta < 2.2 && Had2.Eta < 2.2)"
            ),
            "Final_States": cms.vstring("HadHad"),
        }
    },
)
