from PhysicsTools.NanoAODTools.postprocessing.framework.postprocessor import PostProcessor
from SimPPS.DirectSimProducer.NanoAODDirectSimProducer import *
from importlib import import_module
import os
import sys
import ROOT

#from SimPPS.DirectSimProducer.simPPS2018_cfi import *
from SimPPS.DirectSimProducer.simPPS2022_cfi import *


ROOT.PyConfig.IgnoreCommandLineOptions = True

#fnames = ["root://eoscms.cern.ch//eos/cms/store/user/cmsbuild/store/group/cat/datasets/NANOAODSIM/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/7B930101-EB91-4F4E-9B90-0861460DBD94.root"]
#fnames = ["root://eoscms.cern.ch//eos/cms/store/data/Run2025A/HLTPhysics/NANOAOD/PromptReco-v2/000/391/312/00000/34d2039d-c580-4e19-bacc-3b6b115dc20c.root"]
#fnames = ["root://eoscms.cern.ch//eos/cms/store/mc/Run3Summer23NanoAODv12/GGToMuMu_PT-25_Inel-El_13p6TeV_superchic/NANOAODSIM/NoPU_130X_mcRun3_2023_realistic_v14-v1/70000/83268898-0f98-443d-b080-bc6cc7d5b298.root"]
fnames = ["root://eoscms.cern.ch//eos/cms/store/mc/Run3Winter22NanoAOD/GGToMuMu_Pt-25_El-El_13p6TeV-lpair/NANOAODSIM/122X_mcRun3_2021_realistic_v9-v3/30000/e29c388c-4a6d-46ac-abe4-4c8529361c24.root"]

profile_2022_default.ctppsLHCInfo.xangle = 130.

p = PostProcessor(outputDir=".",
                  inputFiles=fnames,
                  cut="",
                  #modules=[directSimModuleConstr(profile_2018_postTS2)],
                  modules=[directSimModuleConstr(profile_2022_default, verbosity=1)],
                  provenance=True,
                  maxEntries=5000, #just read the first maxEntries events
                  )
p.run()
