import sys, os
sys.path.append(os.environ.get("CMSSW_BASE") + "/src/VHbbAnalysis/Heppy/test")

from vhbb_combined import *
components = [
    cfg.MCComponent(
        files = [
            "root://xrootd-cms.infn.it///store/mc/RunIISpring16MiniAODv2/ttHTobb_M125_13TeV_powheg_pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14-v1/40000/0089CC67-6338-E611-947D-0025904C4E2A.root"
        ],
        name = "tth_hbb",
        isMC = True
    ),
    cfg.MCComponent(
        files = [
            "root://xrootd-cms.infn.it///store/mc/RunIISpring16MiniAODv2/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/PUSpring16RAWAODSIM_reHLT_80X_mcRun2_asymptotic_v14_ext3-v1/00000/0064B539-803A-E611-BDEA-002590D0B060.root"
        ],
        name = "ttjets",
        isMC = True
    )
]

if __name__ == '__main__':
    from PhysicsTools.HeppyCore.framework.looper import Looper
    for comp in components:
        print "processing",comp
        config.components = [comp] 
        looper = Looper( 'Loop_validation_tth_sl_dl_' + comp.name, config, nPrint = 0, nEvents = 1000)
        looper.loop()
        looper.write()
