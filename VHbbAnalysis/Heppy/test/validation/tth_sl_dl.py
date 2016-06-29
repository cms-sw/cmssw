import sys, os
sys.path.append(os.environ.get("CMSSW_BASE") + "/src/VHbbAnalysis/Heppy/test")

from vhbb_combined import *
components = [
    cfg.MCComponent(
        files = [
            "root://xrootd-cms.infn.it///store/mc/RunIISpring15DR74/ttHTobb_M125_13TeV_powheg_pythia8/MINIAODSIM/Asympt25ns_MCRUN2_74_V9-v1/00000/141B9915-1F08-E511-B9FF-001E675A6AB3.root"
        ],
        name = "tth_hbb",
        isMC = True
    ),
    cfg.MCComponent(
        files = [
            "root://xrootd-cms.infn.it///store/mc/RunIISpring15DR74/TT_TuneCUETP8M1_13TeV-powheg-pythia8/MINIAODSIM/Asympt25ns_MCRUN2_74_V9-v2/00000/0AB045B5-BB0C-E511-81FD-0025905A60B8.root"
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
        looper = Looper( 'Loop_validation_tth_sl_dl_' + comp.name, config, nPrint = 0, nEvents = 10)
        looper.loop()
        looper.write()
