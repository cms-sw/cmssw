import sys, os
sys.path.append(os.environ.get("CMSSW_BASE") + "/src/VHbbAnalysis/Heppy/test")

from vhbb_combined_data import *
components = [
#    cfg.MCComponent(
#        files = [
#            "root://xrootd-cms.infn.it///store/data/Run2016B/SingleMuon/MINIAOD/PromptReco-v1/000/272/775/00000/1A8A805A-1916-E611-AE2B-02163E0145EF.root"
#        ],
#        name = "SingleMuon",
#    ),
    cfg.MCComponent(
        files = [
            "root://xrootd-cms.infn.it///store/data/Run2016B/SingleElectron/MINIAOD/PromptReco-v2/000/273/150/00000/0A6284C7-D719-E611-93E6-02163E01421D.root"
        ],
        name = "SingleElectron",
    ),
]

for samp in components:
    samp.isMC = False
    samp.isData = True
    samp.json="json.txt"

if __name__ == '__main__':
    from PhysicsTools.HeppyCore.framework.looper import Looper
    for comp in components:
        print "processing",comp
        config.components = [comp] 
        looper = Looper( 'Loop_validation_tth_sl_dl_' + comp.name, config, nPrint = 0, nEvents = 5000)
        looper.loop()
        looper.write()
