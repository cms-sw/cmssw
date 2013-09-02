## import skeleton process
from PhysicsTools.PatAlgos.patTemplate_cfg import *


## ------------------------------------------------------
#  NOTE: you can use a bunch of core tools of PAT to
#  taylor your PAT configuration; for a few examples
#  uncomment the lines below
## ------------------------------------------------------
#from PhysicsTools.PatAlgos.tools.coreTools import *

## remove MC matching from the default sequence
# removeMCMatching(process, ['Muons'])
# runOnData(process)

## remove certain objects from the default sequence
# removeAllPATObjectsBut(process, ['Muons'])
# removeSpecrecoTauClassicHPSSeqificPATObjects(process, ['Electrons', 'Muons', 'Taus'])

process.load("RecoTauTag.Configuration.RecoPFTauTag_cff")

from PhysicsTools.PatAlgos.tools.tauTools import *

switchToPFTauHPS(process)

## let it run
process.p = cms.Path(
    process.recoTauClassicHPSSequence*
    process.patDefaultSequence
    )

## ------------------------------------------------------
#  In addition you usually want to change the following
#  parameters:
## ------------------------------------------------------
#
#   process.GlobalTag.globaltag =  ...    ##  (according to https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideFrontierConditions)
#                                         ##
#from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValProdTTbarAODSIM
process.source.fileNames = cms.untracked.vstring('/store/relval/CMSSW_5_3_6-START53_V14/RelValProdTTbar/AODSIM/v2/00000/76ED0FA6-1E2A-E211-B8F1-001A92971B72.root')
#                                         ##
process.maxEvents.input = 100
#                                         ##
#   process.out.outputCommands = [ ... ]  ##  (e.g. taken from PhysicsTools/PatAlgos/python/patEventContent_cff.py)
#                                         ##
process.out.fileName = 'patTuple_standard.root'
#                                         ##
#   process.options.wantSummary = False   ##  (to suppress the long output at the end of the job)

