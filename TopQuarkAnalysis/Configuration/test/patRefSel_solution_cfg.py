
"""
Build your CMSSW area by copying & pasting the following command lines (one go):

setenv SCRAM_ARCH slc5_amd64_gcc462
cmsrel CMSSW_5_2_6
cd CMSSW_5_2_6/src
cmsenv
cvs co -r V07-00-00    TopQuarkAnalysis/Configuration
cvs co                 TopQuarkAnalysis/Configuration/test/patRefSel_exercise_cfg.py
cvs co                 TopQuarkAnalysis/Configuration/test/patRefSel_solution_cfg.py
cvs co -r V06-05-01    DataFormats/PatCandidates
cvs co -r V08-09-11-02 PhysicsTools/PatAlgos
cvs co -r V00-03-14    CommonTools/ParticleFlow
cvs co -r V06-07-11-01 TopQuarkAnalysis/TopTools
cvs co -r V00-00-13 -d EGamma/EGammaAnalysisTools UserCode/EGamma/EGammaAnalysisTools
cd EGamma/EGammaAnalysisTools/data
cat download.url | xargs wget
cd -
scram b -j 9

"""

# Load existing PAT skeleton configuration

from PhysicsTools.PatAlgos.patTemplate_cfg import *
from PhysicsTools.PatAlgos.patInputFiles_cff import filesRelValProdTTbarAODSIM
process.source.fileNames = filesRelValProdTTbarAODSIM
process.options.wantSummary = True
process.out.fileName        = 'patRefSel_solution.root'

process.maxEvents.input = 1000 # reduce for testing


# Activate PF2PAT

process.load( "PhysicsTools.PatAlgos.patSequences_cff" )

from PhysicsTools.PatAlgos.tools.pfTools import *
usePF2PAT( process
         , runPF2PAT = True
         , jetAlgo   = 'AK5'
         , runOnMC   = True
         , postfix   = ''
         )


### Exercise 1 ###

# Switch the isolation cone of the muons
# from 0.4 (default)
# to   0.3
# An example is found, if you follow the parameter "pfMuonIsoConeR03" in
# TopQuarkAnalysis/Configuration/test/patRefSel_muJets_cfg.py
# under the assumption that it is set to 'True'
# for "runPF2PAT" == 'True'
# Consider, that here no "postfix" is set

process.pfIsolatedMuons.isolationValueMapsCharged  = cms.VInputTag( cms.InputTag( 'muPFIsoValueCharged03' )
                                                                  )
process.pfIsolatedMuons.deltaBetaIsolationValueMap = cms.InputTag( 'muPFIsoValuePU03' )
process.pfIsolatedMuons.isolationValueMapsNeutral  = cms.VInputTag( cms.InputTag( 'muPFIsoValueNeutral03' )
                                                                  , cms.InputTag( 'muPFIsoValueGamma03' )
                                                                  )
process.pfMuons.isolationValueMapsCharged  = cms.VInputTag( cms.InputTag( 'muPFIsoValueCharged03' )
                                                          )
process.pfMuons.deltaBetaIsolationValueMap = cms.InputTag( 'muPFIsoValuePU03' )
process.pfMuons.isolationValueMapsNeutral  = cms.VInputTag( cms.InputTag( 'muPFIsoValueNeutral03' )
                                                          , cms.InputTag( 'muPFIsoValueGamma03' )
                                                          )
process.patMuons.isolationValues.pfNeutralHadrons   = cms.InputTag( 'muPFIsoValueNeutral03' )
process.patMuons.isolationValues.pfChargedAll       = cms.InputTag( 'muPFIsoValueChargedAll03' )
process.patMuons.isolationValues.pfPUChargedHadrons = cms.InputTag( 'muPFIsoValuePU03' )
process.patMuons.isolationValues.pfPhotons          = cms.InputTag( 'muPFIsoValueGamma03' )
process.patMuons.isolationValues.pfChargedHadrons   = cms.InputTag( 'muPFIsoValueCharged03' )

### Exercise 2 ###

# Include the MVA electron IDs into the PAT electrons.
# The software is already present in your CMSSW area. You only need to modify
# this configuration as shown in the example in
# EGamma/EGammaAnalysisTools/test/patTuple_electronId_cfg.py

process.load('EGamma.EGammaAnalysisTools.electronIdMVAProducer_cfi')
process.mvaID = cms.Sequence(  process.mvaTrigV0 + process.mvaNonTrigV0 )
process.patElectrons.electronIDSources = cms.PSet(
  #MVA
  mvaTrigV0    = cms.InputTag( 'mvaTrigV0' )
, mvaNonTrigV0 = cms.InputTag( 'mvaNonTrigV0' )
)

### Exercise 3 ###

# Apply the veto electron selection, as described in
# https://twiki.cern.ch/twiki/bin/view/CMS/TWikiTopRefEventSel#Electrons
# and the conversion rejection on top of it.
# An example is found, if you follow the parameter "electronCutPF" in
# TopQuarkAnalysis/Configuration/test/patRefSel_muJets_cfg.py

process.selectedPatElectrons.cut = 'pt > 20. && abs(eta) < 2.5 && electronID("mvaTrigV0") > 0. && (chargedHadronIso+max(0.,neutralHadronIso)+photonIso-0.5*puChargedHadronIso)/et < 0.2 && passConversionVeto'

### Exercise 4 ###

# Select only events with at least on electron and four jets.
# To find the corresponding configuration parameters, run this configuration
# interactively:
"""

python -i patRefSel_exercise_cfg.py
>>> process.countPatElectrons
>>> process.countPatJets

"""
# and modify the parameters you find accordingly.
# You should see 25 events passing.

process.countPatElectrons.minNumber = 1
process.countPatJets.minNumber      = 4

# Path

process.p = cms.Path(
  process.mvaID
* process.patPF2PATSequence
)
