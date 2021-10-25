
setenv PYTHONPATH ${CMSSW_BASE}/src/Validation/RecoTau/Tools:${PYTHONPATH}
setenv VALTOOLS ${CMSSW_BASE}/src/Validation/RecoTau/Tools
setenv VALTEST ${CMSSW_BASE}/src/Validation/RecoTau/test
setenv PFVALTOOLS ${CMSSW_RELEASE_BASE}/src/Validation/RecoParticleFlow/Benchmarks/Tools
setenv PYTHONPATH ${PFVALTOOLS}:${PYTHONPATH}
setenv PATH ${VALTOOLS}:${PATH}
setenv PastResults /afs/cern.ch/cms/Physics/tau/Validation/cms-project-tauvalidation/releases/

cmsenv

alias MergeBatchJob 'cmsRun ${VALTOOLS}/MergeFilesAndCalculateEfficiencies_cfg.py `ls BatchJobs/*_0.root | sed "s|_0.root|.root|" | sed "s|BatchJobs/||"` BatchJobs/*root'
alias MergeGridJob 'cmsRun ${VALTOOLS}/MergeFilesAndCalculateEfficiencies_cfg.py'
alias Compare 'cmsRun ${VALTOOLS}/Compare_cfg.py'
alias Summarize 'cmsRun ${VALTOOLS}/SummaryPlots_cfg.py'
alias PerformanceCurves 'python3 ${VALTOOLS}/PlotPerformanceCurves.py'
alias MultipleCompare 'python3 ${VALTOOLS}/MultipleCompare.py'
alias BuildWebpage 'python3 ${VALTOOLS}/BuildWebpage.py'
alias SubmitResults 'python3 ${PFVALTOOLS}/submit.py'
alias getZTTRecoFiles 'dbs search --query="find dataset where release = CMSSW_3_1_0_pre10 and primds = RelValZTT and tier like *RECO* and dataset not like *FastSim*" --createCFF=tempZTTRecoFiles'
alias removeLogs 'rm -r $VALTEST/LSFJOB_[0-9]*'

