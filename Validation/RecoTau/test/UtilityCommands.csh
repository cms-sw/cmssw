
export PYTHONPATH=$CMSSW_BASE/src/Validation/RecoTau/Tools:$PYTHONPATH
export VALTOOLS=$CMSSW_BASE/src/Validation/RecoTau/Tools
export VALTEST=$CMSSW_BASE/src/Validation/RecoTau/test
export PFVALTOOLS=$CMSSW_RELEASE_BASE/src/Validation/RecoParticleFlow/Benchmarks/Tools
export PYTHONPATH=$PFVALTOOLS:$PYTHONPATH
export PATH=$VALTOOLS:$PATH

#temp workaround for dbs command
export ADSHOME=$VALTEST
source /afs/cern.ch/cms/sw/slc4_ia32_gcc345/cms/dbs-client/DBS_2_0_6/lib/setup.sh
cmsenv

export ADSHOME=$VALTEST

alias MergeBatchJob='cmsRun $VALTOOLS/MergeFilesAndCalculateEfficiencies_cfg.py `ls BatchJobs/*_0.root | sed "s|_0.root|.root|" | sed "s|BatchJobs/||"` BatchJobs/*root'
alias Compare='cmsRun $VALTOOLS/Compare_cfg.py'
alias Summarize='cmsRun $VALTOOLS/SummaryPlots_cfg.py'
alias PerformanceCurves='python $VALTOOLS/PlotPerformanceCurves.py'
alias BuildWebpage='python $VALTOOLS/BuildWebpage.py'
alias SubmitResults='python $PFVALTOOLS/submit.py'
alias getZTTRecoFiles='dbs search --query="find dataset where release = CMSSW_3_1_0_pre10 and primds = RelValZTT and tier like *RECO* and dataset not like *FastSim*" --createCFF=tempZTTRecoFiles'

