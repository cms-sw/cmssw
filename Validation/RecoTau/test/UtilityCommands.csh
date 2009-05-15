
export PYTHONPATH=$CMSSW_BASE/src/Validation/RecoTau/Tools:$PYTHONPATH
export VALTOOLS=$CMSSW_BASE/src/Validation/RecoTau/Tools
export VALTEST=$CMSSW_BASE/src/Validation/RecoTau/test
export PATH=$VALTOOLS:$PATH

alias MergeBatchJob='cmsRun $VALTOOLS/MergeFilesAndCalculateEfficiencies_cfg.py `ls BatchJobs/*_0.root | sed "s|_0.root|.root|" | sed "s|BatchJobs/||"` BatchJobs/*root'
alias Compare='cmsRun $VALTOOLS/Compare_cfg.py'
alias Summarize='cmsRun $VALTOOLS/SummaryPlots_cfg.py'
alias PerformanceCurves='python $VALTOOLS/PlotPerformanceCurves.py'
alias BuildWebpage='python $VALTOOLS/BuildWebpage.py'


