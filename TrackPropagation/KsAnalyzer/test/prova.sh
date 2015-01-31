cd /afs/cern.ch/work/l/lviliani/Geant4e_G4-9.5/CMSSW_5_3_17/src/TrackPropagation/KsAnalyzer/test 

eval `scram runtime -sh`
export X509_USER_PROXY=/afs/cern.ch/user/l/lviliani/proxy.proxy

cmsRun KsAnalysis_75_cfg.py &> err_500evts.log 
