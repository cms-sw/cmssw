export SCRAM_ARCH=slc6_amd64_gcc630
cmsrel CMSSW_9_4_11
cd CMSSW_9_4_11
eval `scramv1 runtime -sh`
git cms-addpkg DQMOffline/PFTau
git cms-addpkg DQMOffline/Configuration
git cms-addpkg PhysicsTools/NanoAOD
git cms-addpkg Validation/Configuration
git cms-addpkg Validation/RecoParticleFlow
git cms-addpkg Validation/RecoTrack
git cms-addpkg Configuration/PyReleaseValidation
