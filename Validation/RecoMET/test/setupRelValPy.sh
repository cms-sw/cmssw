#! /bin/bash

current_area=`pwd`
echo $current_area
# Define the directory that will hold the histogram root files for Full Simulation
# Note: Both Full Sim and Fast Sim will produce histogram root files with the same name, e.g METTester_data_QCD_30-50.root, so they need to be output to different directories!!!

FullSimRootFileDirectory=${current_area}/FullSim/
mkdir $FullSimRootFileDirectory -p

#======= Define list of samples that you will be validating ========#
#dirlist="ZDimu ZprimeDijets QCD_0-15 QCD_15-20 QCD_20-30 QCD_30-50 QCD_50-80 QCD_80-120 QCD_120-170 QCD_170-230 QCD_230-300 QCD_300-380 QCD_380-470 QCD_470-600 QCD_600-800 QCD_800-1000 ttbar QCD_3000-3500"
dirlist="ttbar"

#======= Define list of modules that will be run for each sample ========#
RunPath="fileSaver, calotoweroptmaker, analyzeRecHits, analyzecaloTowers, analyzeGenMET, analyzeGenMETFromGenJets, analyzeHTMET, analyzeCaloMET"


echo "Run path = {" $RunPath "}"


#========Make MaxEvents file===============#
#echo "'untracked PSet maxEvents = {untracked int32 input = -1}'" >> MaxEvents.cfi
#==========================================#
cd $current_area

for i in $dirlist; do


#========Make path file====================#
#echo "untracked vstring fileNames = {
#
#}" >> FilePaths-$i.cfi
#==========================================#
cd $current_area

#mkdir $i 
#cd $i
#echo `pwd`

#======Make RunAnalyzers.cfg=================#
echo "import FWCore.ParameterSet.Config as cms

process = cms.Process(\"TEST\")
process.load(\"RecoMET.Configuration.CaloTowersOptForMET_cff\")

process.load(\"RecoMET.Configuration.RecoMET_cff\")

process.load(\"RecoMET.Configuration.RecoHTMET_cff\")

process.load(\"RecoMET.Configuration.RecoGenMET_cff\")

process.load(\"RecoMET.Configuration.GenMETParticles_cff\")

process.load(\"RecoJets.Configuration.CaloTowersRec_cff\")

process.load(\"Validation.RecoMET.CaloMET_cff\")

process.load(\"Validation.RecoMET.GenMET_cff\")

process.load(\"Validation.RecoMET.HTMET_cff\")

process.load(\"Validation.RecoMET.GenMETFromGenJets_cff\")

process.load(\"Validation.RecoMET.caloTowers_cff\")

process.load(\"Validation.RecoMET.RecHits_cff\")

process.load(\"Configuration.StandardSequences.Geometry_cff\")

process.load(\"Configuration.StandardSequences.MagneticField_cff\")

process.DQMStore = cms.Service(\"DQMStore\")

process.source = cms.Source(\"PoolSource\",
    debugFlag = cms.untracked.bool(True),
    debugVebosity = cms.untracked.uint32(10),
    fileNames = cms.untracked.vstring(

)


)





process.fileSaver = cms.EDFilter(\"METFileSaver\",
    OutputFile = cms.untracked.string('METTester_data_${i}.root')
) 

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.p = cms.Path(process.fileSaver*process.calotoweroptmaker*process.analyzeRecHits*process.analyzecaloTowers*process.analyzeGenMET*process.analyzeGenMETFromGenJets*process.analyzeHTMET*process.analyzeCaloMET)
process.schedule = cms.Schedule(process.p)

" > ${FullSimRootFileDirectory}/RunAnalyzers-${i}_cfg.py
#============================================#
cd $current_area
done
