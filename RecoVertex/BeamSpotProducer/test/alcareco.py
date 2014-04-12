import FWCore.ParameterSet.Config as cms

process = cms.Process("AlcaRecoBeamSpot")

# initialize MessageLogger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

# Beam fitter
process.load("RecoVertex.BeamSpotProducer.d0_phi_analyzer_cff")

process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_1_7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/02F50799-FB7D-DD11-80FC-000423D98DC4.root',
       '/store/relval/CMSSW_2_1_7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/066D059D-FB7D-DD11-A998-000423D98800.root',
       '/store/relval/CMSSW_2_1_7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/08A9C567-FB7D-DD11-8B00-000423DD2F34.root',
       '/store/relval/CMSSW_2_1_7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/0C4D30C3-FB7D-DD11-94C2-000423D98DD4.root',
       '/store/relval/CMSSW_2_1_7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/1697AF63-FB7D-DD11-A122-000423D98AF0.root',
       '/store/relval/CMSSW_2_1_7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/1A6D5F2A-FB7D-DD11-A860-000423D94C68.root',
       '/store/relval/CMSSW_2_1_7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/1C3D4D97-FB7D-DD11-A5AB-000423D9890C.root',
       '/store/relval/CMSSW_2_1_7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/24BBE96A-FB7D-DD11-914F-000423D98834.root',
       '/store/relval/CMSSW_2_1_7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/2E2850F6-FB7D-DD11-AB6C-000423D6AF24.root',
       '/store/relval/CMSSW_2_1_7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/34BD8472-FB7D-DD11-ADAE-000423D6B48C.root',
       '/store/relval/CMSSW_2_1_7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/36B15966-FB7D-DD11-B785-000423D99896.root',
       '/store/relval/CMSSW_2_1_7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/48BE7490-FB7D-DD11-BD97-000423D99AA2.root',
       '/store/relval/CMSSW_2_1_7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/521EC634-FB7D-DD11-872C-000423D99AAE.root',
       '/store/relval/CMSSW_2_1_7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/744F149D-FB7D-DD11-B095-000423D99658.root',
       '/store/relval/CMSSW_2_1_7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/76C46999-FB7D-DD11-B900-000423D98750.root',
       '/store/relval/CMSSW_2_1_7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/848DDA6F-FB7D-DD11-8874-000423D99A8E.root',
       '/store/relval/CMSSW_2_1_7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/8663FD5F-FC7D-DD11-8AAF-000423D98834.root',
       '/store/relval/CMSSW_2_1_7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/8ABB0C69-FB7D-DD11-9AFA-000423D98B6C.root',
       '/store/relval/CMSSW_2_1_7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/8C4C798C-FB7D-DD11-AA5E-000423D8FA38.root',
       '/store/relval/CMSSW_2_1_7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/8ECDBE32-FB7D-DD11-A4DC-000423D986A8.root',
       '/store/relval/CMSSW_2_1_7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/90C2B3C8-FB7D-DD11-B485-000423D98634.root',
       '/store/relval/CMSSW_2_1_7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/94B8975E-FB7D-DD11-BEAF-000423D94C68.root',
       '/store/relval/CMSSW_2_1_7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/9680F65D-FB7D-DD11-8DFB-001617E30D0A.root',
       '/store/relval/CMSSW_2_1_7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/9EA1BDF9-FB7D-DD11-83A2-000423D98AF0.root',
       '/store/relval/CMSSW_2_1_7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/AE2B6AC2-FB7D-DD11-AA76-000423D99CEE.root',
       '/store/relval/CMSSW_2_1_7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/AEA5A698-FB7D-DD11-9B19-000423D98F98.root',
       '/store/relval/CMSSW_2_1_7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/D0CFA0F2-FB7D-DD11-8AC2-000423D94534.root',
       '/store/relval/CMSSW_2_1_7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/D4D2885D-FC7D-DD11-94CD-000423D952C0.root',
       '/store/relval/CMSSW_2_1_7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/E801C5F7-FB7D-DD11-B414-000423D990CC.root',
       '/store/relval/CMSSW_2_1_7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0001/FAD35597-FB7D-DD11-8946-000423D99660.root',
       '/store/relval/CMSSW_2_1_7/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V9_v1/0002/98057D1C-437E-DD11-A450-001617C3B654.root')
                            )

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(1500)
)

## Geometry
##
process.load("Configuration.StandardSequences.Geometry_cff")

## Magnetic Field
##
process.load("Configuration.StandardSequences.MagneticField_cff")

#process.p = cms.Path(process.d0_phi_analyzer)

process.MessageLogger.debugModules = ['BeamSpotAnalyzer']
process.d0_phi_analyzer.OutputFileName = 'EarlyCollision.root'
process.d0_phi_analyzer.BSAnalyzerParameters.TrackCollection = 'TrackRefitter'
#process.d0_phi_analyzer.InputBeamWidth = 

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
#process.GlobalTag.globaltag = 'IDEAL_V9::All'  # take your favourite

process.GlobalTag.connect = "sqlite_file:/afs/cern.ch/user/f/futyand/public/globaltag/10PB_21X.db"
process.GlobalTag.globaltag = '10PB_V1::All'

process.load("RecoTracker.TrackProducer.RefitterWithMaterial_cff")
# process.TrackRefitter.src = 'generalTracks' # should be default

process.p1 = cms.Path(process.TrackRefitter
                      *process.d0_phi_analyzer
                      )
