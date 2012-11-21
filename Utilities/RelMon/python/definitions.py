#################################################################################
# RelMon: a tool for automatic Release Comparison                              
# https://twiki.cern.ch/twiki/bin/view/CMSPublic/RelMon
#
# $Author: dpiparo $
# $Date: 2012/06/12 12:25:27 $
# $Revision: 1.1 $
#
#                                                                              
# Danilo Piparo CERN - danilo.piparo@cern.ch                                   
#                                                                              
################################################################################

# States:
FAIL=-1
NULL=0
SUCCESS=1
NONE=-2
SKIPED=-3
#-------------------------------------------------------------------------------  
#Error Codes
test_codes={"EMPTY":-101,
            "2D":-102,
            "DIFF_BIN":-103,
            "DIFF_TYPES":-104,
            "NO_HIST":-105,
            "FEW_BINS":-105}
#-------------------------------------------------------------------------------  
relmon_mainpage="https://cms-service-reldqm.web.cern.ch/cms-service-reldqm"

#-------------------------------------------------------------------------------  

server="http://cmsweb.cern.ch" # <--- Check carefully!
base_url="/dqm/relval"

url_encode_dict={"/":"%2F",
                 "+":"%2B",
                 "-":"%2D"}
                 
#-------------------------------------------------------------------------------                   
# Names displayed on the HTML pages  
cat_names={FAIL:"Failing",
            NULL:"Null",
            SUCCESS:"Successful",
            SKIPED: "Skiped"}
# Names used internally
cat_states={FAIL:-1,
            NULL:0,
            SUCCESS:1,
            SKIPED:-3}
# Names used for the different classes of images to obtain the coloured border
cat_classes={FAIL:"fail",
             NULL:"null",
             SUCCESS:"succes",
             SKIPED: "skiped"}     # 1s to avoid conflicts with blueoprint

#-------------------------------------------------------------------------------                                 
# Aggregation of names for the global directory
# These are used to build the condensed summary at the beggining of the page
original=[\
('Level 1 Trigger',['L1T', 'L1TEMU']),

('Pixel Detector',['Pixel', 'OfflinePV', 'Vertexing']),
('Silicon Strips',['SiStrip']),
('Tracking System',['Tracking', 'TrackerDigisV', 'TrackerHitsV', 'TrackerRecHitsV']),

('Ecal Calorimeter',['EcalBarrel', 'EcalEndcap', 'EcalPreshower', 'EcalClusterV', 'EcalDigisV', 'EcalHitsV', 'EcalRecHitsV']),
('Electrons and Photons',['Egamma','EgammaV']),

('Hcal Calorimeter',['Hcal', 'HcalHitsV', 'HcalRecHitsV', 'CaloTowersV']),
('Castor Calorimeter', ['Castor']),
('Jet and Met',['JetMET']),

('Drift Tubes',['DT']),
('Cathode Strip Chambers', ['CSC']),
('Resistive Plate Chambers',['RPC', 'RPCDigisV']),
('Muon Objects',['Muons', 'MuonCSCDigisV', 'MuonDTDigisV' ,'MuonDTHitsV']),

('B Tagging' , ['Btag']),

('Miscellanea: Simulation',['Generator','GlobalDigisV','GlobalHitsV','GlobalRecHitsV','MixingV','NoiseRatesV']),
('Miscellanea',['Info','MessageLogger','ParticleFlow','Physics'])]

# designed for the Reconstruction
reco_aggr=[\
# Detectors
('Tracking System',['TrackerDigisV', 'TrackerHitsV', 'TrackerRecHitsV']+['Pixel']+['SiStrip']),
('Ecal Calorimeter',['EcalBarrel', 'EcalEndcap', 'EcalPreshower', 'EcalClusterV', 'EcalDigisV', 'EcalHitsV', 'EcalRecHitsV']),
('Hcal Calorimeter',['Hcal', 'HcalHitsV', 'HcalRecHitsV', 'CaloTowersV']),
('Drift Tubes',['DT']),
('Cathode Strip Chambers', ['CSC']),
('Resistive Plate Chambers',['RPC', 'RPCDigisV']),

# Actions
('Tracking',['Tracking']),

# Objects
('Electrons',['Egamma/Electrons','EgammaV/ElectronMcFakeValidator','EgammaV/ElectronMcSignalValidator']),
('Photons',['Egamma/PhotonAnalyzer','Egamma/PiZeroAnalyzer','EgammaV/PhotonValidator','EgammaV/ConversionValidator']),

('Muon Objects',['Muons', 'MuonCSCDigisV', 'MuonDTDigisV' ,'MuonDTHitsV']),
('Jet',['JetMET/Jet','JetMET/RecoJetsV','ParticleFlow/PFJetValidation']),
('MET',['JetMET/MET','JetMET/METv','ParticleFlow/PFMETValidation']),
('B Tagging' , ['Btag']),
('Tau' , ['RecoTauV']),

# Other
('Castor Calorimeter', ['Castor']),
('Level 1 Trigger',['L1T', 'L1TEMU']),
('Miscellanea: Sim.',['Generator','GlobalDigisV','GlobalHitsV','GlobalRecHitsV','MixingV','NoiseRatesV']),
('Miscellanea',['Info','MessageLogger','ParticleFlow','Physics'])]

# Designed for the HLT
hlt_aggr=[\
('EGamma',['EgOffline','HLTEgammaValidation']),
('Muon',['Muon'',HLTMonMuon']),
('Tau',['TauRelVal','TauOffline','TauOnline']),
('JetMet',['JetMET','HLTJETMET']),
('Top',['Top']),
('Higgs',['Higgs']),
('HeavyFlavor',['HeavyFlavor']),
('SusyExo',['SusyExo']),
('Alca',['AlCaEcalPi0','EcalPhiSym','HcalIsoTrack']),
('Generic',['FourVector_Val','FourVector'])
]

aggr_pairs_dict={"reco":reco_aggr,"HLT":hlt_aggr}


# This is used to build the table for the twiki, can be different from AGGR_PAIRS in general!
reco_aggr_twiki=[\
# Detectors
('Tk',['TrackerDigisV', 'TrackerHitsV', 'TrackerRecHitsV']+['Pixel']+['SiStrip']),
('Ecal',['EcalBarrel', 'EcalEndcap', 'EcalPreshower', 'EcalClusterV', 'EcalDigisV', 'EcalHitsV', 'EcalRecHitsV']),
('Hcal',['Hcal', 'HcalHitsV', 'HcalRecHitsV', 'CaloTowersV']),
('DT',['DT']),
('CSC', ['CSC']),
('RPC',['RPC', 'RPCDigisV']),

# Actions
('Tracking',['Tracking']),

# Objects
('Electrons',['Egamma/Electrons','EgammaV/ElectronMcFakeValidator','EgammaV/ElectronMcSignalValidator']),
('Photons',['Egamma/PhotonAnalyzer','Egamma/PiZeroAnalyzer','EgammaV/PhotonValidator','EgammaV/ConversionValidator']),

('Muons',['Muons', 'MuonCSCDigisV', 'MuonDTDigisV' ,'MuonDTHitsV']),
('Jet',['JetMET/Jet','JetMET/RecoJetsV','ParticleFlow/PFJetValidation']),
('MET',['JetMET/MET','JetMET/METv','ParticleFlow/PFMETValidation']),
('BTag' , ['Btag']),
('Tau' , ['RecoTauV'])]

# Designed for the HLT
hlt_aggr_twiki=hlt_aggr

aggr_pairs_twiki_dict={'reco':reco_aggr_twiki,'HLT':hlt_aggr_twiki}


#-------------------------------------------------------------------------------
# Blacklisting HLT 
hlt_mc_pattern_blist_pairs=(\
        ("!(TTbar+|H130GGgluon+|PhotonJets+|WE+|ZEE+)", "Egamma@2,EgOffline@2,Ecal@2"),
        ("!(TTbar+|JpsiMM+|SingleMu+)", "Muon@2"),
        ("!(TTbar+)", "Jet@2,JET@2,Met@2,MET@2,Top@2"),
        ("!(TTbar+|ChargedTaus+|QQH135+|ZTT+)", "Tau@2"),
        ("!(TTbar+|H130GGgluon+|ChargedTaus+|QQH135+|Hbb+)", "Higgs@2"),
        ("!(TTbar+|JpsiMM+)", "HeavyFlavor@2"),
        ("!(TTbar+|LM1+|SGrav+|ZPrime+)", "SusyExo@2"))

#Blacklisting RECO
mc_pattern_blist_pairs=(\
        ("WE+|ZEE+|SingleGamma+|SingleElectron+",
            "Btag@1,CSC@1,DT@1,RPC@1,HCAL@1,JetMET@1,CaloTowersV@1,Hcal@1,"+\
            "Muons@1,MuonDTDigisV@1,MuonCSCDigisV@1,MuonDTHitsV@1,"+\
            "RPCDigisV@1,Castor@1,RecoTau@1,ParticleFlow@1,L1T@1"),
        ("WM+|ZMM+|JpsiMM+|SingleMu+",
            "Btag@1,JetMET@1,Castor@1,Egamma@1,Ecal@1,Hcal@1,CaloTowersV@1,RecoTau@1,ParticleFlow@1,L1T@1"),
        ("H130GGgluonfusion+","^[ABCDFHIJKLMNOPQRSTUVWXYZ]@1,Ecal@1"),
        ("RelValMinBias+","AlCaReco@1"),
        ("!(SingleGamma+|SingleElectron+)","Ecal@1"),
        ("!(Mu+|mu+|TTbar+|ZMM+|WM+|MM+)","RPC@1"),
        ("!(SingleMuPt100+|ZMM+)","DT@1"),
        ("!(QCD+|MinBias+|TTbar+)","Hcal@1,CaloTowersV@1,Castor@1"),
        ("!(QCD_Pt_80_120+|TTbar+)","Btag@1"),
        ("!(QCD_Pt_80_120+|MuPt1000+|ZEE+|SingleElectronPt10+|Gamma+|Wjet+|LM1_sfts+)",
            "Track@1,Pixel@1,OfflinePV@1,Vertexing@1,SiStrip@1"),
        ("!(QCD_Pt_80_120+|TTbar+|Wjet+|ChargedTaus+)","RecoTau@1"))
        
mc_pattern_blist_pairs=()

data_pattern_blist_pairs=(\
          ("!(mb20+|run20+)","Ecal@1,Track@1,Pixel@1,OfflinePV@1,Vertexing@1,SiStrip@1"),
          ("!(cos20+|mu20+|wzMu20+)","CSC@1,RPC@1,DT@1"),
          ("!(mb20+|jet20+|run20+)","JetMET@1,Hcal@1,Castor@1,ParticleFlow@1"),
          ("!(wzMu20+|mu20+|run20+)","Muon@1"),
          ("!(wzEG20+|electron20+|photon20+|run20+)","Egamma@1"),
          ("!(mu20+|wzMu20+|jet20+)","Btag@1"))
data_pattern_blist_pairs=()

