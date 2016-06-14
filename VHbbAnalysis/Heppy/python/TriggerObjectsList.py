### Use the format:
# variableName:[triggerObjectProducerInpuTag,filterName,HLTPathName]


triggerObjectCollectionsFull = {
    "hltIsoMu20":[[],"hltL3crIsoL1sMu18L1f0L2f10QL3f20QL3trkIsoFiltered0p09","HLT_IsoMu20_v*"],
    "hltEle25WPTight":[[],"hltEle25WPTightGsfTrackIsoFilter","HLT_Ele25_WPTight_Gsf_v*"],
    "hltEle25eta2p1WPLoose":[[],"hltEle25erWPTightGsfTrackIsoFilter","HLT_Ele25_eta2p1_WPLoose_Gsf_v*"],
    
    # triggers for the MSSM analysis
    "hltBTagCaloCSVp026DoubleWithMatching" : [[],"hltBTagCaloCSVp026DoubleWithMatching", "HLT_DoubleJetsC100_DoubleBTagCSV_p026_DoublePFJetsC160_v*"],
    "hltBTagCaloCSVp014DoubleWithMatching" : [[], "hltBTagCaloCSVp014DoubleWithMatching", "HLT_DoubleJetsC100_DoubleBTagCSV_p014_DoublePFJetsC100MaxDeta1p6_v*"],
    "hltDoublePFJetsC100" : [[], "hltDoublePFJetsC100", ""],
}
triggerObjectCollectionsOnlyPt = {
#    "caloMet":[["hltMet","","HLT"],"hltMET90","HLT_PFMET90_PFMHT90_IDTight*"] hltL1extraParticles:Central:HLT

#   to be checked with L1 stage-2 objects
#    "l1CentralJets":[["l1extraParticles","Central"]],
#    "l1ForwardJets":[["l1extraParticles","Forward"]],
#    "l1Met":[["l1extraParticles","MET"]],
#    "l1Mht":[["l1extraParticles","MHT"]],

    "caloJets":[["hltAK4CaloJetsCorrectedIDPassed"]],
    "pfJets":[["hltAK4PFJetsCorrected"]],
#    "pfJetsIDTight":[["hltAK4PFJetsTightIDCorrected"]],

    "caloMet":[["hltMet"]],
    "caloMht":[["hltMht"]],
    "caloMhtNoPU":[["hltMHTNoPU"]],
    "pfMet":[["hltPFMETProducer"]],
    "pfMht":[["hltPFMHTTightID"]],
    "pfHt":[["hltPFHT"]],
}

triggerObjectCollectionsOnlySize = {
    ## VBF triggers
    "hltL1sTripleJetVBFIorHTTIorDoubleJetCIorSingleJet":[[],"hltL1sTripleJetVBFIorHTTIorDoubleJetCIorSingleJet",""],
    "hltQuadJet15":[[],"hltQuadJet15",""],
    "hltTripleJet50":[[],"hltTripleJet50",""],
    "hltDoubleJet65":[[],"hltDoubleJet65",""],
    "hltSingleJet80":[[],"hltSingleJet80",""],
    "hltVBFCaloJetEtaSortedMqq150Deta1p5":[[],"hltVBFCaloJetEtaSortedMqq150Deta1p5",""],
    "hltBTagCaloCSVp022Single":[[],"hltBTagCaloCSVp022Single",""],
    "hltPFQuadJetLooseID15":[[],"hltPFQuadJetLooseID15",""],
    "hltPFTripleJetLooseID64":[[],"hltPFTripleJetLooseID64",""],
    "hltPFDoubleJetLooseID76":[[],"hltPFDoubleJetLooseID76",""],
    "hltPFSingleJetLooseID92":[[],"hltPFSingleJetLooseID92",""],

    "hltBTagPFCSVp11DoubleWithMatching":[[],"hltBTagPFCSVp11DoubleWithMatching",""],
    "hltBTagPFCSVp016SingleWithMatching":[[],"hltBTagPFCSVp016SingleWithMatching",""],
    "hltVBFPFJetCSVSortedMqq200Detaqq1p2":[[],"hltVBFPFJetCSVSortedMqq200Detaqq1p2",""],
    "hltVBFPFJetCSVSortedMqq460Detaqq4p1":[[],"hltVBFPFJetCSVSortedMqq460Detaqq4p1",""],

    ## HH->4b triggers

    "hltL1sQuadJetCIorTripleJetVBFIorHTT":[[],"hltL1sQuadJetCIorTripleJetVBFIorHTT",""],

    "hltQuadCentralJet45":[[],"hltQuadCentralJet45",""],
    "hltBTagCaloCSVp087Triple":[[],"hltBTagCaloCSVp087Triple",""],
    "hltQuadPFCentralJetLooseID45":[[],"hltQuadPFCentralJetLooseID45",""],

    "hltL1sTripleJetVBFIorHTTIorDoubleJetCIorSingleJet":[[],"hltL1sTripleJetVBFIorHTTIorDoubleJetCIorSingleJet",""],

    "hltQuadCentralJet30":[[],"hltQuadCentralJet30",""],
    "hltDoubleCentralJet90":[[],"hltDoubleCentralJet90",""],
    "hltBTagCaloCSVp087Triple":[[],"hltBTagCaloCSVp087Triple",""],
    "hltQuadPFCentralJetLooseID30":[[],"hltQuadPFCentralJetLooseID30",""],
    "hltDoublePFCentralJetLooseID90":[[],"hltDoublePFCentralJetLooseID90",""],

    ## ZvvHbb triggers 

    "hltL1sETM50ToETM100IorETM60Jet60dPhiMin0p4IorDoubleJetC60ETM60":[[],"hltL1sETM50ToETM100IorETM60Jet60dPhiMin0p4IorDoubleJetC60ETM60",""],
    "hltMHTNoPU90":[[],"hltMHTNoPU90",""],
    "hltBTagCaloCSVp067Single":[[],"hltBTagCaloCSVp067Single",""],
    "hltPFMHTTightID90":[[],"hltPFMHTTightID90",""],

    "hltMET70":[[],"hltMET70",""],
    "hltMHT70":[[],"hltMHT70",""],
    "hltPFMHTTightID90":[[],"hltPFMHTTightID90",""],
    "hltPFMET90":[[],"hltPFMET90",""],

}

triggerObjectCollections = dict(
    triggerObjectCollectionsOnlySize.items() + 
    triggerObjectCollectionsOnlyPt.items() + 
    triggerObjectCollectionsFull.items()
    )
