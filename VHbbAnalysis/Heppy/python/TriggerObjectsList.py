### Use the format:
# variableName:[triggerObjectProducerInpuTag,filterName,HLTPathName]


triggerObjectCollectionsFull = {
    "hltIsoMu18":[[],"hltL3crIsoL1sMu16L1f0L2f10QL3f18QL3trkIsoFiltered0p09","HLT_IsoMu18_v*"],
    "hltEle23WPLoose":[[],"hltEle23WPLooseGsfTrackIsoFilter","HLT_Ele23_WPLoose_Gsf_v*"],
    "hltEle22eta2p1WPLoose":[[],"hltSingleEle22WPLooseGsfTrackIsoFilter","HLT_Ele22_eta2p1_WPLoose_Gsf_v*"],
}
triggerObjectCollectionsOnlyPt = {
#    "caloMet":[["hltMet","","HLT"],"hltMET90","HLT_PFMET90_PFMHT90_IDTight*"] hltL1extraParticles:Central:HLT
    "l1CentralJets":[["hltL1extraParticles","Central"]],
    "l1ForwardJets":[["hltL1extraParticles","Forward"]],
    "l1Met":[["hltL1extraParticles","MET"]],
    "l1Mht":[["hltL1extraParticles","MHT"]],
    "caloJets":[["hltAK4CaloJetsCorrectedIDPassed"]],
    "pfJets":[["hltAK4PFJetsCorrected"]],

    "caloMet":[["hltMet"]],
    "caloMht":[["hltMht"]],
    "caloMhtNoPU":[["hltMHTNoPU"]],
    "pfMet":[["hltPFMETProducer"]],
    "pfMht":[["hltPFMHTTightID"]],
    "pfHt":[["hltPFHT"]],
}

triggerObjectCollectionsOnlySize = {
    ## VBF triggers
    "hltL1sL1TripleJet927664VBFORL1TripleJet846848VBFORL1HTT100ORL1HTT125ORL1HTT150ORL1HTT175ORL1SingleJet128ORL1DoubleJetC84":[[],"hltL1sL1TripleJet927664VBFORL1TripleJet846848VBFORL1HTT100ORL1HTT125ORL1HTT150ORL1HTT175ORL1SingleJet128ORL1DoubleJetC84",""],
    "hltQuadJet15":[[],"hltQuadJet15",""],
    "hltTripleJet50":[[],"hltTripleJet50",""],
    "hltDoubleJet65":[[],"hltDoubleJet65",""],
    "hltSingleJet80":[[],"hltSingleJet80",""],
    "hltVBFCaloJetEtaSortedMqq150Deta1p5":[[],"hltVBFCaloJetEtaSortedMqq150Deta1p5",""],
    "hltCSVL30p74":[[],"hltCSVL30p74",""],
    "hltPFQuadJetLooseID15":[[],"hltPFQuadJetLooseID15",""],
    "hltPFTripleJetLooseID64":[[],"hltPFTripleJetLooseID64",""],
    "hltPFDoubleJetLooseID76":[[],"hltPFDoubleJetLooseID76",""],
    "hltPFSingleJetLooseID92":[[],"hltPFSingleJetLooseID92",""],

    "hltDoubleCSVPF0p58":[[],"hltDoubleCSVPF0p58",""],
    "hltCSVPF0p78":[[],"hltCSVPF0p78",""],
    "hltVBFPFJetCSVSortedMqq200Detaqq1p2":[[],"hltVBFPFJetCSVSortedMqq200Detaqq1p2",""],

    "hltCSVPF0p78":[[],"hltCSVPF0p78",""],
    "hltVBFPFJetCSVSortedMqq460Detaqq4p1":[[],"hltVBFPFJetCSVSortedMqq460Detaqq4p1",""],

    "hltCSVPF0p78":[[],"hltCSVPF0p78",""],
    "hltVBFPFJetCSVSortedMqq460Detaqq4p1":[[],"hltVBFPFJetCSVSortedMqq460Detaqq4p1",""],

    ## HH->4b triggers

    "hltL1sL1HTT175ORL1QuadJetC60ORL1HTT100ORL1HTT125ORL1HTT150ORL1QuadJetC40":[[],"hltL1sL1HTT175ORL1QuadJetC60ORL1HTT100ORL1HTT125ORL1HTT150ORL1QuadJetC40",""],

    "hltQuadCentralJet45":[[],"hltQuadCentralJet45",""],
    "hltTripleCSV0p67":[[],"hltTripleCSV0p67",""],
    "hltQuadPFCentralJetLooseID45":[[],"hltQuadPFCentralJetLooseID45",""],

    "hltL1sL1TripleJet927664VBFORL1DoubleJetC100ORL1TripleJet846848VBFORL1DoubleJetC84ORL1HTT100ORL1HTT125ORL1HTT150ORL1HTT175":[[],"hltL1sL1TripleJet927664VBFORL1DoubleJetC100ORL1TripleJet846848VBFORL1DoubleJetC84ORL1HTT100ORL1HTT125ORL1HTT150ORL1HTT175",""],

    "hltQuadCentralJet30":[[],"hltQuadCentralJet30",""],
    "hltDoubleCentralJet90":[[],"hltDoubleCentralJet90",""],
    "hltTripleCSV0p67":[[],"hltTripleCSV0p67",""],
    "hltQuadPFCentralJetLooseID30":[[],"hltQuadPFCentralJetLooseID30",""],
    "hltDoublePFCentralJetLooseID90":[[],"hltDoublePFCentralJetLooseID90",""],

    ## ZvvHbb triggers 

    "hltL1sL1ETM70ORETM60ORETM50ORDoubleJetC56ETM60":[[],"hltL1sL1ETM70ORETM60ORETM50ORDoubleJetC56ETM60",""],
    "hltMHTNoPU90":[[],"hltMHTNoPU90",""],
    "hltCSV0p72L3":[[],"hltCSV0p72L3",""],
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
