import FWCore.ParameterSet.Config as cms

def add_hlt_validation(process,hltProcessName=None,sampleLabel=""): 
    if hltProcessName==None:
        hltProcessName = process.name_()

    ptBins=cms.vdouble(0, 10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85,90,95 , 100,105,110,115,120,125, 130, 135 ,140, 145, 150)
    ptBinsHighPt = cms.vdouble([i*100+0 for i in range(0,40)])
    ptBinsHT=cms.vdouble(0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 1000, 1050, 1100, 1150, 1200, 1300)
    ptBinsJet=cms.vdouble(0, 100, 200, 300, 350, 375, 400, 425, 450, 475, 500, 550, 600, 700, 800, 900, 1000) 
    ptBinsMET=cms.vdouble([i*20+100 for i in range(20)])
    etaBins=cms.vdouble(-4,-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 4)
    phiBins=cms.vdouble([-3.2+i*0.2 for i in range(0,32+1)])
    etaCut=cms.PSet(
        rangeVar=cms.string("eta"),
        allowedRanges=cms.vstring("-2.4:2.4")
    )
    eta2p1Cut=cms.PSet(
        rangeVar=cms.string("eta"),
        allowedRanges=cms.vstring("-2.1:2.1")
    )
    eleEtaCut = cms.PSet(
     rangeVar=cms.string("eta"),
        allowedRanges=cms.vstring("-1.4442:1.4442","1.566:2.5","-2.5:-1.566")
    )   

    jetEtaCut=cms.PSet(
        rangeVar=cms.string("eta"),
        allowedRanges=cms.vstring("-5:5")
    )
    ptCut=cms.PSet(
        rangeVar=cms.string("pt"),
        allowedRanges=cms.vstring("40:9999")
    )
    jetPtCut=cms.PSet(
        rangeVar=cms.string("pt"),
        allowedRanges=cms.vstring("200:9999")
    )
    tauPtCut=cms.PSet(
        rangeVar=cms.string("pt"),
        allowedRanges=cms.vstring("50:9999")
    )


    process.HLTGenValSourceHT = cms.EDProducer('HLTGenValSource',
        # these are the only one the user needs to specify
        objType = cms.string("AK4HT"),
        hltProcessName = cms.string(hltProcessName),
        trigEvent = cms.InputTag(f"hltTriggerSummaryAOD::{hltProcessName}"),
        sampleLabel = cms.string(sampleLabel),
        hltPathsToCheck = cms.vstring(
        "HLT_PFHT1050_v",
        ),
        doOnlyLastFilter = cms.bool(False),
        histConfigs = cms.VPSet(
            cms.PSet(
                vsVar = cms.string("pt"),
                binLowEdges = ptBinsHT,
            ),
        ),
    )

    process.HLTGenValSourceMU = cms.EDProducer('HLTGenValSource',
        # these are the only one the user needs to specify
        objType = cms.string("mu"),
        hltProcessName = cms.string(hltProcessName),
        sampleLabel = cms.string(sampleLabel),
        trigEvent = cms.InputTag(f"hltTriggerSummaryAOD::{hltProcessName}"),
        hltPathsToCheck = cms.vstring(
        "HLT_Mu50_v:absEtaCut=1.2,tag=centralbarrel",
        "HLT_Mu50_v:bins=ptBinsHighPt,tag=highpt_bins",
        "HLT_IsoMu24_v"
        ),
        doOnlyLastFilter = cms.bool(False), 
        binnings = cms.VPSet(
            cms.PSet(
                name = cms.string("ptBinsHighPt"),
                vsVar = cms.string("pt"),
                binLowEdges = ptBinsHighPt
            )
        ),
        histConfigs = cms.VPSet(
            cms.PSet(
                vsVar = cms.string("pt"),
                binLowEdges = ptBins,
                rangeCuts = cms.VPSet(etaCut)
            ),
            cms.PSet(
                vsVar = cms.string("eta"),
                binLowEdges = etaBins,
            ),
        ),
    )
    process.HLTGenValSourceTAU = cms.EDProducer('HLTGenValSource',
#     these are the only one the user needs to specify
        objType = cms.string("tauHAD"),
        hltProcessName = cms.string(hltProcessName),
        trigEvent = cms.InputTag(f"hltTriggerSummaryAOD::{hltProcessName}"),
        sampleLabel = cms.string(sampleLabel),
        hltPathsToCheck = cms.vstring(),
        doOnlyLastFilter = cms.bool(False),
        histConfigs = cms.VPSet(
            cms.PSet(
                vsVar = cms.string("pt"),
                binLowEdges = ptBins,                
                rangeCuts = cms.VPSet(eta2p1Cut)
            ),
            cms.PSet(
                vsVar = cms.string("eta"),
                binLowEdges = etaBins,
                rangeCuts = cms.VPSet(tauPtCut)
            ),
        ),
    )

    process.HLTGenValSourceELE = cms.EDProducer('HLTGenValSource',
        # these are the only one the user needs to specify
        objType = cms.string("ele"),
        hltProcessName = cms.string(hltProcessName), 
        sampleLabel = cms.string(sampleLabel),
        trigEvent = cms.InputTag(f"hltTriggerSummaryAOD::{hltProcessName}"),
        hltPathsToCheck = cms.vstring(
        "HLT_Ele35_WPTight_Gsf_v",
        "HLT_Ele35_WPTight_Gsf_v:region=EB,tag=barrel",
        "HLT_Ele35_WPTight_Gsf_v:region=EE,tag=endcap",
        "HLT_Ele35_WPTight_Gsf_v:bins=ptBins,region=EB,tag=lowpt_barrel",
        "HLT_Ele35_WPTight_Gsf_v:bins=ptBins,region=EE,tag=lowpt_endcap",
        "HLT_Ele115_CaloIdVT_GsfTrkIdT_v:region=EB,ptMin=140,tag=barrel",
        "HLT_Ele115_CaloIdVT_GsfTrkIdT_v:region=EE,ptMin=140,tag=endcap",
        "HLT_Photon200_v:region=EB,ptMin=240,tag=barrel",
        "HLT_Photon200_v:region=EE,ptMin=240,tag=endcap"
        ),
        binnings = cms.VPSet(
            cms.PSet(
                name = cms.string("ptBins"),
                vsVar = cms.string("pt"),
                binLowEdges = ptBins
            )
        ),
        doOnlyLastFilter = cms.bool(False),
        histConfigs = cms.VPSet(
            cms.PSet(
                vsVar = cms.string("pt"),
                binLowEdges = ptBinsHighPt,
                rangeCuts = cms.VPSet(eleEtaCut)
            ),
            cms.PSet(
                vsVar = cms.string("eta"),
                binLowEdges = etaBins,
                rangeCuts = cms.VPSet(ptCut)
            ),
        ),
        histConfigs2D = cms.VPSet(
            cms.PSet(
                vsVarX = cms.string("eta"),
                vsVarY = cms.string("phi"),
                binLowEdgesX = etaBins,
                binLowEdgesY = phiBins
            )
        )
    )

    process.HLTGenValSourceAK4 = cms.EDProducer('HLTGenValSource',
        # these are the only one the user needs to specify
        objType = cms.string("AK4jet"),
        hltProcessName = cms.string(hltProcessName),
        trigEvent = cms.InputTag(f"hltTriggerSummaryAOD::{hltProcessName}"),
        sampleLabel = cms.string(sampleLabel),
        hltPathsToCheck = cms.vstring(
        "HLT_PFJet500",
        ),
        doOnlyLastFilter = cms.bool(False),
        histConfigs = cms.VPSet(
            cms.PSet(
                vsVar = cms.string("pt"),
                binLowEdges = ptBinsJet,
                rangeCuts = cms.VPSet(jetEtaCut)
            ),
            cms.PSet(
                vsVar = cms.string("eta"),
                binLowEdges = etaBins,
                rangeCuts = cms.VPSet(jetPtCut)
            ),
        ),
    )

    process.HLTGenValSourceAK8 = cms.EDProducer('HLTGenValSource',
        # these are the only one the user needs to specify
        objType = cms.string("AK8jet"),
        hltProcessName = cms.string(hltProcessName),
        trigEvent = cms.InputTag(f"hltTriggerSummaryAOD::{hltProcessName}"),
        sampleLabel = cms.string(sampleLabel),
        hltPathsToCheck = cms.vstring(
        "HLT_AK8PFJet500",
        "HLT_AK8PFJet400_TrimMass30:minMass=50",
        ),
        doOnlyLastFilter = cms.bool(False),
        histConfigs = cms.VPSet(
            cms.PSet(
                vsVar = cms.string("pt"),
                binLowEdges = ptBinsJet,
                rangeCuts = cms.VPSet(jetEtaCut)
            ),
            cms.PSet(
                vsVar = cms.string("eta"),
                binLowEdges = etaBins, 
                rangeCuts = cms.VPSet(jetPtCut)
            ),
        ),
    )

    process.HLTGenValSourceMET = cms.EDProducer('HLTGenValSource',
        # these are the only one the user needs to specify
        objType = cms.string("MET"),
        hltProcessName = cms.string(hltProcessName),
        trigEvent = cms.InputTag(f"hltTriggerSummaryAOD::{hltProcessName}"),
        sampleLabel = cms.string(sampleLabel),
        hltPathsToCheck = cms.vstring(
        "HLT_PFMET120_PFMHT120_IDTight", 
        "HLT_PFMET200_NotCleaned_v11",
        ),
        doOnlyLastFilter = cms.bool(False),
        histConfigs = cms.VPSet(
            cms.PSet(
                vsVar = cms.string("pt"),
                binLowEdges = ptBinsMET
            ),
        ),
    )



    process.HLTValidationPath = cms.Path(
            process.HLTGenValSourceMU *
            process.HLTGenValSourceELE *
            process.HLTGenValSourceTAU *
            process.HLTGenValSourceHT *
            process.HLTGenValSourceAK4 *
            process.HLTGenValSourceAK8 *
            process.HLTGenValSourceMET
            )

    process.load("RecoMET.Configuration.RecoGenMET_cff")
    process.load("RecoMET.Configuration.GenMETParticles_cff")
    process.load("PhysicsTools.JetMCAlgos.TauGenJets_cfi")
    process.HLTValidationPath.associate(process.recoGenMETTask)
    process.HLTValidationPath.associate(process.genMETParticlesTask)
    process.HLTValidationTauGenJetsTask = cms.Task(process.tauGenJets)
    process.HLTValidationPath.associate(process.HLTValidationTauGenJetsTask)

    if process.schedule is not None:
        process.schedule.append(process.HLTValidationPath)
    elif hasattr(process,"HLTSchedule"):
        process.HLTSchedule.append(process.HLTValidationPath)
    return process


def add_hlt_validation_phaseII(process,hltProcessName=None,sampleLabel=""):
    if hltProcessName==None:
        hltProcessName = process.name_()
    process = add_hlt_validation(process,hltProcessName,sampleLabel)
    process.HLTGenValSourceELE.hltPathsToCheck = cms.vstring(
        "HLT_Ele115_NonIso_L1Seeded",
        "HLT_Ele26_WP70_L1Seeded",
        "HLT_Ele26_WP70_L1Seeded:region=EB,tag=barrel",
        "HLT_Ele26_WP70_L1Seeded:region=EE,tag=endcap",
        "HLT_Ele26_WP70_L1Seeded:bins=ptBins,region=EB,tag=lowpt_barrel",
        "HLT_Ele26_WP70_L1Seeded:bins=ptBins,region=EE,tag=lowpt_endcap",
        "HLT_Ele32_WPTight_L1Seeded",
        "HLT_Ele32_WPTight_L1Seeded:region=EB,tag=barrel",
        "HLT_Ele32_WPTight_L1Seeded:region=EE,tag=endcap",
        "HLT_Ele32_WPTight_L1Seeded:bins=ptBins,region=EB,tag=lowpt_barrel",
        "HLT_Ele32_WPTight_L1Seeded:bins=ptBins,region=EE,tag=lowpt_endcap",        
        "HLT_Photon108EB_TightID_TightIso_L1Seeded:region=EB,ptMin=120,tag=barrel",
        "HLT_Photon187_L1Seeded:region=EB,ptMin=200,tag=barrel",
        "HLT_Photon187_L1Seeded:region=EE,ptMin=200,tag=endcap"
    )
    process.HLTGenValSourceMU.hltPathsToCheck = cms.vstring(
        "HLT_IsoMu24_FromL1TkMuon:ptMin=30",
        "HLT_Mu50_FromL1TkMuon:ptMin=60,absEtaCut=1.2,tag=centralbarrel",        
        "HLT_Mu50_FromL1TkMuon:ptMin=60,bins=ptBinsHighPt,tag=highpt_bins",        
    )
    process.HLTGenValSourceAK4.hltPathsToCheck = cms.vstring(
        "HLT_AK4PFPuppiJet520"
    )
    process.HLTGenValSourceTAU.hltPathsToCheck = cms.vstring(
        "HLT_DoubleMediumChargedIsoPFTauHPS40_eta2p1",
        "HLT_DoubleMediumDeepTauPFTauHPS35_eta2p1"
    )
    process.HLTGenValSourceHT.hltPathsToCheck = cms.vstring(
        "HLT_PFPuppiHT1070"
    )
    process.HLTGenValSourceMET.hltPathsToCheck = cms.vstring(
        "HLT_PFPuppiMETTypeOne140_PFPuppiMHT140"
    )
    #this should go into the other function
    ptBinsRes = cms.vdouble(0,5,10,15,20,25,30,40,50,75,100,150,400,800,1500,10000)
    resBins = cms.vdouble([0+0.01*i for i in range(0,151)])

    etaCut = cms.PSet(
        rangeVar=cms.string("eta"),
        allowedRanges=cms.vstring("-2.4:2.4")
    )
    eleEtaCut = cms.PSet(
        rangeVar=cms.string("eta"),
        allowedRanges=cms.vstring("-1.4442:1.4442","1.566:2.5","-2.5:-1.566")
    )
    muEtaCut = cms.PSet(
        rangeVar=cms.string("eta"),
        allowedRanges=cms.vstring("-2.4:2.4")
    )
    process.HLTGenResSource = cms.EDProducer('HLTGenResSource',
        hltProcessName = cms.string(hltProcessName),
        resCollConfigs = cms.VPSet(
            cms.PSet(
                objType = cms.string("ele"),
                collectionName = cms.string("hltEgammaCandidatesL1Seeded"),
                histConfigs = cms.VPSet(
                    cms.PSet(
                        resVar = cms.string("ptRes"),
                        vsVar = cms.string("pt"),
                        resBinLowEdges = resBins,
                        vsBinLowEdges = ptBinsRes,
                        rangeCuts = cms.VPSet(eleEtaCut)
                    ),
                )),
            cms.PSet(
                objType = cms.string("pho"),
                collectionName = cms.string("hltEgammaCandidatesL1Seeded"),
                histConfigs = cms.VPSet(
                    cms.PSet(
                        resVar = cms.string("ptRes"),
                        vsVar = cms.string("pt"),
                        resBinLowEdges = resBins,
                        vsBinLowEdges = ptBinsRes,
                        rangeCuts = cms.VPSet(eleEtaCut)
                    ),
                )
            ),
            cms.PSet(
                objType = cms.string("mu"),
                collectionName = cms.string("hltPhase2L3MuonCandidates"),
                histConfigs = cms.VPSet( 
                    cms.PSet(
                        resVar = cms.string("ptRes"),
                        vsVar = cms.string("pt"),
                        resBinLowEdges = resBins,
                        vsBinLowEdges = ptBinsRes,
                        rangeCuts = cms.VPSet(muEtaCut)
                    ),
                ),
            ),
            cms.PSet(
                objType = cms.string("tau"),
                collectionName = cms.string("hltHpsPFTauProducer"),
                histConfigs = cms.VPSet(
                    cms.PSet(
                        resVar = cms.string("ptRes"),
                        vsVar = cms.string("pt"),
                        resBinLowEdges = resBins,
                        vsBinLowEdges = ptBinsRes,
                        rangeCuts = cms.VPSet(etaCut)
                    )
                )
            ),
            cms.PSet(
                objType = cms.string("AK4jet"),
                collectionName = cms.string("hltAK4PFPuppiJetsCorrected"),
                histConfigs = cms.VPSet(
                    cms.PSet(
                        resVar = cms.string("ptRes"),
                        vsVar = cms.string("pt"),
                        resBinLowEdges = resBins,
                        vsBinLowEdges = ptBinsRes,
                        rangeCuts = cms.VPSet(etaCut)
                    )
                )
            ),
        ),
    )
    process.HLTValidationPath.insert(0,process.HLTGenResSource)
    return process
