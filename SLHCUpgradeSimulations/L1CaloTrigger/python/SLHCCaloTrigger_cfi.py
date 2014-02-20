import FWCore.ParameterSet.Config as cms


L1CaloTriggerSetupSource = cms.ESSource("EmptyESSource",
                                        recordName = cms.string('L1CaloTriggerSetupRcd'),
                                        firstValid = cms.vuint32(1),
                                        iovIsRunNotTime = cms.bool(True)
                                        )

L1CaloTriggerSetup = cms.ESProducer("L1CaloTriggerSetupProducer",
                                    InputXMLFile = cms.FileInPath('SLHCUpgradeSimulations/L1CaloTrigger/data/setup.xml')
                                    )


#----------------------------------------------------------------------------------------------------
# Test Pattern Producer - Preliminary
#----------------------------------------------------------------------------------------------------
#L1TestPatternCaloTowerProducer = cms.EDProducer("L1TestPatternCaloTowerProducer",
#    TestPatternFile  = cms.FileInPath('SLHCUpgradeSimulations/L1CaloTrigger/data/testPattern.txt'),
#)
#----------------------------------------------------------------------------------------------------


#UNCOMMENT HERE TO RUN ON DATA - IO
L1CaloTowerProducer = cms.EDProducer("L1CaloTowerProducer",
    ECALDigis      = cms.InputTag("ecalDigis:EcalTriggerPrimitives"),
    HCALDigis      = cms.InputTag("hcalDigis"),
    UseUpgradeHCAL = cms.bool(False),

)

#COMMENT OUT FOLLOWING LINES TO RUN ON DATA  (Data is more useful than simulations anyways)- IO
#L1CaloTowerProducer = cms.EDProducer("L1CaloTowerProducer",
#    ECALDigis = cms.InputTag("simEcalTriggerPrimitiveDigis"),
#    HCALDigis = cms.InputTag("simHcalTriggerPrimitiveDigis"),
#    UseUpgradeHCAL = cms.bool(False) #added to allow use of Upgrade HCAL - AWR 12/05/2011
#)

L1RingSubtractionProducer = cms.EDProducer("L1RingSubtractionProducer",
    src = cms.InputTag("L1CaloTowerProducer"),
	RingSubtractionType = cms.string("median") # "mean", "median" or "constant"
)


L1CaloRegionProducer = cms.EDProducer("L1CaloRegionProducer",
    src = cms.InputTag("L1CaloTowerProducer")
)                                      

L1CaloClusterProducer = cms.EDProducer("L1CaloClusterProducer",
    src = cms.InputTag("L1CaloTowerProducer")
)

#currently taus and e/gamma have diverged, this is for taus
L1CaloClusterFilter = cms.EDProducer("L1CaloClusterFilter",
    src = cms.InputTag("L1CaloClusterProducer"),
    etMode = cms.int32(3)
)

#currently taus and e/gamma have diverged, this is for taus
L1CaloClusterIsolator = cms.EDProducer("L1CaloClusterIsolator",
    src = cms.InputTag("L1CaloClusterFilter")
)


#currently taus and e/gamma have diverged, this is for eles
L1CaloClusterEGFilter = cms.EDProducer("L1CaloClusterFilter",
    src = cms.InputTag("L1CaloClusterProducer"),
    etMode = cms.int32(1)
)

#currently taus and e/gamma have diverged, this is for eles
L1CaloClusterEGIsolator = cms.EDProducer("L1CaloClusterEGIsolator",
                                         caloClustersTag = cms.InputTag("L1CaloClusterFilter"),
                                         caloTowersTag = cms.InputTag("L1CaloTowerProducer"),
                                         rhoTag = cms.InputTag("L1RhoProducer"),
                                         maxTowerIEta=cms.int32(27), #HF is 29 and up, currently excluding tower 28 as well
                                         isolEtCut=cms.int32(-1) # isolEt<=X (goes negative due to PU subtraction)
)

## New e/g clustering

# Find seeds and build 3x3 clusters
L1CaloProtoClusterProducer = cms.EDProducer("L1CaloProtoClusterProducer",
    src = cms.InputTag("L1CaloTowerProducer")
)

# Keep only local maxima
L1CaloProtoClusterFilter = cms.EDProducer("L1CaloProtoClusterFilter",
    src = cms.InputTag("L1CaloProtoClusterProducer")
)

# Share towers for overlapping clusters
# The e/g identification bit is computed here
L1CaloProtoClusterSharing = cms.EDProducer("L1CaloProtoClusterSharing",
    src = cms.InputTag("L1CaloProtoClusterFilter")
)

# Trim the 3x3 cluster for e/g clusters
# Extend the e/g clusters in the phi direction
# The cluster position is computed here
L1CaloEgammaClusterProducer = cms.EDProducer("L1CaloEgammaClusterProducer",
    src = cms.InputTag("L1CaloProtoClusterSharing")
)

# Extend the e/g clusters in the phi direction
# The cluster position is computed here
#L1CaloExtendedEgammaClusterProducer = cms.EDProducer("L1CaloExtendedEgammaClusterProducer",
#    src = cms.InputTag("L1CaloEgammaClusterProducer")
#)

# Isolation for e/g clusters
L1CaloEgammaClusterIsolator = cms.EDProducer("L1CaloClusterWithSeedEGIsolator",
                                         caloClustersTag = cms.InputTag("L1CaloEgammaClusterProducer"),
                                         caloTowersTag = cms.InputTag("L1CaloTowerProducer"),
                                         rhoTag = cms.InputTag("L1RhoProducer"),
                                         maxTowerIEta=cms.int32(27), #HF is 29 and up, currently excluding tower 28 as well
                                         isolEtCut=cms.int32(-1) # isolEt<=X (goes negative due to PU subtraction)
)
## End new e/g clustering




L1CaloJetProducer = cms.EDProducer("L1CaloJetProducer",
    src = cms.InputTag("L1CaloRegionProducer")
)

L1CaloJetFilter = cms.EDProducer("L1CaloJetFilter",
    src = cms.InputTag("L1CaloJetProducer")
)

L1CaloJetExpander = cms.EDProducer("L1CaloJetExpander",
    src = cms.InputTag("L1CaloJetFilter")
)

L1TowerJetProducer = cms.EDProducer("L1TowerJetProducer",
    src = cms.InputTag("L1CaloTowerProducer"),
	JetDiameter = cms.uint32(8),
	JetShape = cms.string("circle") # "circle" or "square"
)

     
L1TowerJetFilter1D = cms.EDProducer("L1TowerJetFilter1D",
    src = cms.InputTag("L1TowerJetProducer"),
	ComparisonDirection = cms.string("eta"), # "eta" or "phi"
	NumOfOutputJets = cms.uint32(4)
)

L1TowerJetFilter2D = cms.EDProducer("L1TowerJetFilter2D",
    src = cms.InputTag("L1TowerJetFilter1D"),
	ComparisonDirection = cms.string("phi"), # "eta" or "phi"
	NumOfOutputJets = cms.uint32(12)
)
L1TowerJetPUEstimator = cms.EDProducer("L1TowerJetPUEstimator",
    inRhodata_file = cms.FileInPath('SLHCUpgradeSimulations/L1CaloTrigger/data/rho_lookup.txt'),
    FilteredCircle8 = cms.InputTag("L1TowerJetFilter2D"),
)

L1TowerJetPUSubtractedProducer =  cms.EDProducer("L1TowerJetPUSubtractedProducer",
    FilteredCircle8 = cms.InputTag("L1TowerJetFilter2D"),
    CalibratedL1Rho = cms.InputTag("L1TowerJetPUEstimator", "Rho"),
)

L1CalibFilterTowerJetProducer = cms.EDProducer("L1CalibFilterTowerJetProducer",
    inMVA_weights_file = cms.FileInPath('SLHCUpgradeSimulations/L1CaloTrigger/data/TMVARegression_BDT.weights.xml'),
    PUSubtractedCentralJets = cms.InputTag("L1TowerJetPUSubtractedProducer","PUSubCenJets"),
)





L1TowerFwdJetProducer = cms.EDProducer("L1TowerFwdJetProducer",
    src = cms.InputTag("L1CaloTowerProducer"),
	JetDiameter = cms.uint32(8),
	JetShape = cms.string("circle") # "circle" or "square"
)    
   
L1TowerFwdJetFilter1D = cms.EDProducer("L1TowerJetFilter1D",
    src = cms.InputTag("L1TowerFwdJetProducer"),
	ComparisonDirection = cms.string("eta"), # "eta" or "phi"
	NumOfOutputJets = cms.uint32(4)
)

L1TowerFwdJetFilter2D = cms.EDProducer("L1TowerJetFilter2D",
    src = cms.InputTag("L1TowerFwdJetFilter1D"),
	ComparisonDirection = cms.string("phi"), # "eta" or "phi"
	NumOfOutputJets = cms.uint32(12)
)




rawSLHCL1ExtraParticles = cms.EDProducer("L1ExtraTranslator",
                                  EGClusters   = cms.InputTag("L1CaloClusterEGIsolator"),
                                  TauClusters   = cms.InputTag("L1CaloClusterIsolator"),
                                  Jets       = cms.InputTag("L1CaloJetExpander"),
                                  Towers     = cms.InputTag("L1CaloTowerProducer"),                                         
                                  NParticles = cms.uint32(999),
                                  NJets      = cms.uint32(999),
                                  maxJetTowerEta = cms.double(3.0)
                              
)

# New e/g clustering
# Translation, only for e/g for the moment
rawSLHCL1ExtraParticlesNewClustering = cms.EDProducer("L1NewEgammaExtraTranslator",
                                  Clusters = cms.InputTag("L1CaloEgammaClusterIsolator"),
                                  NParticles = cms.uint32(999)
)
# End new e/g clustering


SLHCL1ExtraParticles = cms.EDProducer("L1ExtraCalibrator",
                                      eGamma = cms.InputTag("rawSLHCL1ExtraParticles","EGamma"),
                                      isoEGamma = cms.InputTag("rawSLHCL1ExtraParticles","IsoEGamma"),
                                      taus = cms.InputTag("rawSLHCL1ExtraParticles","Taus"),
                                      isoTaus = cms.InputTag("rawSLHCL1ExtraParticles","IsoTaus"),
                                      jets = cms.InputTag("rawSLHCL1ExtraParticles","Jets"),

                                      ##How to calibrate  
                                      ##Scale factor = MC/RAW = a+b |eta| +c|eta|^2 
                                      ##Give the coeffs for egamma and taus 
                                      ##So you need to fit the eta correction with a  
                                      ##parabola and add the coeffs here
                                      ## Same as we do for RCT Calibration
                                      ## 6.Aug.2010 -- Barrel/endcap calibration separated
                                      ## L1ExtraCalibrator.cc currently ignores the second coefficient, to provide symmetric 

                                      eGammaCoefficientsB = cms.vdouble(1.112,-0.02623,0.08898),
                                      tauCoefficientsB    = cms.vdouble(1.175,0.1656,0.03392),
                                      eGammaCoefficientsE = cms.vdouble(1.506,0.2938,-0.1501),
                                      tauCoefficientsE    = cms.vdouble(1.175,0.1656,0.03392),

                                      #eGammaCoefficientsB = cms.vdouble(1.0,0.0,0.0),
                                      #tauCoefficientsB    = cms.vdouble(1.0,0.0,0.0),
                                      #eGammaCoefficientsE = cms.vdouble(1.0,0.0,0.0),
                                      #tauCoefficientsE    = cms.vdouble(1.0,0.0,0.0) 

                                      ## 18.Aug.2010 -- Added bin-by-bin correction factors to further improve upon functional form.
                                      ## These correction factors are pulled from the MC/Reco( |eta| ) plots after the above calibrations
                                      ## were applied (ZEE and HTT)

                                      #Calibration good for 5GeV threshold
                                      #eGammaBinCorr = cms.vdouble( 0.995280 , 0.999074 , 1.014316 , 1.005148 , 0.992236 , 0.998246 , 1.017787 , 1.137927 , 1.105248 , 0.948162 , 0.896218 , 0.774255 , 0.890730),
                                      #tauBinCorr = cms.vdouble( 1.065728, 1.054150, 1.090152, 1.102329, 1.105562, 1.143280, 1.149978, 1.187399, 1.156080, 1.095267, 1.078722, 0.994069, 1.034597)
                                      
                                      #Calibration good for 30GeV threshold
                                      eGammaBinCorr = cms.vdouble( 0.980478 , 0.977497 , 0.991688 , 0.974088 , 0.957387 , 0.958428 , 0.946692 , 1.010764 , 1.000194 , 0.876818 , 0.859533 , 0.766770 , 0.852342),
                                      tauBinCorr=cms.vdouble( 0.966896, 0.968547, 0.981441, 0.993066, 1.003556, 1.036209, 1.043883, 1.089409, 1.051449, 0.968977, 0.962960, 0.875468, 0.944211),
                                      
                                      #eGammaBinCorr = cms.vdouble( 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0),
                                      #tauBinCorr = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

                                      ## June 2013
                                      ## New calibration. Corrections computed in bins of |eta| + linear interpolation
                                      ## Only for e/g
                                      applyNewCalib = cms.bool(True), # switch between old and new calibration
                                      eGammaEtaPoints = cms.vdouble(0.125, 0.375, 0.625, 0.875, 1.125, 1.3645, 1.6145, 1.875, 2.125, 2.375),
                                      eGammaNewCorr = cms.vdouble(0.0952467, 0.101389, 0.10598, 0.12605, 0.162749, 0.193123, 0.249227, 0.2800289, 0.271548, 0.27855), 
                                      )



l1extraParticlesCalibrated = cms.EDProducer("L1ExtraCalibrator",
                                            eGamma = cms.InputTag("l1extraParticles","NonIsolated"),
                                            isoEGamma = cms.InputTag("l1extraParticles","Isolated"),
                                            taus = cms.InputTag("l1extraParticles","Tau"),
                                            isoTaus = cms.InputTag("l1extraParticles","isoTau"), #comment out?
                                            jets = cms.InputTag("l1extraParticles","Central"), #no forward jets in SLHC, so only considering centra LHC jets
                                            
                                            ##How to calibrate
                                            ##Scale factor = MC/RAW = a+b |eta| +c|eta|^2
                                            ##Give the coeffs for egamma and taus
                                            ##So you need to fit the eta correction with a
                                            ##parabola and add the coeffs here
                                            ## Same as we do for RCT Calibration
                                            ## 6.Aug.2010 -- Barrel/endcap calibration separated

                                            eGammaCoefficientsB = cms.vdouble(1.093,-0.1662,0.1321),
                                            tauCoefficientsB    = cms.vdouble(0.6865,-0.0261,0.02295),
                                            eGammaCoefficientsE = cms.vdouble(1.33,-0.0949,0.0),
                                            tauCoefficientsE    = cms.vdouble(0.6865,-0.0261,0.02295),
                                            
                                            #eGammaCoefficientsB = cms.vdouble(1.0,0.0,0.0),
                                            #tauCoefficientsB    = cms.vdouble(1.0,0.0,0.0),
                                            #eGammaCoefficientsE = cms.vdouble(1.0,0.0,0.0),
                                            #tauCoefficientsE    = cms.vdouble(1.0,0.0,0.0)
                                            
                                            ## 18.Aug.2010 -- Added bin-by-bin correction factors to further improve upon functional form.
                                            ## These correction factors are pulled from the MC/Reco( |eta| ) plots after the above calibrations
                                            ## were applied.  Factors of 1.0 correspond to the bins where the LHC trigger objects are not produced.


                                            #Calibration good for 5 GeV thresholds
                                            #eGammaBinCorr = cms.vdouble( 1.001927, 1.000000, 1.051150, 1.000000, 1.054222, 1.000000, 1.041449, 1.094124, 1.000000, 1.035675, 1.000000, 1.000000, 1.022532),
                                            #tauBinCorr = cms.vdouble( 0.922142, 1.000000, 0.934793, 1.000000, 0.960299, 1.000000, 0.977273, 1.023410, 1.000000, 1.004015, 1.000000, 1.000000, 0.928593)
                                            
                                            #Calibration good for 30 GeV threshold.
                                            eGammaBinCorr = cms.vdouble( 1.044700, 1.000000, 1.045531, 1.000000, 1.025850, 1.000000, 0.994333, 1.000238, 1.000000, 0.984849, 1.000000, 1.000000, 0.988361),
                                            tauBinCorr=cms.vdouble( 0.908360, 1.000000, 0.904475, 1.000000, 0.913780, 1.000000, 0.927715, 0.969766, 1.000000, 0.953172, 1.000000, 1.000000, 0.900906),
                                                                                
                                            #eGammaBinCorr = cms.vdouble( 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0 , 1.0),
                                            #tauBinCorr = cms.vdouble( 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
                                                                                                                         ## June 2013
                                            ## New calibration. Corrections computed in bins of |eta| + linear interpolation
                                            ## Only for e/g
                                            applyNewCalib = cms.bool(False), # switch between old and new calibration
					    # These numbers are for SLHC, don't use them here
                                            eGammaEtaPoints = cms.vdouble(0.125, 0.375, 0.625, 0.875, 1.125, 1.3645, 1.6145, 1.875, 2.125, 2.375),
                                            eGammaNewCorr = cms.vdouble(0.0952467, 0.101389, 0.10598, 0.12605, 0.162749, 0.193123, 0.249227, 0.2800289, 0.271548, 0.27855)
                                            )


## New e/g clustering
# Calibration, only for e/g for the moment
SLHCL1ExtraParticlesNewClustering = cms.EDProducer("L1NewEgammaExtraCalibrator",
                                                   eGamma = cms.InputTag("rawSLHCL1ExtraParticlesNewClustering","EGamma"),
                                                   isoEGamma = cms.InputTag("rawSLHCL1ExtraParticlesNewClustering","IsoEGamma"),

                                                   ## June 2013
                                                   ## New calibration. Corrections computed in bins of |eta| and interpolated with a pol1
                                                   eGammaEtaPoints = cms.vdouble(0.1250,0.3750,0.6250,0.8750,1.1250,1.3645,1.6145,1.8750,2.1250,2.3750),
                                                   eGammaNewCorr = cms.vdouble(0.0932,0.1016,0.1049,0.1262,0.1569,0.1862,0.2320,0.2765,0.2679,0.2679)
                                                  )
## Calibration factors for DATA
#SLHCL1ExtraParticlesNewClustering.eGammaNewCorr = cms.vdouble(0.0932,0.1016,0.1049,0.1262,0.1569,0.1862,0.2320,0.2765,0.2679,0.2679)

## Calibration factors for MC
#SLHCL1ExtraParticlesNewClustering.eGammaNewCorr = cms.vdouble(0.0429,0.0492,0.0546,0.0722,0.1047,0.1238,0.2220,0.2436,0.2106,0.2042)

## Calibration factors to scale new clusters on current trigger, DATA
#SLHCL1ExtraParticlesNewClustering.eGammaNewCorr = cms.vdouble(0.0771,0.0826,0.0839,0.0934,0.1240,0.1349,0.1923,0.2193,0.2052,0.1733)

## Calibration factors to scale new clusters on current trigger, MC
#SLHCL1ExtraParticlesNewClustering.eGammaNewCorr = cms.vdouble(0.0866,0.1035,0.1067,0.1285,0.1734,0.1959,0.2239,0.2382,0.1797,0.1325)

# End new e/g clustering
