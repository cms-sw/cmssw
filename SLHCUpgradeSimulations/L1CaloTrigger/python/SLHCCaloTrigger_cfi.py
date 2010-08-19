import FWCore.ParameterSet.Config as cms

L1CaloTriggerSetupSource = cms.ESSource("EmptyESSource",
                                        recordName = cms.string('L1CaloTriggerSetupRcd'),
                                        firstValid = cms.vuint32(1),
                                        iovIsRunNotTime = cms.bool(True)
                                        )

L1CaloTriggerSetup = cms.ESProducer("L1CaloTriggerSetupProducer",
                                    InputXMLFile = cms.FileInPath('SLHCUpgradeSimulations/L1CaloTrigger/data/setup.xml')
                                    )

L1CaloTowerProducer = cms.EDProducer("L1CaloTowerProducer",
    ECALDigis = cms.InputTag("simEcalTriggerPrimitiveDigis"),
    HCALDigis = cms.InputTag("simHcalTriggerPrimitiveDigis")
)

L1CaloRegionProducer = cms.EDProducer("L1CaloRegionProducer",
                                      src = cms.InputTag("L1CaloTowerProducer")
)                                      

L1CaloClusterProducer = cms.EDProducer("L1CaloClusterProducer",
    src = cms.InputTag("L1CaloTowerProducer")
)

L1CaloClusterFilter = cms.EDProducer("L1CaloClusterFilter",
    src = cms.InputTag("L1CaloClusterProducer")
)

L1CaloClusterIsolator = cms.EDProducer("L1CaloClusterIsolator",
    src = cms.InputTag("L1CaloClusterFilter")
)

L1CaloJetProducer = cms.EDProducer("L1CaloJetProducer",
    src = cms.InputTag("L1CaloRegionProducer")
)

L1CaloJetFilter = cms.EDProducer("L1CaloJetFilter",
    src = cms.InputTag("L1CaloJetProducer")
)

L1CaloJetExpander = cms.EDProducer("L1CaloJetExpander",
    src = cms.InputTag("L1CaloJetFilter")
)


rawSLHCL1ExtraParticles = cms.EDProducer("L1ExtraTranslator",
                                  Clusters = cms.InputTag("L1CaloClusterIsolator"),
                                  Jets = cms.InputTag("L1CaloJetExpander"),
                                  NParticles = cms.int32(8),
                                  NJets      = cms.int32(12)
                              
)

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

                                      eGammaCoefficientsB = cms.vdouble(1.05,0.0,0.111),
                                      tauCoefficientsB    = cms.vdouble(1.36,0.0,0.0286),
                                      eGammaCoefficientsE = cms.vdouble(1.43,-0.470,0.0),
                                      tauCoefficientsE    = cms.vdouble(1.52,-0.197,0.0),

                                      ## 18.Aug.2010 -- Added bin-by-bin correction factors to further improve upon functional form.
                                      ## These correction factors are pulled from the MC/Reco( |eta| ) plots after the above calibrations
                                      ## were applied.
                                      eGammaBinCorr = cms.vdouble( 0.995280 , 0.999074 , 1.014316 , 1.005148 , 0.992236 , 0.998246 , 1.017787 , 1.137927 , 1.105248 , 0.948162 , 0.896218 , 0.774255 , 0.886956),
                                      tauBinCorr = cms.vdouble( 1.065728, 1.054150, 1.090152, 1.102329, 1.105562, 1.143280, 1.149978, 1.187399, 1.156080, 1.095267, 1.078722, 0.994069, 1.053809)
                                                                          
                                      #eGammaCoefficientsB = cms.vdouble(1.0,0.0,0.0),
                                      #tauCoefficientsB    = cms.vdouble(1.0,0.0,0.0),
                                      #eGammaCoefficientsE = cms.vdouble(1.0,0.0,0.0),
                                      #tauCoefficientsE    = cms.vdouble(1.0,0.0,0.0)
                                      
                                      #eGammaCoefficients = cms.vdouble(1.05,-0.00288,0.108),
                                      #tauCoefficients    = cms.vdouble(1.34,0.002,0.06)
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

                                            eGammaCoefficientsB = cms.vdouble(1.25,0.0,0.03199),
                                            tauCoefficientsB    = cms.vdouble(1.26,0.0,0.0610),
                                            eGammaCoefficientsE = cms.vdouble(1.33,-0.0949,0.0),
                                            tauCoefficientsE    = cms.vdouble(1.32,-0.278,0.0),

                                            ## 18.Aug.2010 -- Added bin-by-bin correction factors to further improve upon functional form.
                                            ## These correction factors are pulled from the MC/Reco( |eta| ) plots after the above calibrations
                                            ## were applied.  Factors of 1.0 correspond to the bins where the LHC trigger objects are not produced.
                                            eGammaBinCorr = cms.vdouble( 1.001927, 1.000000, 1.051150, 1.000000, 1.054222, 1.000000, 1.041449, 1.094124, 1.000000, 1.035675, 1.000000, 1.000000, 1.228740),
                                            tauBinCorr = cms.vdouble( 0.922142, 1.000000, 0.934793, 1.000000, 0.960299, 1.000000, 0.977273, 1.023410, 1.000000, 1.004015, 1.000000, 1.000000, 0.969207)
                                                                     
                                            #eGammaCoefficientsB = cms.vdouble(1.0,0.0,0.0),
#                                            tauCoefficientsB    = cms.vdouble(1.0,0.0,0.0),
#                                            eGammaCoefficientsE = cms.vdouble(1.0,0.0,0.0),
#                                            tauCoefficientsE    = cms.vdouble(1.0,0.0,0.0)
                                            
                                            #eGammaCoefficients = cms.vdouble(1.217,-0.0086,0.0184),
                                            #tauCoefficients    = cms.vdouble(1.148,0.0007,0.08637)
                                           )
