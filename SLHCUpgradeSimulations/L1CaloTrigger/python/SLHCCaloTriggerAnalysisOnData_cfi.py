import FWCore.ParameterSet.Config as cms


SelectTaus = cms.EDFilter("PFTauSelector",
                        src = cms.InputTag("hpsPFTauProducer"),
                        discriminators = cms.VPSet(
                              cms.PSet( discriminator=cms.InputTag("hpsPFTauDiscriminationByDecayModeFinding"),selectionCut=cms.double(0.5)),
                              cms.PSet( discriminator=cms.InputTag("hpsPFTauDiscriminationByMediumIsolation"),selectionCut=cms.double(0.5)),
                              cms.PSet( discriminator=cms.InputTag("hpsPFTauDiscriminationByTightMuonRejection"),selectionCut=cms.double(0.5))
)
)


IsolationAnalyzer= cms.EDAnalyzer('L1CaloClusterAnalyzer',
                              src  = cms.InputTag("L1CaloClusterIsolator"),
                              electrons      = cms.InputTag("gsfElectrons")
                              )


isoElectrons_Rate = cms.EDAnalyzer('CaloTriggerAnalyzerOnData',
                              SLHCsrc = cms.InputTag("rawSLHCL1ExtraParticles","IsoEGamma"),
                              LHCisosrc = cms.InputTag("l1extraParticles","Isolated"),
                              LHCsrc  = cms.InputTag("l1extraParticles","NonIsolated"),
                              iso    = cms.double(1)  ###1 for isolated 0 for not isolated 
                              )

Electrons_Rate = cms.EDAnalyzer('CaloTriggerAnalyzerOnData',
                              SLHCsrc = cms.InputTag("rawSLHCL1ExtraParticles","EGamma"),
                              LHCisosrc = cms.InputTag("l1extraParticles","Isolated"),
                              LHCsrc  = cms.InputTag("l1extraParticles","NonIsolated"),
                              iso     = cms.double(0)  ###1 for isolated 0 for not isolated 
                              )

isoElectrons_Rate_Cal = cms.EDAnalyzer('CaloTriggerAnalyzerOnData',
                              SLHCsrc = cms.InputTag("SLHCL1ExtraParticles","IsoEGamma"),
                              LHCisosrc = cms.InputTag("l1extraParticles","Isolated"),
                              LHCsrc  = cms.InputTag("l1extraParticles","NonIsolated"),
                              iso    = cms.double(1)  ###1 for isolated 0 for not isolated 
                              )

Electrons_Rate_Cal = cms.EDAnalyzer('CaloTriggerAnalyzerOnData',
                              SLHCsrc = cms.InputTag("SLHCL1ExtraParticles","EGamma"),
                              LHCisosrc = cms.InputTag("l1extraParticles","Isolated"),
                              LHCsrc  = cms.InputTag("l1extraParticles","NonIsolated"),
                              iso     = cms.double(0)  ###1 for isolated 0 for not isolated 
                              )


Taus_Rate = cms.EDAnalyzer('CaloTriggerAnalyzerOnData',
                              SLHCsrc = cms.InputTag("rawSLHCL1ExtraParticles","Taus"),
                              LHCisosrc = cms.InputTag("l1extraParticles","Tau"),
                              LHCsrc  = cms.InputTag("l1extraParticles","Tau"),
                              iso     = cms.double(0)  ###1 for isolated 0 for not isolated 
                              )


isoTaus_Rate = cms.EDAnalyzer('CaloTriggerAnalyzerOnData',
                              SLHCsrc = cms.InputTag("rawSLHCL1ExtraParticles","IsoTaus"),
                              LHCisosrc = cms.InputTag("l1extraParticles","Tau"),
                              LHCsrc  = cms.InputTag("l1extraParticles","Tau"),
                              iso     = cms.double(1)  ###1 for isolated 0 for not isolated 
                              )


Taus_Rate_Cal = cms.EDAnalyzer('CaloTriggerAnalyzerOnData',
                              SLHCsrc = cms.InputTag("SLHCL1ExtraParticles","Taus"),
                              LHCisosrc = cms.InputTag("l1extraParticles","Tau"),
                              LHCsrc  = cms.InputTag("l1extraParticles","Tau"),
                              iso     = cms.double(0)  ###1 for isolated 0 for not isolated 
                              )


isoTaus_Rate_Cal = cms.EDAnalyzer('CaloTriggerAnalyzerOnData',
                              SLHCsrc = cms.InputTag("SLHCL1ExtraParticles","IsoTaus"),
                              LHCisosrc = cms.InputTag("l1extraParticles","Tau"),
                              LHCsrc  = cms.InputTag("l1extraParticles","Tau"),
                              iso     = cms.double(1)  ###1 for isolated 0 for not isolated 
                              )

####Trees:
isoElectrons = cms.EDAnalyzer('CaloTriggerAnalyzerOnDataTrees',
                           LHCsrc    = cms.InputTag("l1extraParticles","Isolated"),
                           LHCisosrc = cms.InputTag("l1extraParticles","Isolated"),
                           SLHCsrc   = cms.InputTag("rawSLHCL1ExtraParticles","IsoEGamma"),
                           electrons      = cms.InputTag("gsfElectrons"), ####added for particleflow
                           iso     = cms.double(1),  ###1 for isolated 0 for not isolated 
                           deltaR = cms.double(0.3),
                           threshold = cms.double(30.0),
                           VertexCollection = cms.InputTag("offlinePrimaryVertices")   
                           )

Electrons = cms.EDAnalyzer('CaloTriggerAnalyzerOnDataTrees',
                           LHCsrc    = cms.InputTag("l1extraParticles","NonIsolated"),
                           LHCisosrc = cms.InputTag("l1extraParticles","Isolated"),
                           SLHCsrc   = cms.InputTag("rawSLHCL1ExtraParticles","EGamma"),
                           electrons = cms.InputTag("gsfElectrons"), ####added for particleflow
                           iso       = cms.double(0),  ###1 for isolated 0 for not isolated 
                           deltaR    = cms.double(0.3),
                           threshold = cms.double(30.0),
                           VertexCollection = cms.InputTag("offlinePrimaryVertices")   
                           )
isoElectrons_Cal = cms.EDAnalyzer('CaloTriggerAnalyzerOnDataTrees',
                           LHCsrc    = cms.InputTag("l1extraParticles","Isolated"),
                           LHCisosrc = cms.InputTag("l1extraParticles","Isolated"),
                           SLHCsrc   = cms.InputTag("SLHCL1ExtraParticles","IsoEGamma"),
                           electrons      = cms.InputTag("gsfElectrons"), ####added for particleflow
                           iso     = cms.double(1),  ###1 for isolated 0 for not isolated 
                           deltaR = cms.double(0.3),
                           threshold = cms.double(30.0),
                           VertexCollection = cms.InputTag("offlinePrimaryVertices")   
                           )

Electrons_Cal = cms.EDAnalyzer('CaloTriggerAnalyzerOnDataTrees',
                           LHCsrc    = cms.InputTag("l1extraParticles","NonIsolated"),
                           LHCisosrc = cms.InputTag("l1extraParticles","Isolated"),
                           SLHCsrc   = cms.InputTag("SLHCL1ExtraParticles","EGamma"),
                           electrons = cms.InputTag("gsfElectrons"), ####added for particleflow
                           iso       = cms.double(0),  ###1 for isolated 0 for not isolated 
                           deltaR    = cms.double(0.3),
                           threshold = cms.double(30.0),
                           VertexCollection = cms.InputTag("offlinePrimaryVertices")   
                           )


Taus = cms.EDAnalyzer('CaloTriggerAnalyzerTaus',
                             LHCisosrc = cms.InputTag("l1extraParticles","Tau"),
                             SLHCsrc   = cms.InputTag("rawSLHCL1ExtraParticles","IsoTaus"),
                             LHCsrc    = cms.InputTag("l1extraParticles","Tau"),
                             iso       = cms.double(0),
                             deltaR    = cms.double(0.5),
                             threshold = cms.double(20.0),
                             goodTaus  = cms.InputTag("SelectTaus")
                             )


Taus_Cal = cms.EDAnalyzer('CaloTriggerAnalyzerTaus',
                             LHCisosrc = cms.InputTag("l1extraParticles","Tau"),
                             SLHCsrc   = cms.InputTag("SLHCL1ExtraParticles","IsoTaus"),
                             LHCsrc    = cms.InputTag("l1extraParticles","Tau"),
                             iso       = cms.double(0),
                             deltaR    = cms.double(0.5),
                             threshold = cms.double(20.0),
                             goodTaus  = cms.InputTag("SelectTaus")
                             )


isoTaus = cms.EDAnalyzer('CaloTriggerAnalyzerTaus',
                             LHCisosrc = cms.InputTag("l1extraParticles","Tau"),
                             SLHCsrc   = cms.InputTag("rawSLHCL1ExtraParticles","Taus"),
                             LHCsrc    = cms.InputTag("l1extraParticles","Tau"),
                             iso       = cms.double(1),
                             deltaR    = cms.double(0.5),
                             threshold = cms.double(30.0),
                             goodTaus  = cms.InputTag("SelectTaus")
                             )


isoTaus_Cal = cms.EDAnalyzer('CaloTriggerAnalyzerTaus',
                             LHCisosrc = cms.InputTag("l1extraParticles","Tau"),
                             SLHCsrc   = cms.InputTag("SLHCL1ExtraParticles","Taus"),
                             LHCsrc    = cms.InputTag("l1extraParticles","Tau"),
                             iso       = cms.double(1),
                             deltaR    = cms.double(0.5),
                             threshold = cms.double(30.0),
                             goodTaus  = cms.InputTag("SelectTaus")
                             )



##Select Processes to run here
analysisSequenceCalibrated = cms.Sequence(
    #SelectTaus*
    #Electrons_Rate*
    Electrons*
    isoElectrons*
    #isoElectrons_Rate*
    isoElectrons_Cal*
    Electrons_Rate_Cal*
    Electrons_Cal*
    isoElectrons_Rate_Cal*
    IsolationAnalyzer
    #isoTaus_Cal*
    #isoTaus_Rate_Cal*
    #Taus_Cal*
    #Taus_Rate_Cal
    #Taus_Rate*
    #isoTaus_Rate
    #Taus*
    #isoTaus
    )

