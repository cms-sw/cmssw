import FWCore.ParameterSet.Config as cms

def applyFilters( process ) :


    ## The beam scraping filter __________________________________________________||
    process.noscraping = cms.EDFilter(
        "FilterOutScraping",
        applyfilter = cms.untracked.bool(True),
        debugOn = cms.untracked.bool(False),
        numtrack = cms.untracked.uint32(10),
        thresh = cms.untracked.double(0.25)
        )

    ## The iso-based HBHE noise filter ___________________________________________||
    #process.load('CommonTools.RecoAlgos.HBHENoiseFilter_cfi')

    ## The CSC beam halo tight filter ____________________________________________||
    #process.load('RecoMET.METAnalyzers.CSCHaloFilter_cfi')
    process.load("RecoMET.METFilters.metFilters_cff")

    ## The HCAL laser filter _____________________________________________________||
    process.load("RecoMET.METFilters.hcalLaserEventFilter_cfi")
    process.hcalLaserEventFilter.vetoByRunEventNumber=cms.untracked.bool(False)
    process.hcalLaserEventFilter.vetoByHBHEOccupancy=cms.untracked.bool(True)

    ## The ECAL dead cell trigger primitive filter _______________________________||
    process.load('RecoMET.METFilters.EcalDeadCellTriggerPrimitiveFilter_cfi')
    ## For AOD and RECO recommendation to use recovered rechits
    process.EcalDeadCellTriggerPrimitiveFilter.tpDigiCollection = cms.InputTag("ecalTPSkimNA")

    ## The EE bad SuperCrystal filter ____________________________________________||
    process.load('RecoMET.METFilters.eeBadScFilter_cfi')

    ## The Good vertices collection needed by the tracking failure filter ________||
    process.goodVertices = cms.EDFilter(
      "VertexSelector",
      filter = cms.bool(False),
      src = cms.InputTag("offlinePrimaryVertices"),
      cut = cms.string("!isFake && ndof > 4 && abs(z) <= 24 && position.rho < 2")
    )

    ## The tracking failure filter _______________________________________________||
    process.load('RecoMET.METFilters.trackingFailureFilter_cfi')
    process.load('RecoMET.METFilters.trackingPOGFilters_cfi')


    # Tracking TOBTEC fakes filter ##
    # if true, only events passing filter (bad events) will pass
    process.tobtecfakesfilter.filter=cms.bool(False) 




    ## The good primary vertex filter ____________________________________________||
    pvSrc = 'offlinePrimaryVertices'
    process.primaryVertexFilter = cms.EDFilter(
        "VertexSelector",
        src = cms.InputTag("offlinePrimaryVertices"),
        cut = cms.string("!isFake & ndof > 4 & abs(z) <= 24 & position.Rho <= 2"),
        filter = cms.bool(True)
        )


    from PhysicsTools.SelectorUtils.pvSelector_cfi import pvSelector
    process.goodOfflinePrimaryVertices = cms.EDFilter(
        "PrimaryVertexObjectFilter",
        filterParams = pvSelector.clone( maxZ = cms.double(24.0),
                                         minNdof = cms.double(4.0) # this is >= 4
                                         ),
        src=cms.InputTag(pvSrc)
        )


