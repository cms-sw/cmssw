import FWCore.ParameterSet.Config as cms

ecalEBTrigPrimPhase2SpikeTaggerESProducer = cms.ESProducer("EcalEBTrigPrimPhase2SpikeTaggerESProducer",
    fwVersion = cms.uint32(1),

    # configuration PSets for the individual payload algorithms
    algoConfigs = cms.VPSet(
        cms.PSet(
            algo = cms.string("ld"),
            perCrystalParams = cms.VPSet(
                cms.PSet(
                    ietaRange = cms.string(":"), # Example range formats "ietaMin:ietaMax", e.g. "-85:42" (user defined), "1:" (positive side), ":" (whole EB eta range)
                    iphiRange = cms.string(":"), # Example range formats "iphiMin:iphiMax", e.g. "90:270" (user defined), ":180" (MIN_IPHI:180), ":" (MIN_IPHI:MAX_IPHI)
                    peakSampleIndex = cms.uint32(5),
                    spikeThreshold = cms.double(-0.1), # if the LD is below the spike flag is set
                    weights = cms.vdouble(1.5173, -2.1034, 1.8117, -0.6451) # LD weights in ascending order
                )
            )
        ),
    )
)

ecalEBPhase2TPGSpikeTaggerParamsSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('EcalEBPhase2TPGSpikeTaggerParamsRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)
