import FWCore.ParameterSet.Config as cms

def customise(process):
    if hasattr(process,'mix') and hasattr(process.mix,'digitizers') and hasattr(process.mix.digitizers,'hcal'):
        process.mix.digitizers.hcal.debugCaloSamples = cms.bool(True)

    from SimCalorimetry.HcalSimProducers.hcalUnsuppressedDigis_cfi import hcalSimBlock
    process.CaloSamplesAnalyzer = cms.EDAnalyzer("CaloSamplesAnalyzer",
        hcalSimBlock,
        CaloSamplesTag = cms.InputTag("mix","HcalSamples"),
    )

    process.TFileService = cms.Service("TFileService",
        fileName = cms.string("debugcalosamples.root")
    )
    
    process.debug_step = cms.Path(process.CaloSamplesAnalyzer)
    process.schedule.extend([process.debug_step])
    
    return process
