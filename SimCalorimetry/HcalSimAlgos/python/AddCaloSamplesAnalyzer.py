import FWCore.ParameterSet.Config as cms

def customise(process):
    # handle normal mixing or premixing
    hcaldigi = None
    if hasattr(process,'mix') and hasattr(process.mix,'digitizers') and hasattr(process.mix.digitizers,'hcal'):
        hcaldigi = process.mix.digitizers.hcal
        cstag = "mix"
    if hasattr(process,'mixData'):
        hcaldigi = process.mixData
        cstag = "mixData"
    if hcaldigi is None:
        raise Exception("CaloSamplesAnalyzer requires a mix module, none found!")

    hcaldigi.debugCaloSamples = cms.bool(True)
    process.CaloSamplesAnalyzer = cms.EDAnalyzer("CaloSamplesAnalyzer",
        # from hcalSimParameters
        hf1 = hcaldigi.hf1,
        hf2 = hcaldigi.hf2,
        ho = hcaldigi.ho,
        hb = hcaldigi.hb,
        he = hcaldigi.he,
        zdc = hcaldigi.zdc,
        hoZecotek = hcaldigi.hoZecotek,
        hoHamamatsu = hcaldigi.hoHamamatsu,
        # from hcalUnsuppressedDigis
        hitsProducer = hcaldigi.hitsProducer,
        TestNumbering = hcaldigi.TestNumbering,
        CaloSamplesTag = cms.InputTag(cstag,"HcalSamples"),
    )

    process.TFileService = cms.Service("TFileService",
        fileName = cms.string("debugcalosamples.root")
    )
    
    process.debug_step = cms.Path(process.CaloSamplesAnalyzer)
    process.schedule.extend([process.debug_step])
    
    return process
