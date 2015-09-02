import FWCore.ParameterSet.Config as cms

def customiseForPreMixingInput(process):
    from PhysicsTools.PatAlgos.tools.helpers import massSearchReplaceAnyInputTag

    # Replace TrackingParticles and TrackingVertices globally
    for s in process.paths_().keys():
        massSearchReplaceAnyInputTag(getattr(process, s), cms.InputTag("mix", "MergedTrackTruth"), cms.InputTag("mixData", "MergedTrackTruth"), skipLabelTest=True) 

    for s in process.endpaths_().keys():
        massSearchReplaceAnyInputTag(getattr(process, s), cms.InputTag("mix", "MergedTrackTruth"), cms.InputTag("mixData", "MergedTrackTruth"), skipLabelTest=True)

    


    # Replace Pixel/StripDigiSimLinks only for the known modules
    def replaceInputTag(tag, old, new):
        if tag.value() == old:
            tag.setValue(new)

    def replacePixelDigiSimLink(tag):
        replaceInputTag(tag, "simSiPixelDigis", "mixData:PixelDigiSimLink")
    def replaceStripDigiSimLink(tag):
        replaceInputTag(tag, "simSiStripDigis", "mixData:StripDigiSimLink")

    for label, producer in process.producers_().iteritems():
        if producer.type_() == "ClusterTPAssociationProducer":
            replacePixelDigiSimLink(producer.pixelSimLinkSrc)
            replaceStripDigiSimLink(producer.stripSimLinkSrc)
        if producer.type_() == "QuickTrackAssociatorByHitsProducer":
            replacePixelDigiSimLink(producer.pixelSimLinkSrc)
            replaceStripDigiSimLink(producer.stripSimLinkSrc)
        if producer.type_() == "TrackAssociatorByHitsProducer":
            replacePixelDigiSimLink(producer.pixelSimLinkSrc)
            replaceStripDigiSimLink(producer.stripSimLinkSrc)
        if producer.type_() == "MuonAssociatorEDProducer":
            producer.DTdigisimlinkTag = cms.InputTag("mixData","simMuonDTDigis")
            producer.CSClinksTag = cms.InputTag("mixData","MuonCSCStripDigiSimLinks")
            producer.CSCwireLinksTag = cms.InputTag("mixData","MuonCSCWireDigiSimLinks")
            producer.RPCdigisimlinkTag = cms.InputTag("mixData","RPCDigiSimLink")
            replacePixelDigiSimLink(producer.pixelSimLinkSrc)
            replaceStripDigiSimLink(producer.stripSimLinkSrc)
        if producer.type_() == "MuonToTrackingParticleAssociatorEDProducer":
            producer.DTdigisimlinkTag = cms.InputTag("mixData","simMuonDTDigis")
            producer.CSClinksTag = cms.InputTag("mixData","MuonCSCStripDigiSimLinks")
            producer.CSCwireLinksTag = cms.InputTag("mixData","MuonCSCWireDigiSimLinks")
            producer.RPCdigisimlinkTag = cms.InputTag("mixData","RPCDigiSimLink")
            replacePixelDigiSimLink(producer.pixelSimLinkSrc)
            replaceStripDigiSimLink(producer.stripSimLinkSrc)

    for label, analyzer in process.analyzers_().iteritems():
        if analyzer.type_() == "GlobalRecHitsAnalyzer":
            replacePixelDigiSimLink(analyzer.pixelSimLinkSrc)
            replaceStripDigiSimLink(analyzer.stripSimLinkSrc)
        if analyzer.type_() == "SiPixelTrackingRecHitsValid":
            replacePixelDigiSimLink(analyzer.pixelSimLinkSrc)
            replaceStripDigiSimLink(analyzer.stripSimLinkSrc)
        if analyzer.type_() == "SiStripTrackingRecHitsValid":
            replacePixelDigiSimLink(analyzer.pixelSimLinkSrc)
            replaceStripDigiSimLink(analyzer.stripSimLinkSrc)
        if analyzer.type_() == "SiPixelRecHitsValid":
            replacePixelDigiSimLink(analyzer.pixelSimLinkSrc)
            replaceStripDigiSimLink(analyzer.stripSimLinkSrc)
        if analyzer.type_() == "SiStripRecHitsValid":
            replacePixelDigiSimLink(analyzer.pixelSimLinkSrc)
            replaceStripDigiSimLink(analyzer.stripSimLinkSrc)




    return process
