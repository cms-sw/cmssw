import FWCore.ParameterSet.Config as cms
def customise(process):
    if hasattr(process,'digitisation_step'):
        process=customise_Digi_TTI(process)
	
    return process	

def customise_Digi_TTI(process):

    # keep bx=0 only for TrackingParticles  (still includes all the in-time PU)
    process.mix.digitizers.mergedtruth.maximumPreviousBunchCrossing = cms.uint32(0)
    process.mix.digitizers.mergedtruth.maximumSubsequentBunchCrossing = cms.uint32(0)

    # remove all PU from TrackingParticles
    process.mix.digitizers.mergedtruth.select = cms.PSet(
        lipTP = cms.double(1000),
        chargedOnlyTP = cms.bool(False),
        pdgIdTP = cms.vint32(),
        signalOnlyTP = cms.bool(True),
        minRapidityTP = cms.double(-5.0),
        minHitTP = cms.int32(0),
        ptMinTP = cms.double(0.),
        maxRapidityTP = cms.double(5.0),
        tipTP = cms.double(1000),
        stableOnlyTP = cms.bool( False )
     )
    return process


def l1EventContent_TTI(process):
    #extend the event content

    alist=['RAWSIM','FEVTDEBUG','FEVTDEBUGHLT','GENRAW','RAWSIMHLT','FEVT']
    for a in alist:
        b=a+'output'
        if hasattr(process,b):

            getattr(process,b).outputCommands.append('keep *_TTClustersFromPixelDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_TTStubsFromPixelDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_TTTracksFromPixelDigis_*_*')

            getattr(process,b).outputCommands.append('keep *_TTClusterAssociatorFromPixelDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_TTStubAssociatorFromPixelDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_TTTrackAssociatorFromPixelDigis_*_*')

            getattr(process,b).outputCommands.append('drop PixelDigiSimLinkedmDetSetVector_mix_*_*')
            getattr(process,b).outputCommands.append('drop PixelDigiedmDetSetVector_mix_*_*')

	    getattr(process,b).outputCommands.append('drop *_simSiPixelDigis_*_*')

	    # drop what is not used in the stubs
            getattr(process,b).outputCommands.append('drop *_TTStubAssociatorFromPixelDigis_StubRejected_*')
	    getattr(process,b).outputCommands.append('drop *_TTStubsFromPixelDigis_StubRejected_*')
	    getattr(process,b).outputCommands.append('drop *_TTClustersFromPixelDigis_ClusterInclusive_*')
	    getattr(process,b).outputCommands.append('drop *_TTClusterAssociatorFromPixelDigis_ClusterInclusive_*')

	    # other savings. The following collections can be obtained from RAW2DIGI
	    #getattr(process,b).outputCommands.append('drop EcalTriggerPrimitiveDigisSorted_simEcalTriggerPrimitiveDigis_*_*')
            #getattr(process,b).outputCommands.append('drop HcalTriggerPrimitiveDigisSorted_simHcalTriggerPrimitiveDigis_*_*')
	    getattr(process,b).outputCommands.append('drop *_simEcalDigis_*_*')
	    getattr(process,b).outputCommands.append('drop *_simEcalPreshowerDigis_*_*')
	    getattr(process,b).outputCommands.append('drop *_simHcalDigis_*_*')
	    #getattr(process,b).outputCommands.append('drop *_simSiStripDigis_*_*')

    return process


def l1EventContent_TTI_forHLT(process):

    # as above, but keep the tracker digis. In case we want to produce some
    # samples that the HLT could use for CPU studies...

    alist=['RAWSIM','FEVTDEBUG','FEVTDEBUGHLT','GENRAW','RAWSIMHLT','FEVT']
    for a in alist:
        b=a+'output'
        if hasattr(process,b):

            getattr(process,b).outputCommands.append('keep *_TTClustersFromPixelDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_TTStubsFromPixelDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_TTTracksFromPixelDigis_*_*')

            getattr(process,b).outputCommands.append('keep *_TTClusterAssociatorFromPixelDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_TTStubAssociatorFromPixelDigis_*_*')
            getattr(process,b).outputCommands.append('keep *_TTTrackAssociatorFromPixelDigis_*_*')

            getattr(process,b).outputCommands.append('drop PixelDigiSimLinkedmDetSetVector_mix_*_*')
            getattr(process,b).outputCommands.append('drop PixelDigiedmDetSetVector_mix_*_*')

	    # that's the only difference w.r.t. l1EventContent_TTI :
            getattr(process,b).outputCommands.append('keep *_simSiPixelDigis_*_*')

            # drop what is not used in the stubs
            getattr(process,b).outputCommands.append('drop *_TTStubAssociatorFromPixelDigis_StubRejected_*')
            getattr(process,b).outputCommands.append('drop *_TTStubsFromPixelDigis_StubRejected_*')
            getattr(process,b).outputCommands.append('drop *_TTClustersFromPixelDigis_ClusterInclusive_*')
            getattr(process,b).outputCommands.append('drop *_TTClusterAssociatorFromPixelDigis_ClusterInclusive_*')

            # other savings. The following collections can be obtained from RAW2DIGI
            getattr(process,b).outputCommands.append('drop *_simEcalDigis_*_*')
            getattr(process,b).outputCommands.append('drop *_simEcalPreshowerDigis_*_*')
            getattr(process,b).outputCommands.append('drop *_simHcalDigis_*_*')

    return process


