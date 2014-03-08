import FWCore.ParameterSet.Config as cms
def customise(process):
    if hasattr(process,'digitisation_step'):
        process=customise_Digi_TTI(process)
	
    return process	

def customise_Digi_TTI(process):
    # needed in addition to customise_Digi (only bx=0 for TrackingParticles)
    process.mix.digitizers.mergedtruth.maximumPreviousBunchCrossing = cms.uint32(0)
    process.mix.digitizers.mergedtruth.maximumSubsequentBunchCrossing = cms.uint32(0)
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
	    getattr(process,b).outputCommands.append('drop EcalTriggerPrimitiveDigisSorted_simEcalTriggerPrimitiveDigis_*_*')
            getattr(process,b).outputCommands.append('drop HcalTriggerPrimitiveDigisSorted_simHcalTriggerPrimitiveDigis_*_*')
	    getattr(process,b).outputCommands.append('drop *_simEcalDigis_*_*')
	    getattr(process,b).outputCommands.append('drop *_simEcalPreshowerDigis_*_*')
	    getattr(process,b).outputCommands.append('drop *_simHcalDigis_*_*')
	    getattr(process,b).outputCommands.append('drop *_simSiStripDigis_*_*')

    return process
