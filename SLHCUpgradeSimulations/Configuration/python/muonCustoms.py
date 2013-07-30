import FWCore.ParameterSet.Config as cms


def unganged_me1a_geometry(process):
    """Customise digi/reco geometry to use unganged ME1/a channels
    """
    process.CSCGeometryESModule.useGangedStripsInME1a = False
    process.idealForDigiCSCGeometry.useGangedStripsInME1a = False
    return process


def digitizer_timing_pre3_median(process):
    """CSC digitizer customization 
    with bunchTimingOffsets tuned to center trigger stubs at bx6
    when pretrigger with 3 layers and median stub timing are used
    """
    ## Make sure there's no bad chambers/channels
    #process.simMuonCSCDigis.strips.readBadChambers = True
    #process.simMuonCSCDigis.wires.readBadChannels = True
    #process.simMuonCSCDigis.digitizeBadChambers = True

    ## Customised timing offsets so that ALCTs and CLCTs times are centered in signal BX. 
    ## These offsets below were tuned for the case of 3 layer pretriggering 
    ## and median stub timing algorithm.
    process.simMuonCSCDigis.strips.bunchTimingOffsets = cms.vdouble(0.0,
        37.53, 37.66, 55.4, 48.2, 54.45, 53.78, 53.38, 54.12, 51.98, 51.28)
    process.simMuonCSCDigis.wires.bunchTimingOffsets = cms.vdouble(0.0,
        22.88, 22.55, 29.28, 30.0, 30.0, 30.5, 31.0, 29.5, 29.1, 29.88)

    return process


def customise_csc_cond_ungangedME11A_mc(process):
    """ Pick up upgrade condions data directly from DB tags using ESPrefer's.
    Might be useful when dealing with a global tag that doesn't include 
    'unganged' CSC conditions.
    """
    myconds = [
        ('CSCDBGainsRcd',       'CSCDBGains_ungangedME11A_mc'),
        ('CSCDBNoiseMatrixRcd', 'CSCDBNoiseMatrix_ungangedME11A_mc'),
        ('CSCDBCrosstalkRcd',   'CSCDBCrosstalk_ungangedME11A_mc'),
        ('CSCDBPedestalsRcd',   'CSCDBPedestals_ungangedME11A_mc'),
        ('CSCDBGasGainCorrectionRcd',   'CSCDBGasGainCorrection_ungangedME11A_mc'),
        ('CSCDBChipSpeedCorrectionRcd', 'CSCDBChipSpeedCorrection_ungangedME11A_mc')
    ]

    from CalibMuon.Configuration.getCSCConditions_frontier_cff import cscConditions
    for (classname, tag) in myconds:
      print classname, tag
      sourcename = 'unganged_' + classname
      process.__setattr__(sourcename, cscConditions.clone())
      process.__getattribute__(sourcename).toGet = cms.VPSet( cms.PSet( record = cms.string(classname), tag = cms.string(tag)) )
      process.__getattribute__(sourcename).connect = cms.string('frontier://FrontierProd/CMS_COND_CSC_000')
      process.__setattr__('esp_' + classname, cms.ESPrefer("PoolDBESSource", sourcename) )
    
    del cscConditions

    return process


def customise_csc_Indexing(process):
    """Settings for the upgrade raw vs offline condition channel translation
    """
    process.CSCIndexerESProducer.AlgoName=cms.string("CSCIndexerPostls1")
    process.CSCChannelMapperESProducer.AlgoName=cms.string("CSCChannelMapperPostls1")
    return process


def remove_from_all_paths(process, module_name):
    """Remove process.module_name from all the paths in process:
    """

    # trivial case first:
    if not hasattr(process, module_name):
        return process

    # walk over all the paths:
    for path_name in process._Process__paths.keys():
        the_path = getattr(process, path_name)
        module_names = the_path.moduleNames()
        if module_name in module_names:
            the_path.remove(getattr(process, module_name))
    return process


def csc_PathVsModule_SanityCheck(process):
    """A sanity check to make sure that standard cmsDriver paths 
    have modules with expected names. If some paths would not have
    expected modules, the procedure would only print WARNINGs.
    """
    # list of (pathName, expected moduleName) tuples:
    paths_modules = [
        ('digitisation_step', 'simMuonCSCDigis'),
        ('L1simulation_step', 'simCscTriggerPrimitiveDigis'),
        ('L1simulation_step', 'simCsctfTrackDigis'),
        ('raw2digi_step', 'muonCSCDigis'),
        ('raw2digi_step', 'csctfDigis'),
        ('digi2raw_step', 'cscpacker'),
        ('digi2raw_step', 'csctfpacker'),
        ('reconstruction', 'csc2DRecHits'),
        ('dqmoffline_step', 'muonAnalyzer'),
        #('dqmHarvesting', ''),
        ('validation_step', 'relvalMuonBits')
    ]
    # verify:
    for path_name, module_name in paths_modules:
        if hasattr(process, path_name) and not hasattr(process, module_name):
            print "WARNING: module %s is not in %s path!!!" % (module_name, path_name)
            print "         This path has the following modules:"
            print "         ", getattr(process, path_name).moduleNames(),"\n"


# ------------------------------------------------------------------ 

def customise_csc_Geometry(process):
    """Customise digi/reco geometry to use unganged ME1/a channels
    """
    process = unganged_me1a_geometry(process)
    return process


def customise_csc_Digitizer(process):
    """Customise CSC digitization to use unganged ME1/a channels
    """
    process = customise_csc_Indexing(process)
    process = digitizer_timing_pre3_median(process)
    return process


def customise_csc_L1Stubs(process):
    """Configure the local CSC trigger stubs emulator with the upgrade 
    algorithm version that efficiently uses unganged ME1a
    """

    process = customise_csc_Indexing(process)

    from L1Trigger.CSCTriggerPrimitives.cscTriggerPrimitiveDigisPostLS1_cfi import cscTriggerPrimitiveDigisPostLS1
    process.simCscTriggerPrimitiveDigis = cscTriggerPrimitiveDigisPostLS1
    process.simCscTriggerPrimitiveDigis.CSCComparatorDigiProducer = cms.InputTag( 'simMuonCSCDigis', 'MuonCSCComparatorDigi')
    process.simCscTriggerPrimitiveDigis.CSCWireDigiProducer = cms.InputTag( 'simMuonCSCDigis', 'MuonCSCWireDigi')

    return process


def customise_csc_L1TrackFinder(process):
    """Regular CSCTF configuration adapted to deal with unganged ME1a
    """

    from L1Trigger.CSCTrackFinder.csctfTrackDigisUngangedME1a_cfi import csctfTrackDigisUngangedME1a
    process.simCsctfTrackDigis = csctfTrackDigisUngangedME1a
    process.simCsctfTrackDigis.DTproducer = cms.untracked.InputTag("simDtTriggerPrimitiveDigis")
    process.simCsctfTrackDigis.SectorReceiverInput = cms.untracked.InputTag("simCscTriggerPrimitiveDigis", "MPCSORTED")

    return process


def customise_csc_L1Emulator(process):
    """Customise both stubs and TF emulators
    """
    process = customise_csc_L1Stubs(process)
    process = customise_csc_L1TrackFinder(process)
    return process


def customise_csc_Packer(process):
    """Get rid of process.cscpacker and process.csctfpacker in all the paths
    """
    process = remove_from_all_paths(process, 'cscpacker')
    process = remove_from_all_paths(process, 'csctfpacker')
    return process


def customise_csc_Unpacker(process):
    """Get rid of process.muonCSCDigis and process.csctfDigis in all the paths
    """
    process = remove_from_all_paths(process, 'muonCSCDigis')
    process = remove_from_all_paths(process, 'csctfDigis')
    return process


def customise_csc_L1Extra_allsim(process):
    """Adjust L1Extra producer's input tags for the use case
    when we want to run L1Extra without packing-unpacking first
    """
    l1ep = process.l1extraParticles
    #l1ep.centralBxOnly = cms.bool(True)
    #l1ep.produceMuonParticles = cms.bool(True)
    #l1ep.produceCaloParticles = cms.bool(False)
    #l1ep.ignoreHtMiss = cms.bool(False)
    l1ep.muonSource = cms.InputTag('simGmtDigis')
    l1ep.etTotalSource = cms.InputTag('simGctDigis')
    l1ep.nonIsolatedEmSource = cms.InputTag('simGctDigis', 'nonIsoEm')
    l1ep.etMissSource = cms.InputTag('simGctDigis')
    l1ep.forwardJetSource = cms.InputTag('simGctDigis', 'forJets')
    l1ep.centralJetSource = cms.InputTag('simGctDigis', 'cenJets')
    l1ep.tauJetSource = cms.InputTag('simGctDigis', 'tauJets')
    l1ep.isolatedEmSource = cms.InputTag('simGctDigis', 'isoEm')
    l1ep.etHadSource = cms.InputTag('simGctDigis')
    l1ep.htMissSource = cms.InputTag("simGctDigis")
    l1ep.hfRingEtSumsSource = cms.InputTag("simGctDigis")
    l1ep.hfRingBitCountsSource = cms.InputTag("simGctDigis")
    return process


def customise_csc_LocalReco(process):
    """Configure the CSC rechit producer 
    to handle unganged ME1a for upgrade studies
    """
    # 
    process = customise_csc_Indexing(process)

    # Turn off some flags for CSCRecHitD that are turned ON in default config
    process.csc2DRecHits.readBadChannels = cms.bool(False)
    process.csc2DRecHits.CSCUseGasGainCorrection = cms.bool(False)

    # Switch input for CSCRecHitD to  s i m u l a t e d  digis
    process.csc2DRecHits.wireDigiTag  = cms.InputTag("simMuonCSCDigis", "MuonCSCWireDigi")
    process.csc2DRecHits.stripDigiTag = cms.InputTag("simMuonCSCDigis", "MuonCSCStripDigi")

    return process


def customise_csc_DQM(process):
    """At this point: get rid of process.muonAnalyzer, adjust cscMonitor's input
    """
    process = remove_from_all_paths(process, 'muonAnalyzer')

    process.cscMonitor.clctDigiTag = cms.InputTag("simCscTriggerPrimitiveDigis")
    process.cscMonitor.stripDigiTag = cms.InputTag("simMuonCSCDigis", "MuonCSCStripDigi")
    process.cscMonitor.wireDigiTag = cms.InputTag("simMuonCSCDigis", "MuonCSCWireDigi")
    process.cscMonitor.alctDigiTag = cms.InputTag("simCscTriggerPrimitiveDigis")

    process.l1tCsctf.statusProducer=cms.InputTag("null")
    process.l1tCsctf.lctProducer=cms.InputTag("null")
    process.l1tCsctf.trackProducer=cms.InputTag("null")
    process.l1tCsctf.mbProducer=cms.InputTag("null")

    return process


def customise_csc_Validation(process):
    """At this point, just get rid of process.relvalMuonBits
    """
    process = remove_from_all_paths(process, 'relvalMuonBits')
    return process


def customise_csc_PostLS1(process):
    """Full set of the CSC PostLS1 related customizations.
    It's tied to specific expected module names.
    Therefore, a sanity check is done first to make sure that 
    standard cmsDriver paths have modules with such expected names.
    """

    # a simple sanity check first
    csc_PathVsModule_SanityCheck(process)

    # use unganged geometry
    process = customise_csc_Geometry(process)

    # digitizer
    if hasattr(process, 'simMuonCSCDigis'):
        process = customise_csc_Digitizer(process)

    # L1 stub emulator upgrade algorithm
    if hasattr(process, 'simCscTriggerPrimitiveDigis'):
        process = customise_csc_L1Stubs(process)

    # CSCTF that can deal with unganged ME1a
    if hasattr(process, 'simCsctfTrackDigis'):
        process = customise_csc_L1TrackFinder(process)

    # packer - simply get rid of it
    if hasattr(process, 'cscpacker') or hasattr(process, 'csctfpacker'):
        process = customise_csc_Packer(process)

    # unpacker - simply get rid of it
    if hasattr(process, 'muonCSCDigis') or hasattr(process, 'csctfDigis'):
        process = customise_csc_Unpacker(process)

    # CSC RecHiti producer adjustments 
    if hasattr(process, 'csc2DRecHits'):
        process = customise_csc_LocalReco(process)

    # DQM 
    if hasattr(process, 'cscMonitor'):
        process = customise_csc_DQM(process)

    # Validation
    if hasattr(process, 'relvalMuonBits'):
        process = customise_csc_Validation(process)

    return process

def customise_csc_hlt(process):
    
    process.CSCGeometryESModule.useGangedStripsInME1a = False
    
    process.hltCsc2DRecHits.readBadChannels = cms.bool(False)
    process.hltCsc2DRecHits.CSCUseGasGainCorrection = cms.bool(False)
    
    # Switch input for CSCRecHitD to  s i m u l a t e d  digis
    
    process.hltCsc2DRecHits.wireDigiTag  = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi")
    process.hltCsc2DRecHits.stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi")
    
    return process
