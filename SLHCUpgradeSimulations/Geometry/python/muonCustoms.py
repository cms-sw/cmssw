import FWCore.ParameterSet.Config as cms

# define all the changes for unganging ME1a
def unganged_me1a(process):
    
    ### CSC geometry customization:

    #from Configuration.StandardSequences.GeometryDB_cff import *
    if not process.es_producers_().has_key('idealForDigiCSCGeometry'):
        process.load('Geometry.CSCGeometryBuilder.cscGeometryDB_cfi')
        process.load('Geometry.CSCGeometryBuilder.idealForDigiCscGeometryDB_cff')
    process.CSCGeometryESModule.useGangedStripsInME1a = False
    process.idealForDigiCSCGeometry.useGangedStripsInME1a = False

    ### Digitizer customization:
    
    if 'simMuonCSCDigis' not in process.producerNames():
        process.load('SimMuon.CSCDigitizer.muonCSCDigis_cfi')
    ## Make sure there's no bad chambers/channels 
    #process.simMuonCSCDigis.strips.readBadChambers = True
    #process.simMuonCSCDigis.wires.readBadChannels = True
    #process.simMuonCSCDigis.digitizeBadChambers = True
    
    ## Customized timing offsets so that ALCTs and CLCTs times
    ## are centered in signal BX. The offsets below were tuned for the case
    ## of 3 layer pretriggering and median stub timing algorithm.
    process.simMuonCSCDigis.strips.bunchTimingOffsets = cms.vdouble(0.0,
        37.53, 37.66, 55.4, 48.2, 54.45, 53.78, 53.38, 54.12, 51.98, 51.28)
    process.simMuonCSCDigis.wires.bunchTimingOffsets = cms.vdouble(0.0,
        22.88, 22.55, 29.28, 30.0, 30.0, 30.5, 31.0, 29.5, 29.1, 29.88)


    #done
    return process

# CSC geometry customization:
def unganged_me1a_geometry(process):
    process.CSCGeometryESModule.useGangedStripsInME1a = False
    process.idealForDigiCSCGeometry.useGangedStripsInME1a = False
    return process

# CSC digitizer customization
def digitizer_timing_pre3_median(process):

    ## Make sure there's no bad chambers/channels
    #process.simMuonCSCDigis.strips.readBadChambers = True
    #process.simMuonCSCDigis.wires.readBadChannels = True
    #process.simMuonCSCDigis.digitizeBadChambers = True

    ## Customized timing offsets so that ALCTs and CLCTs times are centered in signal BX. 
    ## These offsets below were tuned for the case of 3 layer pretriggering 
    ## and median stub timing algorithm.
    process.simMuonCSCDigis.strips.bunchTimingOffsets = cms.vdouble(0.0,
        37.53, 37.66, 55.4, 48.2, 54.45, 53.78, 53.38, 54.12, 51.98, 51.28)
    process.simMuonCSCDigis.wires.bunchTimingOffsets = cms.vdouble(0.0,
        22.88, 22.55, 29.28, 30.0, 30.0, 30.5, 31.0, 29.5, 29.1, 29.88)

    return process

# pick up upgrade condions data directly from DB tags using ESPrefer's
def customise_csc_cond_ungangedME11A_mc(process):

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


# Adjust L1Extra producer's input tags
def customize_l1extra(process):
    l1ep = process.l1extraParticles
    #l1ep.centralBxOnly = cms.bool(True)
    #l1ep.produceMuonParticles = cms.bool(True)
    #l1ep.produceCaloParticles = cms.bool(False)
    #l1ep.ignoreHtMiss = cms.bool(False)
    l1ep.muonSource = cms.InputTag('simGmtDigis')
    l1ep.etTotalSource = cms.InputTag('simGctDigis')
    l1ep.nonIsolatedEmSource = cms.InputTag('simGctDigis','nonIsoEm')
    l1ep.etMissSource = cms.InputTag('simGctDigis')
    l1ep.forwardJetSource = cms.InputTag('simGctDigis','forJets')
    l1ep.centralJetSource = cms.InputTag('simGctDigis','cenJets')
    l1ep.tauJetSource = cms.InputTag('simGctDigis','tauJets')
    l1ep.isolatedEmSource = cms.InputTag('simGctDigis','isoEm')
    l1ep.etHadSource = cms.InputTag('simGctDigis')
    l1ep.htMissSource = cms.InputTag("simGctDigis")
    l1ep.hfRingEtSumsSource = cms.InputTag("simGctDigis")
    l1ep.hfRingBitCountsSource = cms.InputTag("simGctDigis")
    return process


def customise_csc_geom_cond_digi(process):
    process = unganged_me1a_geometry(process)
    process = customise_csc_cond_ungangedME11A_mc(process)
    process = digitizer_timing_pre3_median(process)
    return process

