import FWCore.ParameterSet.Config as cms

phase2PixelDigitizer = cms.PSet(
    accumulatorType = cms.string("SiPhase2Digitizer"),
    hitsProducer = cms.string('g4SimHits'),
    makeDigiSimLinks = cms.untracked.bool(True),
#D.B.:the noise should be a function of strip capacitance, roughly: ReadoutNoiseInElec=500+(64*Cdet[pF]) ~= 500+(64*1.5[cm])
    ReadoutNoiseInElec = cms.double(1000.0),#D.B.:Fill readout noise, including all readout chain, relevant for smearing
    DeltaProductionCut = cms.double(0.03),
    ROUList = cms.vstring(
        'TrackerHitsPixelBarrelLowTof', 
        'TrackerHitsPixelBarrelHighTof', 
        'TrackerHitsPixelEndcapLowTof', 
        'TrackerHitsPixelEndcapHighTof'),
    OffsetSmearing = cms.double(0.0),
    ThresholdInElectrons_FPix = cms.double(4759.8), #D.B.: this should correspond to a threshold of 530mV    
    ThresholdInElectrons_BPix = cms.double(4759.8),
    ThresholdInElectrons_BPix_L1 = cms.double(4759.8),
    AddThresholdSmearing = cms.bool(True),
    ThresholdSmearing_FPix = cms.double(204.0),#D.B.: changed (~5mV peakToPeak --> 1.76mV rms) (was 210.0)
    ThresholdSmearing_BPix = cms.double(204.0),#D.B.: changed (~5mV peakToPeak --> 1.76mV rms) (was 245.0)
    ThresholdSmearing_BPix_L1 = cms.double(204.0),#D.B.: changed (~5mV peakToPeak --> 1.76mV rms) (was 245.0)
#    NoiseInElectrons = cms.double(1000.0),	#D.B.:this was set as the pixel cell noise, relevant for generating noisy pixels (was 175.0). But is should be the same as the total readout noise, ReadoutNoiseInElec
    NoiseInElectrons = cms.double(0.1),	#D.B.:this was set as the pixel cell noise, relevant for generating noisy pixels (was 175.0). But is should be the same as the total readout noise, ReadoutNoiseInElec NB: does not work if ==0!
    MissCalibrate = cms.bool(False),
    FPix_SignalResponse_p0 = cms.double(0.0043),	#D.B.:for pixel calibration only (not for PS or 2S)
    FPix_SignalResponse_p1 = cms.double(1.31),		#D.B.:for pixel calibration only (not for PS or 2S)
    FPix_SignalResponse_p2 = cms.double(93.6),		#D.B.:for pixel calibration only (not for PS or 2S)
    FPix_SignalResponse_p3 = cms.double(134.6),		#D.B.:for pixel calibration only (not for PS or 2S)
    BPix_SignalResponse_p0 = cms.double(0.0035),	#D.B.:for pixel calibration only (not for PS or 2S)
    BPix_SignalResponse_p1 = cms.double(1.23),		#D.B.:for pixel calibration only (not for PS or 2S)
    BPix_SignalResponse_p2 = cms.double(97.4),		#D.B.:for pixel calibration only (not for PS or 2S)
    BPix_SignalResponse_p3 = cms.double(126.5),		#D.B.:for pixel calibration only (not for PS or 2S)
    ElectronsPerVcal = cms.double(65.5),		#D.B.:used for misscalibration
    ElectronsPerVcal_Offset = cms.double(-414.0),	#D.B.:used for misscalibration
    ElectronPerAdc = cms.double(135.0),	#D.B.:used for misscalibration
    TofUpperCut = cms.double(12.5),
    AdcFullScale = cms.int32(255),
    AdcFullScaleStack = cms.int32(255),
    FirstStackLayer = cms.int32(5),   #D.B.:not used
    TofLowerCut = cms.double(-12.5),
    TanLorentzAnglePerTesla_FPix = cms.double(0.106),	#D.B.:this I have not checked yet
    TanLorentzAnglePerTesla_BPix = cms.double(0.106),	#D.B.:this I have not checked yet
    AddNoisyPixels = cms.bool(True),
    Alpha2Order = cms.bool(True),			#D.B.: second order effect, does not switch off magnetic field as described
    AddPixelInefficiency = cms.int32(0), # deprecated, use next option
    AddPixelInefficiencyFromPython = cms.bool(True),
    AddNoise = cms.bool(True),
#    AddXTalk = cms.bool(True),
    AddXTalk = cms.bool(True),			#D.B.
    InterstripCoupling = cms.double(0.08),	#D.B.
    SigmaZero = cms.double(0.00037),  		#D.B.: 3.7um spread for 300um-thick sensor, renormalized in digitizerAlgo
    SigmaCoeff = cms.double(1.80),  		#D.B.: to be confirmed with simulations in CMSSW_6.X
    ClusterWidth = cms.double(3),		#D.B.: this is used as number of sigmas for charge collection (3=+-3sigmas)
##    Dist300 = cms.double(0.03),  		#D.B.: use moduleThickness instead
    ChargeVCALSmearing = cms.bool(False),	#D.B.:changed from true, this smearing uses a calibration for the pixels, use gaussDistribution_ instead
    GainSmearing = cms.double(0.0),		#D.B.:is not used in phase2PixelDigitizer
    GeometryType = cms.string('idealForDigi'),                           
    useDB = cms.bool(True),				
    LorentzAngle_DB = cms.bool(True),			
    DeadModules_DB = cms.bool(True),
    killModules = cms.bool(True),
    NumPixelBarrel = cms.int32(3),
    NumPixelEndcap = cms.int32(2),
    thePixelColEfficiency_BPix1 = cms.double(0.999), 	# Only used when AddPixelInefficiency = true
    thePixelColEfficiency_BPix2 = cms.double(0.999),
    thePixelColEfficiency_BPix3 = cms.double(0.999),
    thePixelColEfficiency_FPix1 = cms.double(0.999),
    thePixelColEfficiency_FPix2 = cms.double(0.999),
    thePixelEfficiency_BPix1 = cms.double(0.999), 	# Only used when AddPixelInefficiency = true
    thePixelEfficiency_BPix2 = cms.double(0.999),
    thePixelEfficiency_BPix3 = cms.double(0.999),
    thePixelEfficiency_FPix1 = cms.double(0.999),
    thePixelEfficiency_FPix2 = cms.double(0.999),
    thePixelChipEfficiency_BPix1 = cms.double(0.999), 	# Only used when AddPixelInefficiency = true
    thePixelChipEfficiency_BPix2 = cms.double(0.999),
    thePixelChipEfficiency_BPix3 = cms.double(0.999),
    thePixelChipEfficiency_FPix1 = cms.double(0.999),
    thePixelChipEfficiency_FPix2 = cms.double(0.999),
    DeadModules = cms.VPSet()
)
