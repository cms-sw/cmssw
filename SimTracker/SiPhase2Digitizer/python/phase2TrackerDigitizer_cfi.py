import FWCore.ParameterSet.Config as cms

PixelDigitizerAlgorithmCommon = cms.PSet(
    ElectronPerAdc = cms.double(1500.0),
    ReadoutNoiseInElec = cms.double(-99.9),       # not used at the moment
    ThresholdInElectrons_Barrel = cms.double(1000.0),
    ThresholdInElectrons_Endcap = cms.double(1000.0),
    AddThresholdSmearing = cms.bool(False),
    ThresholdSmearing_Barrel = cms.double(0.0),
    ThresholdSmearing_Endcap = cms.double(0.0),
    HIPThresholdInElectrons_Barrel = cms.double(1.0e10), # very high value to avoid Over threshold bit
    HIPThresholdInElectrons_Endcap = cms.double(1.0e10), # very high value to avoid Over threshold bit
    NoiseInElectrons = cms.double(0.0),
    Phase2ReadoutMode = cms.int32(3), # Flag to decide Readout Mode :Digital(0) or Analog (linear TDR (-1), dual slope with slope parameters (+1,+2,+3,+4) with threshold subtraction
    AdcFullScale = cms.int32(15),
    TofUpperCut = cms.double(20.),
    TofLowerCut = cms.double(-5.),
    AddNoisyPixels = cms.bool(False),
    Alpha2Order = cms.bool(True),			#D.B.: second order effect, does not switch off magnetic field as described
    AddNoise = cms.bool(False),
    AddXTalk = cms.bool(False),			#D.B.
    InterstripCoupling = cms.double(0.0),	#D.B. # No need to be used in PixelDigitizerAlgorithm
    Odd_row_interchannelCoupling_next_row = cms.double(0.20),
    Even_row_interchannelCoupling_next_row = cms.double(0.0),
    Odd_column_interchannelCoupling_next_column = cms.double(0.0),
    Even_column_interchannelCoupling_next_column = cms.double(0.0),
    SigmaZero = cms.double(0.00037),  		#D.B.: 3.7um spread for 300um-thick sensor, renormalized in digitizerAlgo
    SigmaCoeff = cms.double(0),  		#S.D: setting SigmaCoeff=0 for IT-pixel
    ClusterWidth = cms.double(3),		#D.B.: this is used as number of sigmas for charge collection (3=+-3sigmas)
    LorentzAngle_DB = cms.bool(True),			
    TanLorentzAnglePerTesla_Endcap = cms.double(0.106),
    TanLorentzAnglePerTesla_Barrel = cms.double(0.106),
    KillModules = cms.bool(False),
    DeadModules_DB = cms.bool(False),
    DeadModules = cms.VPSet(),
    AddInefficiency = cms.bool(False),
    Inefficiency_DB = cms.bool(False),				
    UseReweighting = cms.bool(False),
    EfficiencyFactors_Barrel = cms.vdouble(0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999 ),
    EfficiencyFactors_Endcap = cms.vdouble(0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 
                                           0.999, 0.999 ),#Efficiencies kept as Side2Disk1,Side1Disk1 and so on
    CellsToKill = cms.VPSet(),
    ApplyTimewalk = cms.bool(False),
    TimewalkModel = cms.PSet(
        ThresholdValues = cms.vdouble(1000, 1200, 1500, 3000),
        Curves = cms.VPSet(
            cms.PSet(
                charge = cms.vdouble(1000, 1025, 1050, 1100, 1200, 1500, 2000, 6000, 10000, 15000, 20000, 30000),
                delay = cms.vdouble(26.8, 23.73, 21.92, 19.46, 16.52, 12.15, 8.88, 3.03, 1.69, 0.95, 0.56, 0.19)
            ),
            cms.PSet(
                charge = cms.vdouble(1200, 1225, 1250, 1500, 2000, 6000, 10000, 15000, 20000, 30000),
                delay = cms.vdouble(26.28, 23.5, 21.79, 14.92, 10.27, 3.33, 1.86, 1.07, 0.66, 0.27)
            ),
            cms.PSet(
                charge = cms.vdouble(1500, 1525, 1550, 1600, 2000, 6000, 10000, 15000, 20000, 30000),
                delay = cms.vdouble(25.36, 23.05, 21.6, 19.56, 12.94, 3.79, 2.14, 1.26, 0.81, 0.39)
            ),
            cms.PSet(
                charge = cms.vdouble(3000, 3025, 3050, 3100, 3500, 6000, 10000, 15000, 20000, 30000),
                delay = cms.vdouble(25.63, 23.63, 22.35, 20.65, 14.92, 6.7, 3.68, 2.29, 1.62, 1.02)
            )
        )
    )
)
phase2TrackerDigitizer = cms.PSet(
# For the Digitizer
    accumulatorType = cms.string("Phase2TrackerDigitizer"),
    hitsProducer = cms.string('g4SimHits'),
    ROUList = cms.vstring(
        'TrackerHitsPixelBarrelLowTof', 
        'TrackerHitsPixelBarrelHighTof', 
        'TrackerHitsPixelEndcapLowTof', 
        'TrackerHitsPixelEndcapHighTof'),
    GeometryType = cms.string('idealForDigi'),
    isOTreadoutAnalog = cms.bool(False),#set this to true if you want analog readout for OT
# Common for Algos
    usePseudoPixel3DAlgo = cms.bool(False),
    premixStage1 = cms.bool(False),
    AlgorithmCommon = cms.PSet(
      DeltaProductionCut = cms.double(0.03),
      makeDigiSimLinks = cms.untracked.bool(True),
    ),
# Specific parameters
#Pixel Digitizer Algorithm
    PixelDigitizerAlgorithm   = PixelDigitizerAlgorithmCommon.clone(
       UseReweighting = cms.bool(False), # will be True for realistic simulations
    ),
#Pixel-3D Digitizer Algorithm
    Pixel3DDigitizerAlgorithm = PixelDigitizerAlgorithmCommon.clone(
        SigmaCoeff = cms.double(1.80),
        NPColumnRadius = cms.double(4.0),
        OhmicColumnRadius = cms.double(4.0),
        NPColumnGap = cms.double(46.0),
        UseReweighting = cms.bool(False),  # will be True for realistic simulations
    ),
#Pixel in PS Module
    PSPDigitizerAlgorithm = cms.PSet(
      ElectronPerAdc = cms.double(135.0),
      ReadoutNoiseInElec = cms.double(-99.9),       # not used at the moment
      ThresholdInElectrons_Barrel = cms.double(6300.), #(0.4 MIP = 0.4 * 16000 e)
      ThresholdInElectrons_Endcap = cms.double(6300.), #(0.4 MIP = 0.4 * 16000 e) 
      AddThresholdSmearing = cms.bool(False),
      ThresholdSmearing_Barrel = cms.double(630.0),
      ThresholdSmearing_Endcap = cms.double(630.0),
      HIPThresholdInElectrons_Barrel = cms.double(1.0e10), # very high value to avoid Over threshold bit
      HIPThresholdInElectrons_Endcap = cms.double(1.0e10), # very high value to avoid Over threshold bit
      NoiseInElectrons = cms.double(200),	         # 30% of the readout noise (should be changed in future)
      Phase2ReadoutMode = cms.int32(0), # Flag to decide Readout Mode :Digital(0) or Analog (linear TDR (-1)), dual slope with slope parameters (+1,+2,+3,+4) with threshold subtraction
      AdcFullScale = cms.int32(255),
      TofUpperCut = cms.double(12.5),
      TofLowerCut = cms.double(-12.5),
      AddNoisyPixels = cms.bool(True),
      Alpha2Order = cms.bool(True),			#D.B.: second order effect, does not switch off magnetic field as described
      AddNoise = cms.bool(True),
      AddXTalk = cms.bool(True),			#D.B.
      InterstripCoupling = cms.double(0.05),	#D.B.
      SigmaZero = cms.double(0.00037),  		#D.B.: 3.7um spread for 300um-thick sensor, renormalized in digitizerAlgo
      SigmaCoeff = cms.double(1.80),  		#D.B.: to be confirmed with simulations in CMSSW_6.X
      ClusterWidth = cms.double(3),		#D.B.: this is used as number of sigmas for charge collection (3=+-3sigmas)
      LorentzAngle_DB = cms.bool(True),			
      TanLorentzAnglePerTesla_Endcap = cms.double(0.07),
      TanLorentzAnglePerTesla_Barrel = cms.double(0.07),
      KillModules = cms.bool(False),
      DeadModules_DB = cms.bool(False),
      DeadModules = cms.VPSet(),
      AddInefficiency = cms.bool(False),
      Inefficiency_DB = cms.bool(False),				
      EfficiencyFactors_Barrel = cms.vdouble(0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999 ),
      EfficiencyFactors_Endcap = cms.vdouble(0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 
      0.999, 0.999 ),#Efficiencies kept as Side2Disk1,Side1Disk1 and so on
      CellsToKill = cms.VPSet(),
      BiasRailInefficiencyFlag = cms.int32(1), # Flag to decide BiasRail inefficiency : no inefficency(0) : inefficiency with optimistic(AND) scenario(1) : inefficiency with pessimistic(OR) scenario(2)
      UseReweighting = cms.bool(False),
    ),
#Strip in PS module
    PSSDigitizerAlgorithm = cms.PSet(
      ElectronPerAdc = cms.double(135.0),
#D.B.:the noise should be a function of strip capacitance, roughly: ReadoutNoiseInElec=500+(64*Cdet[pF]) ~= 500+(64*1.5[cm])
      ReadoutNoiseInElec = cms.double(-99.9),       # not used at the moment
      ThresholdInElectrons_Barrel = cms.double(4800.), #(0.4 MIP = 0.4 * 16000 e)
      ThresholdInElectrons_Endcap = cms.double(4800.), #(0.4 MIP = 0.4 * 16000 e)
      AddThresholdSmearing = cms.bool(False),
      ThresholdSmearing_Barrel = cms.double(480.0),
      ThresholdSmearing_Endcap = cms.double(480.0),
      HIPThresholdInElectrons_Barrel = cms.double(21000.), # 1.4 MIP considered as HIP
      HIPThresholdInElectrons_Endcap = cms.double(21000.), # 1.4 MIP considered as HIP 
      NoiseInElectrons = cms.double(1010), # threshold = 4800e, noise=4800e/4.75=1010 (4.75 sigma=>occupancy =1e-6)
      Phase2ReadoutMode = cms.int32(0), # Flag to decide Readout Mode :Digital(0) or Analog (linear TDR (-1)), dual slope with slope parameters (+1,+2,+3,+4) with threshold subtraction
      AdcFullScale = cms.int32(255),
      TofUpperCut = cms.double(12.5),
      TofLowerCut = cms.double(-12.5),
      AddNoisyPixels = cms.bool(True),
      Alpha2Order = cms.bool(True),			#D.B.: second order effect, does not switch off magnetic field as described
      AddNoise = cms.bool(True),
      AddXTalk = cms.bool(True),			#D.B.
      InterstripCoupling = cms.double(0.05),	#D.B.
      SigmaZero = cms.double(0.00037),  		#D.B.: 3.7um spread for 300um-thick sensor, renormalized in digitizerAlgo
      SigmaCoeff = cms.double(1.80),  		#D.B.: to be confirmed with simulations in CMSSW_6.X
      ClusterWidth = cms.double(3),		#D.B.: this is used as number of sigmas for charge collection (3=+-3sigmas)
      LorentzAngle_DB = cms.bool(True),			
      TanLorentzAnglePerTesla_Endcap = cms.double(0.07),
      TanLorentzAnglePerTesla_Barrel = cms.double(0.07),
      KillModules = cms.bool(False),
      DeadModules_DB = cms.bool(False),
      DeadModules = cms.VPSet(),
      AddInefficiency = cms.bool(False),
      Inefficiency_DB = cms.bool(False),				
      EfficiencyFactors_Barrel = cms.vdouble(0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999 ),
      EfficiencyFactors_Endcap = cms.vdouble(0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 
      0.999, 0.999 ),#Efficiencies kept as Side2Disk1,Side1Disk1 and so on
      CellsToKill = cms.VPSet(),
      UseReweighting = cms.bool(False),
    ),
#Two Strip Module
    SSDigitizerAlgorithm = cms.PSet(
      ElectronPerAdc = cms.double(135.0),
#D.B.:the noise should be a function of strip capacitance, roughly: ReadoutNoiseInElec=500+(64*Cdet[pF]) ~= 500+(64*1.5[cm])
      ReadoutNoiseInElec = cms.double(-99.9),       # not used at the moment
      ThresholdInElectrons_Barrel = cms.double(6000.), 
      ThresholdInElectrons_Endcap = cms.double(6000.),
      AddThresholdSmearing = cms.bool(False),    
      ThresholdSmearing_Barrel = cms.double(600.0),
      ThresholdSmearing_Endcap = cms.double(600.0),
      HIPThresholdInElectrons_Barrel = cms.double(1.0e10), # very high value to avoid Over threshold bit
      HIPThresholdInElectrons_Endcap = cms.double(1.0e10), # very high value to avoid Over threshold bit
      NoiseInElectrons = cms.double(1263), # threshold = 6000e, noise=6000e/4.75=1263e (4.75 sigma=>occupancy =1e-6)
      Phase2ReadoutMode = cms.int32(0), # Flag to decide Readout Mode :Digital(0) or Analog (linear TDR (-1)), dual slope with slope parameters (+1,+2,+3,+4) with threshold subtraction
      AdcFullScale = cms.int32(255),
      TofUpperCut = cms.double(12.5),
      TofLowerCut = cms.double(-12.5),
      AddNoisyPixels = cms.bool(True),
      Alpha2Order = cms.bool(True),			#D.B.: second order effect, does not switch off magnetic field as described
      AddNoise = cms.bool(True),
      AddXTalk = cms.bool(True),			#D.B.
      InterstripCoupling = cms.double(0.05),	#D.B.
      SigmaZero = cms.double(0.00037),  		#D.B.: 3.7um spread for 300um-thick sensor, renormalized in digitizerAlgo
      SigmaCoeff = cms.double(1.80),  		#D.B.: to be confirmed with simulations in CMSSW_6.X
      ClusterWidth = cms.double(3),		#D.B.: this is used as number of sigmas for charge collection (3=+-3sigmas)
      LorentzAngle_DB = cms.bool(True),			
      TanLorentzAnglePerTesla_Endcap = cms.double(0.07),
      TanLorentzAnglePerTesla_Barrel = cms.double(0.07),
      KillModules = cms.bool(False),
      DeadModules_DB = cms.bool(False),
      DeadModules = cms.VPSet(),
      AddInefficiency = cms.bool(False),
      Inefficiency_DB = cms.bool(False),				
      EfficiencyFactors_Barrel = cms.vdouble(0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999 ),
      EfficiencyFactors_Endcap = cms.vdouble(0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 
      0.999, 0.999 ),#Efficiencies kept as Side2Disk1,Side1Disk1 and so on
      CellsToKill = cms.VPSet(),
      HitDetectionMode = cms.int32(0),  # (0/1/2/3/4 => SquareWindow/SampledMode/LatchedMode/SampledOrLachedMode/HIPFindingMode)
      PulseShapeParameters = cms.vdouble(-3.0, 16.043703, 99.999857, 40.571650, 2.0, 1.2459094),
      CBCDeadTime = cms.double(0.0), # (2.7 ns deadtime in latched mode)
      UseReweighting = cms.bool(False),
    )
)

# For premixing stage1
# - add noise as by default
# - do not add noisy pixels (to be done in stage2)
# - do not apply inefficiency (to be done in stage2)
# - disable threshold smearing
# - disable x-talk simulatiom
#
# For both Inner and Outer tracker
# - force analog readout to get Full Charge  ADCs 
# - for Inner Tracker Dual Slope signal scaling NOT used here to avoid any singal loss.
#   At step 2 Dual Slope signal scaling is used as default. To keep the full precision
#   ADCFull scaling is also changed to 255 for Inner Tracker
# 
# - 
# NOTE: It is currently assumed that all sub-digitizers have the same ElectronPerAdc.
from Configuration.ProcessModifiers.premix_stage1_cff import premix_stage1
_premixStage1ModifyDict = dict(
    premixStage1 = True,
    PixelDigitizerAlgorithm = dict(
        AddNoisyPixels = False,
        AddInefficiency = False,
        AddThresholdSmearing = False,
        AddXTalk = False,
        Phase2ReadoutMode = -1,
        AdcFullScale = 255,
    ),
    Pixel3DDigitizerAlgorithm = dict(
        AddNoisyPixels = False,
        AddInefficiency = False,
        AddThresholdSmearing = False,
        AddXTalk = False,
        Phase2ReadoutMode = -1,
        AdcFullScale = 255,
    ),
    PSPDigitizerAlgorithm = dict(
        AddNoisyPixels = False,
        AddInefficiency = False,
        AddThresholdSmearing = False,
        AddXTalk = False,
        Phase2ReadoutMode = -1,
    ),
    PSSDigitizerAlgorithm = dict(
        AddNoisyPixels = False,
        AddInefficiency = False,
        AddThresholdSmearing = False,
        AddXTalk = False,
        Phase2ReadoutMode = -1,
    ),
    SSDigitizerAlgorithm = dict(
        AddNoisyPixels = False,
        AddInefficiency = False,
        AddThresholdSmearing = False,
        AddXTalk = False,
        Phase2ReadoutMode = -1,
    ),
)

premix_stage1.toModify(phase2TrackerDigitizer, **_premixStage1ModifyDict)

from Configuration.ProcessModifiers.enableXTalkInPhase2Pixel_cff import enableXTalkInPhase2Pixel
_enableXTalkInPhase2PixelModifyDict = dict( 
    PixelDigitizerAlgorithm = dict(
        AddXTalk = True, 
        Odd_row_interchannelCoupling_next_row = 0.00,
        Even_row_interchannelCoupling_next_row = 0.06
        )
)

enableXTalkInPhase2Pixel.toModify(phase2TrackerDigitizer, **_enableXTalkInPhase2PixelModifyDict)



