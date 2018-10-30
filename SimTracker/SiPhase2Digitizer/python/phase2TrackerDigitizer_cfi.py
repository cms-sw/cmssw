import FWCore.ParameterSet.Config as cms

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
    premixStage1 = cms.bool(False),
    AlgorithmCommon = cms.PSet(
      DeltaProductionCut = cms.double(0.03),
      makeDigiSimLinks = cms.untracked.bool(True),
    ),
# Specific parameters
#Pixel Digitizer Algorithm
    PixelDigitizerAlgorithm = cms.PSet(
      ElectronPerAdc = cms.double(600.0),
      ReadoutNoiseInElec = cms.double(0.0),
      ThresholdInElectrons_Barrel = cms.double(1200.0),
      ThresholdInElectrons_Endcap = cms.double(1200.0),
      AddThresholdSmearing = cms.bool(False),
      ThresholdSmearing_Barrel = cms.double(0.0),
      ThresholdSmearing_Endcap = cms.double(0.0),
      HIPThresholdInElectrons_Barrel = cms.double(1.0e10), # very high value to avoid Over threshold bit
      HIPThresholdInElectrons_Endcap = cms.double(1.0e10), # very high value to avoid Over threshold bit
      NoiseInElectrons = cms.double(0.0),
      Phase2ReadoutMode = cms.int32(-1), # Flag to decide Readout Mode :Digital(0) or Analog (linear TDR (-1), dual slope with slope parameters (+1,+2,+3,+4) with threshold subtraction
      AdcFullScale = cms.int32(15),
      TofUpperCut = cms.double(12.5),
      TofLowerCut = cms.double(-12.5),
      AddNoisyPixels = cms.bool(False),
      Alpha2Order = cms.bool(True),			#D.B.: second order effect, does not switch off magnetic field as described
      AddNoise = cms.bool(False),
      AddXTalk = cms.bool(True),			#D.B.
      InterstripCoupling = cms.double(0.05),	#D.B.
      SigmaZero = cms.double(0.00037),  		#D.B.: 3.7um spread for 300um-thick sensor, renormalized in digitizerAlgo
      SigmaCoeff = cms.double(1.80),  		#D.B.: to be confirmed with simulations in CMSSW_6.X
      ClusterWidth = cms.double(3),		#D.B.: this is used as number of sigmas for charge collection (3=+-3sigmas)
      LorentzAngle_DB = cms.bool(False),			
      TanLorentzAnglePerTesla_Endcap = cms.double(0.106),
      TanLorentzAnglePerTesla_Barrel = cms.double(0.106),
      KillModules = cms.bool(False),
      DeadModules_DB = cms.bool(False),
      DeadModules = cms.VPSet(),
      AddInefficiency = cms.bool(False),
      Inefficiency_DB = cms.bool(False),				
      EfficiencyFactors_Barrel = cms.vdouble(0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999 ),
      EfficiencyFactors_Endcap = cms.vdouble(0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 
      0.999, 0.999 ),#Efficiencies kept as Side2Disk1,Side1Disk1 and so on
      CellsToKill = cms.VPSet()
    ),
#Pixel in PS Module
    PSPDigitizerAlgorithm = cms.PSet(
      ElectronPerAdc = cms.double(135.0),
      ReadoutNoiseInElec = cms.double(200.0),#D.B.:Fill readout noise, including all readout chain, relevant for smearing
      ThresholdInElectrons_Barrel = cms.double(6300.), #(0.4 MIP = 0.4 * 16000 e)
      ThresholdInElectrons_Endcap = cms.double(6300.), #(0.4 MIP = 0.4 * 16000 e) 
      AddThresholdSmearing = cms.bool(True),
      ThresholdSmearing_Barrel = cms.double(630.0),
      ThresholdSmearing_Endcap = cms.double(630.0),
      HIPThresholdInElectrons_Barrel = cms.double(1.0e10), # very high value to avoid Over threshold bit
      HIPThresholdInElectrons_Endcap = cms.double(1.0e10), # very high value to avoid Over threshold bit
      NoiseInElectrons = cms.double(200),	         # 30% of the readout noise (should be changed in future)
      Phase2ReadoutMode = cms.int32(0), # Flag to decide Readout Mode :Digital(0) or Analog (linear TDR (-1), dual slope with slope parameters (+1,+2,+3,+4) with threshold subtraction
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
      LorentzAngle_DB = cms.bool(False),			
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
      CellsToKill = cms.VPSet()
    ),
#Strip in PS module
    PSSDigitizerAlgorithm = cms.PSet(
      ElectronPerAdc = cms.double(135.0),
#D.B.:the noise should be a function of strip capacitance, roughly: ReadoutNoiseInElec=500+(64*Cdet[pF]) ~= 500+(64*1.5[cm])
      ReadoutNoiseInElec = cms.double(700.0),#D.B.:Fill readout noise, including all readout chain, relevant for smearing
      ThresholdInElectrons_Barrel = cms.double(6300.), #(0.4 MIP = 0.4 * 16000 e)
      ThresholdInElectrons_Endcap = cms.double(6300.), #(0.4 MIP = 0.4 * 16000 e)
      AddThresholdSmearing = cms.bool(True),
      ThresholdSmearing_Barrel = cms.double(630.0),
      ThresholdSmearing_Endcap = cms.double(630.0),
      HIPThresholdInElectrons_Barrel = cms.double(21000.), # 1.4 MIP considered as HIP
      HIPThresholdInElectrons_Endcap = cms.double(21000.), # 1.4 MIP considered as HIP 
      NoiseInElectrons = cms.double(700),	         # 30% of the readout noise (should be changed in future)
      Phase2ReadoutMode = cms.int32(0), # Flag to decide Readout Mode :Digital(0) or Analog (linear TDR (-1), dual slope with slope parameters (+1,+2,+3,+4) with threshold subtraction
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
      LorentzAngle_DB = cms.bool(False),			
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
      CellsToKill = cms.VPSet()
    ),
#Two Strip Module
    SSDigitizerAlgorithm = cms.PSet(
      ElectronPerAdc = cms.double(135.0),
#D.B.:the noise should be a function of strip capacitance, roughly: ReadoutNoiseInElec=500+(64*Cdet[pF]) ~= 500+(64*1.5[cm])
      ReadoutNoiseInElec = cms.double(1000.0),#D.B.:Fill readout noise, including all readout chain, relevant for smearing
      ThresholdInElectrons_Barrel = cms.double(5800.), #D.B.: this should correspond to a threshold of 530mV    
      ThresholdInElectrons_Endcap = cms.double(5800.),
      AddThresholdSmearing = cms.bool(True),
      ThresholdSmearing_Barrel = cms.double(580.0),#D.B.: changed (~5mV peakToPeak --> 1.76mV rms) (was 210.0)
      ThresholdSmearing_Endcap = cms.double(580.0),#D.B.: changed (~5mV peakToPeak --> 1.76mV rms) (was 245.0)
      HIPThresholdInElectrons_Barrel = cms.double(1.0e10), # very high value to avoid Over threshold bit
      HIPThresholdInElectrons_Endcap = cms.double(1.0e10), # very high value to avoid Over threshold bit
      NoiseInElectrons = cms.double(1000),	         # 30% of the readout noise (should be changed in future)
      Phase2ReadoutMode = cms.int32(0), # Flag to decide Readout Mode :Digital(0) or Analog (linear TDR (-1), dual slope with slope parameters (+1,+2,+3,+4) with threshold subtraction
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
      LorentzAngle_DB = cms.bool(False),			
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
      CellsToKill = cms.VPSet()
    )
)

# For premixing stage1
# - add noise as by default
# - do not add noisy pixels (to be done in stage2)
# - do not apply inefficiency (to be done in stage2)
# - disable threshold smearing
#
# For inner pixel
# - extend the dynamic range of ADCs
#
# For outer tracker
# - force analog readout to get the ADCs
#
# NOTE: It is currently assumed that all sub-digitizers have the same ElectronPerAdc.
from Configuration.ProcessModifiers.premix_stage1_cff import premix_stage1
_premixStage1ModifyDict = dict(
    premixStage1 = True,
    PixelDigitizerAlgorithm = dict(
        AddNoisyPixels = False,
        AddInefficiency = False,
        AddThresholdSmearing = False,
        ElectronPerAdc = phase2TrackerDigitizer.PSPDigitizerAlgorithm.ElectronPerAdc.value(),
        AdcFullScale = phase2TrackerDigitizer.PSPDigitizerAlgorithm.AdcFullScale.value(),
    ),
    PSPDigitizerAlgorithm = dict(
        AddNoisyPixels = False,
        AddInefficiency = False,
        AddThresholdSmearing = False,
    ),
    PSSDigitizerAlgorithm = dict(
        AddNoisyPixels = False,
        AddInefficiency = False,
        AddThresholdSmearing = False,
    ),
    SSDigitizerAlgorithm = dict(
        AddNoisyPixels = False,
        AddInefficiency = False,
        AddThresholdSmearing = False,
    ),
)
premix_stage1.toModify(phase2TrackerDigitizer, **_premixStage1ModifyDict)
