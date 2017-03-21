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
    AlgorithmCommon = cms.PSet(
      DeltaProductionCut = cms.double(0.03)
    ),
# Specific parameters
#Pixel Digitizer Algorithm
    PixelDigitizerAlgorithm = cms.PSet(
      makeDigiSimLinks = cms.untracked.bool(True),
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
      DigitalReadout           = cms.bool(False), # Flag to decide analog or digital readout
      AdcFullScale = cms.int32(16),
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
      makeDigiSimLinks = cms.untracked.bool(True),
      ElectronPerAdc = cms.double(135.0),
      ReadoutNoiseInElec = cms.double(1000.0),#D.B.:Fill readout noise, including all readout chain, relevant for smearing
      ThresholdInElectrons_Barrel = cms.double(6300.), #(0.4 MIP = 0.4 * 16000 e)
      ThresholdInElectrons_Endcap = cms.double(6300.), #(0.4 MIP = 0.4 * 16000 e) 
      AddThresholdSmearing = cms.bool(True),
      ThresholdSmearing_Barrel = cms.double(630.0),
      ThresholdSmearing_Endcap = cms.double(630.0),
      HIPThresholdInElectrons_Barrel = cms.double(1.0e10), # very high value to avoid Over threshold bit
      HIPThresholdInElectrons_Endcap = cms.double(1.0e10), # very high value to avoid Over threshold bit
      NoiseInElectrons = cms.double(300),	         # 30% of the readout noise (should be changed in future)
      DigitalReadout           = cms.bool(True), # Flag to decide analog or digital readout 
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
      makeDigiSimLinks = cms.untracked.bool(True),
      ElectronPerAdc = cms.double(135.0),
#D.B.:the noise should be a function of strip capacitance, roughly: ReadoutNoiseInElec=500+(64*Cdet[pF]) ~= 500+(64*1.5[cm])
      ReadoutNoiseInElec = cms.double(1000.0),#D.B.:Fill readout noise, including all readout chain, relevant for smearing
      ThresholdInElectrons_Barrel = cms.double(6300.), #(0.4 MIP = 0.4 * 16000 e)
      ThresholdInElectrons_Endcap = cms.double(6300.), #(0.4 MIP = 0.4 * 16000 e)
      AddThresholdSmearing = cms.bool(True),
      ThresholdSmearing_Barrel = cms.double(630.0),
      ThresholdSmearing_Endcap = cms.double(630.0),
      HIPThresholdInElectrons_Barrel = cms.double(21000.), # 1.4 MIP considered as HIP
      HIPThresholdInElectrons_Endcap = cms.double(21000.), # 1.4 MIP considered as HIP 
      NoiseInElectrons = cms.double(300),	         # 30% of the readout noise (should be changed in future)
      DigitalReadout           = cms.bool(True), # Flag to decide analog or digital readout 
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
      makeDigiSimLinks = cms.untracked.bool(True),
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
      NoiseInElectrons = cms.double(300),	         # 30% of the readout noise (should be changed in future)
      DigitalReadout           = cms.bool(True), # Flag to decide analog or digital readout 
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
