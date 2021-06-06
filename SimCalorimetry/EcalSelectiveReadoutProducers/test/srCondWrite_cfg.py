import FWCore.ParameterSet.Config as cms

process = cms.Process("ProcessOne")
process.load("CondCore.DBCommon.CondDBCommon_cfi")
process.CondDBCommon.connect = 'sqlite_file:EcalSRSettings_beam2015_option1_v00_mc.db'
process.CondDBCommon.DBParameters.authenticationPath = '/afs/cern.ch/cms/DB/conddb'

process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True)
    ),
    debugModules = cms.untracked.vstring('*')
)

process.source = cms.Source("EmptyIOVSource",
                                firstValue = cms.uint64(1),
                                lastValue = cms.uint64(1),
                                timetype = cms.string('runnumber'),
                                interval = cms.uint64(1)
                            )

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    process.CondDBCommon,
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('EcalSRSettingsRcd'),
        tag = cms.string('EcalSRSettings_beam2015_option1_v00_mc')
    ))
)

process.writeInDB = cms.EDAnalyzer("EcalSRCondTools",
#   mode = cms.string("combine_config"), #Gets config from EcalSRCondTools module parameters,
                                         #use values from onlineSrpConfigFile for the configuration
                                         #not defined as module parameters. Values from module parameters
                                         #take the precedence.
                                   
    mode = cms.string("python_config"), #configuration read from EcalSRCondTools module parameters (e.g. to produce MC config.)

#    mode = cms.string("online_config"), #import online SRP config from onlineSrpConfigFile file and bxGlobalOffset,
                                         #automaticSrpSelect, automaticMasks parameters


    onlineSrpConfigFile = cms.string("srp.cfg"),
                                 
    # Neighbour eta range, neighborhood: (2*deltaEta+1)*(2*deltaPhi+1)
    deltaEta = cms.int32(1),

    # Neighbouring eta range, neighborhood: (2*deltaEta+1)*(2*deltaPhi+1)
    deltaPhi = cms.int32(1),

    # Index of time sample (staring from 1) the first DCC weights is implied
    ecalDccZs1stSample = cms.int32(2),

    # ADC to GeV conversion factor used in ZS filter for EB
    ebDccAdcToGeV = cms.double(0.035),

    # ADC to GeV conversion factor used in ZS filter for EE
    eeDccAdcToGeV = cms.double(0.06),

    #DCC ZS FIR weights: weights are rounded in such way that in Hw
    #representation (weigth*1024 rounded to nearest integer) the sum is null:
    dccNormalizedWeights = cms.vdouble(-0.374, -0.374, -0.3629, 0.2721, 0.4681, 
        0.3707),

    # Switch to use a symetric zero suppression (cut on absolute value). For
    # studies only, for time being it is not supported by the hardware.
    symetricZS = cms.bool(False),

    # ZS energy threshold in GeV to apply to low interest channels of barrel
    srpBarrelLowInterestChannelZS = cms.double(0.1),

    # ZS energy threshold in GeV to apply to low interest channels of endcap
    srpEndcapLowInterestChannelZS = cms.double(0.3),

    # ZS energy threshold in GeV to apply to high interest channels of barrel
    srpBarrelHighInterestChannelZS = cms.double(-1.e9),

    # ZS energy threshold in GeV to apply to high interest channels of endcap
    srpEndcapHighInterestChannelZS = cms.double(-1.e9),

    #switch to run w/o trigger primitive. For debug use only
    trigPrimBypass = cms.bool(False),

    # Mode selection for "Trig bypass" mode
    # 0: TT thresholds applied on sum of crystal Et's
    # 1: TT thresholds applies on compressed Et from Trigger primitive
    # @ee trigPrimByPass_ switch
    trigPrimBypassMode = cms.int32(0),
                              
    #for debug mode only:
    trigPrimBypassLTH = cms.double(1.0),

    #for debug mode only:
    trigPrimBypassHTH = cms.double(1.0),

    #for debug mode only
    trigPrimBypassWithPeakFinder = cms.bool(True),
                              
    #Trigger Tower Flag to use when a flag is not found from the input
    #Trigger Primitive collection. Must be one of the following values:
    # 0: low interest, 1: mid interest, 3: high interest
    # 4: forced low interest, 5: forced mid interest, 7: forced high interest
    defaultTtf = cms.int32(4),

    # SR->action flag map
    actions = cms.vint32(1, 3, 3, 3, 5, 7, 7, 7),

    #Bx offset common to every SRP card. used in both write mdes
    #called SRP0BUNCHADJUSTPOSITION in online configuration database
    bxGlobalOffset = cms.int32(3447),

    #Switch for selecion of SRP board to controls base on
    #the list of ECAL FEDs included in the run (online specific parameter) 
    automaticSrpSelect = cms.int32(1),

    #Switch for automatic masking TCC input channels of SRP boards
    #if the correcponding ECAL FED is excluded from the run (online specific parameter)
    automaticMasks = cms.int32(1)
)


## Changes settings to 2009 and 2010 beam ones:
##
## DCC ZS FIR weights.
#process.writeInDB.dccNormalizedWeights = cms.vdouble(-1.1865, 0.0195, 0.2900, 0.3477, 0.3008, 0.2266)
#
## Index of time sample (starting from 1) the first DCC weights is implied
#process.writeInDB.ecalDccZs1stSample = cms.int32(3)
#
## ZS energy threshold in GeV to apply to low interest channels of barrel
#process.writeInDB.ebDccAdcToGeV = cms.double(0.035)
#process.writeInDB.srpBarrelLowInterestChannelZS = cms.double(2.25*0.035)
#
## ZS energy threshold in GeV to apply to low interest channels of endcap
#process.writeInDB.eeDccAdcToGeV = cms.double(0.06)
#process.writeInDB.srpEndcapLowInterestChannelZS = cms.double(3.75*0.06)


## Changes settings to 2011 beam ones:
## Index of time sample (starting from 1) the first DCC weights is implied
#process.writeInDB.ecalDccZs1stSample = cms.int32(2)
#
## ZS energy threshold in GeV to apply to low interest channels of barrel
#process.writeInDB.ebDccAdcToGeV = cms.double(0.035)
#process.writeInDB.srpBarrelLowInterestChannelZS = cms.double(2.25*0.035)
#
## ZS energy threshold in GeV to apply to low interest channels of endcap
#process.writeInDB.eeDccAdcToGeV = cms.double(0.06)
#process.writeInDB.srpEndcapLowInterestChannelZS = cms.double(3.75*0.06)


## Changes settings to 2012 beam ones:
## Index of time sample (starting from 1) the first DCC weights is implied
#process.writeInDB.ecalDccZs1stSample = cms.int32(2)
#
## ZS energy threshold in GeV to apply to low interest channels of barrel
#process.writeInDB.ebDccAdcToGeV = cms.double(0.035)
#process.writeInDB.srpBarrelLowInterestChannelZS = cms.double(2.75*0.035)
#
## ZS energy threshold in GeV to apply to low interest channels of endcap
#process.writeInDB.eeDccAdcToGeV = cms.double(0.06)
#process.writeInDB.srpEndcapLowInterestChannelZS = cms.double(6*0.06)

# Changes settings to 2015 beam ones:
# Index of time sample (starting from 1) the first DCC weights is implied
process.writeInDB.ecalDccZs1stSample = cms.int32(3)

# ZS energy threshold in GeV to apply to low interest channels of barrel
process.writeInDB.ebDccAdcToGeV = cms.double(0.035)
process.writeInDB.srpBarrelLowInterestChannelZS = cms.double(2.75*0.035)

# ZS energy threshold in GeV to apply to low interest channels of endcap
process.writeInDB.eeDccAdcToGeV = cms.double(0.06)
process.writeInDB.srpEndcapLowInterestChannelZS = cms.double(6*0.06)

process.p = cms.Path(process.writeInDB)
