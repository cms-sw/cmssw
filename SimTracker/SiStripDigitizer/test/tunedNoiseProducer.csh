#! /bin/csh

### check out: V03-07-07      CalibTracker/SiStripESProducers
### and change EDFilter into EDAnalyzer, and electronPerAdc into electronPerAdcDec

### parameters to be set by hand:
set output = dbfile

set scaleTIB = 1.28
set scaleTID = 1.28
set scaleTOB = 1.09
set scaleTEC = 1.13

set Slope = 51.0
set Quote = 630.0

### derived parameters:

set SlopeTIB = `echo "scale=1; ${Slope}/${scaleTIB}" | bc`
set SlopeTID = `echo "scale=1; ${Slope}/${scaleTID}" | bc`
set SlopeTOB = `echo "scale=1; ${Slope}/${scaleTOB}" | bc`
set SlopeTEC = `echo "scale=1; ${Slope}/${scaleTEC}" | bc`

set QuoteTIB = `echo "scale=1; ${Quote}/${scaleTIB}" | bc`
set QuoteTID = `echo "scale=1; ${Quote}/${scaleTID}" | bc`
set QuoteTOB = `echo "scale=1; ${Quote}/${scaleTOB}" | bc`
set QuoteTEC = `echo "scale=1; ${Quote}/${scaleTEC}" | bc`

echo TIB $SlopeTIB $QuoteTIB
echo TID $SlopeTID $QuoteTID
echo TOB $SlopeTOB $QuoteTOB
echo TEC $SlopeTEC $QuoteTEC

### execution:


echo creating files ${output}.db and ${output}_cfg.py

setenv SCRAM_ARCH slc5_ia32_gcc434
eval `scramv1 runtime -csh`

if (-e ${output}.db) rm ${output}.db
$CMSSW_RELEASE_BASE/src/CondTools/SiStrip/scripts/CreatingTables.sh sqlite_file:${output}.db a a

if (-e ${output}_cfg.py) rm ${output}_cfg.py
cat <<EOF > ${output}_cfg.py
import FWCore.ParameterSet.Config as cms

process = cms.Process("NOISEGAINBUILDER")
process.MessageLogger = cms.Service("MessageLogger",
    threshold = cms.untracked.string('INFO'),
    destinations = cms.untracked.vstring('cout')
)

process.source = cms.Source("EmptySource",
    numberEventsInRun = cms.untracked.uint32(1),
    firstRun = cms.untracked.uint32(128408)
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

process.poolDBESSource = cms.ESSource("PoolDBESSource",
   BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
   DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(2),
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    # connect = cms.string('oracle://cms_orcoff_prod/CMS_COND_31X_STRIP'),
    connect = cms.string('sqlite_file:dbfile_gainFromData.db'),
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('SiStripApvGainRcd'),
        # tag = cms.string('SiStripApvGain_GR10_v1_hlt')
        tag = cms.string('SiStripApvGain_default')
        #tag = cms.string('SiStripApvGain_Ideal_31X')
    ))
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
    DBParameters = cms.PSet(
        authenticationPath = cms.untracked.string('/afs/cern.ch/cms/DB/conddb')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:dbfile.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('SiStripNoisesRcd'),
        tag = cms.string('SiStripNoiseNormalizedWithIdealGain')
    ))
)

from SimTracker.SiStripDigitizer.SiStripDigi_cfi import *
process.prod = cms.EDAnalyzer("SiStripNoiseNormalizedWithApvGainBuilder",
                            printDebug = cms.untracked.uint32(5),
                            file = cms.untracked.FileInPath('CalibTracker/SiStripCommon/data/SiStripDetInfo.dat'),
                                                                          
                            StripLengthMode = cms.bool(True),

                            #relevant if striplenght mode is chosen
                            # standard value for deconvolution mode is 51. For peak mode 38.8.
                            # standard value for deconvolution mode is 630. For peak mode  414.
                            
                            # TIB
                            NoiseStripLengthSlopeTIB = cms.vdouble(${SlopeTIB}, ${SlopeTIB}, ${SlopeTIB}, ${SlopeTIB}),
                            NoiseStripLengthQuoteTIB = cms.vdouble(${QuoteTIB}, ${QuoteTIB}, ${QuoteTIB}, ${QuoteTIB}),
                            # TID                         
                            NoiseStripLengthSlopeTID = cms.vdouble(${SlopeTID}, ${SlopeTID}, ${SlopeTID}),
                            NoiseStripLengthQuoteTID = cms.vdouble(${QuoteTID}, ${QuoteTID}, ${QuoteTID}),
                            # TOB                         
                            NoiseStripLengthSlopeTOB = cms.vdouble(${SlopeTOB}, ${SlopeTOB}, ${SlopeTOB}, ${SlopeTOB}, ${SlopeTOB}, ${SlopeTOB}),
                            NoiseStripLengthQuoteTOB = cms.vdouble(${QuoteTOB}, ${QuoteTOB}, ${QuoteTOB}, ${QuoteTOB}, ${QuoteTOB}, ${QuoteTOB}),
                            # TEC
                            NoiseStripLengthSlopeTEC = cms.vdouble(${SlopeTEC}, ${SlopeTEC}, ${SlopeTEC}, ${SlopeTEC}, ${SlopeTEC}, ${SlopeTEC}, ${SlopeTEC}),
                            NoiseStripLengthQuoteTEC = cms.vdouble(${QuoteTEC}, ${QuoteTEC}, ${QuoteTEC}, ${QuoteTEC}, ${QuoteTEC}, ${QuoteTEC}, ${QuoteTEC}),
                            
                            #electronPerAdc = cms.double(1.0),
                            electronPerAdc = simSiStripDigis.electronPerAdcDec,

                            #relevant if random mode is chosen
                            # TIB
                            MeanNoiseTIB  = cms.vdouble(4.0, 4.0, 4.0, 4.0),
                            SigmaNoiseTIB = cms.vdouble(0.5, 0.5, 0.5, 0.5),
                            # TID
                            MeanNoiseTID  = cms.vdouble(4.0, 4.0, 4.0),
                            SigmaNoiseTID = cms.vdouble(0.5, 0.5, 0.5),
                            # TOB
                            MeanNoiseTOB  = cms.vdouble(4.0, 4.0, 4.0, 4.0, 4.0, 4.0),
                            SigmaNoiseTOB = cms.vdouble(0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                            # TEC
                            MeanNoiseTEC  = cms.vdouble(4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0),
                            SigmaNoiseTEC = cms.vdouble(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                            
                            MinPositiveNoise = cms.double(0.1)
)

process.p = cms.Path(process.prod)
EOF

cmsRun ${output}_cfg.py
