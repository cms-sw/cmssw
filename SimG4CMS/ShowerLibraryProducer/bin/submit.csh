#!/bin/csh
#
cat > runCastorSLMaker_cfg.template <<EOF
import FWCore.ParameterSet.Config as cms
import time
import datetime

process = cms.Process("CastorShowerLibraryMaker")

process.common_maximum_timex = cms.PSet(
  MaxTrackTime = cms.double(500.0),
  MaxTimeNames = cms.vstring('ZDCRegion','QuadRegion','InterimRegion'),
  MaxTrackTimes = cms.vdouble(2000.0,0.0,0.0)
)

process.common_pgun_particleID = cms.PSet(
        PartID = cms.vint32(11,211)
)

process.load("SimGeneral.HepPDTESSource.pdt_cfi")

process.load("IOMC.EventVertexGenerators.VtxSmearedGauss_cfi")

# process.load("Configuration.StandardSequences.GeometryExtended_cff")
process.load("SimG4CMS.Forward.castorGeometryXML_cfi")
#process.load("Geometry.CMSCommonData.cmsAllGeometryXML_cfi")
#process.load("Geometry.TrackerNumberingBuilder.trackerNumberingGeometry_cfi")

#process.load("Configuration.StandardSequences.SimulationRandomNumberGeneratorSeeds_cff")

#process.load("Configuration.EventContent.EventContent_cff")

process.load("SimG4Core.Application.g4SimHits_cfi")

process.MessageLogger = cms.Service("MessageLogger",
    destinations = cms.untracked.vstring('cout')
#    categories = cms.untracked.vstring('ForwardSim'),
#    debugModules = cms.untracked.vstring('*'),
#    cout = cms.untracked.PSet(
#        threshold = cms.untracked.string('DEBUG'),
#        DEBUG = cms.untracked.PSet(
#            limit = cms.untracked.int32(0)
#        ),
#        ForwardSim = cms.untracked.PSet(
#            limit = cms.untracked.int32(0)
#        )
#    )
)

# Define the random generator seeds based on the current clock
t = datetime.datetime.now()

process.RandomNumberGeneratorService = cms.Service("RandomNumberGeneratorService",
    moduleSeeds = cms.PSet(
        g4SimHits  = cms.untracked.uint32(t.second),         # std: 9784
        VtxSmeared = cms.untracked.uint32(t.microsecond),
        generator  = cms.untracked.uint32(t.second*t.microsecond)     # std: 135799753
    )
    #sourceSeed = cms.untracked.uint32(135799753)         # std: 135799753
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000000000)
)


process.g4SimHits.UseMagneticField = False
#process.g4SimHits.Physics.DefaultCutValue = 10.
process.g4SimHits.Generator.MinEtaCut        = -7.0
process.g4SimHits.Generator.MaxEtaCut        = 7.0
process.g4SimHits.Generator.Verbosity        = 0
process.g4SimHits.CaloTrkProcessing.TestBeam = True
#process.g4SimHits.Physics.type = 'SimG4Core/Physics/FTFP_BERT'

process.CaloSD = cms.PSet(
    DetailedTiming = cms.bool(False),
    EminTrack      = cms.double(1.0),
    Verbosity      = cms.int32(0),
    UseMap         = cms.bool(True),
    CheckHits      = cms.int32(25),
    TmaxHit        = cms.int32(500)  # L.M. testing
)

process.g4SimHits.StackingAction = cms.PSet(
   process.common_heavy_suppression,
   process.common_maximum_timex,        # need to be localy redefined
   TrackNeutrino = cms.bool(False),
   KillHeavy     = cms.bool(False),
   KillDeltaRay  = cms.bool(False),
   SaveFirstLevelSecondary = cms.untracked.bool(True),
   SavePrimaryDecayProductsAndConversionsInTracker = cms.untracked.bool(True),
   SavePrimaryDecayProductsAndConversionsInCalo    = cms.untracked.bool(True),
   SavePrimaryDecayProductsAndConversionsInMuon    = cms.untracked.bool(True)
)

process.g4SimHits.SteppingAction = cms.PSet(
   process.common_maximum_timex, # need to be localy redefined
   KillBeamPipe            = cms.bool(True),
   CriticalEnergyForVacuum = cms.double(2.0),
   CriticalDensity         = cms.double(1e-15),
   EkinNames               = cms.vstring(),
   EkinThresholds          = cms.vdouble(),
   EkinParticles           = cms.vstring(),
   Verbosity               = cms.untracked.int32(0)
)

process.generator = cms.EDProducer("FlatRandomEGunProducer",
    PGunParameters = cms.PSet(
        process.common_pgun_particleID,
        MinEta = cms.double(-6.6),
        MaxEta = cms.double(-5.2),
        MinPhi = cms.double(0.),
        MaxPhi = cms.double(0.7854), # PI/4 = 0.7854
        MinE = cms.double(12.00),
        #MeanE = cms.double(12.00),
        MaxE = cms.double(14.00)
        #Energybins = cms.vdouble(1.,2.,3.,5.,7.,10.,20.,30.,45.,60.,75.,100.,140.,200.,300.,600.,1000.,1500.)
    ),
    AddAntiParticle = cms.bool(False),
    Verbosity = cms.untracked.int32(False)
)

process.g4SimHits.CastorSD.useShowerLibrary = False

process.source = cms.Source("EmptySource")
#process.o1 = cms.OutputModule("PoolOutputModule",
#    fileName = cms.untracked.string('sim_pion_1events-ppON.root')
#)


process.g4SimHits.Watchers = cms.VPSet(cms.PSet(
    type = cms.string('CastorShowerLibraryMaker'),
    CastorShowerLibraryMaker = cms.PSet(
        process.common_pgun_particleID,
        EventNtupleFileName = cms.string('SL_had_E12GeV_eta-6.0phi0.3_1events-ppON.root'),
        Verbosity = cms.int32(0),
        DeActivatePhysicsProcess = cms.bool(False),
        StepNtupleFileName = cms.string('stepNtuple_pion_electron_E12GeV_1event-ppON.root'),
        StepNtupleFlag = cms.int32(0),
        EventNtupleFlag = cms.int32(0),
        # for shower library
        nemEvents       = cms.int32(5),
        SLemEnergyBins  = cms.vdouble(10.),
        SLemEtaBins     = cms.vdouble(-6.6,-6.4,-6.2,-6.0,-5.8,-5.6,-5.4),
        SLemPhiBins     = cms.vdouble(0.,0.15708,0.31416,0.47124,0.62832),
        nhadEvents       = cms.int32(5),
        SLhadEnergyBins  = cms.vdouble(10.),
        #SLhadEnergyBins  = cms.vdouble(1.,2.,3.,5.,7.,10.,20.,30.,45.,60.,75.,100.,140.,200.),
        SLhadEtaBins     = cms.vdouble(-6.6,-6.4,-6.2,-6.0,-5.8,-5.6,-5.4),
        SLhadPhiBins     = cms.vdouble(0.,0.15708,0.31416,0.47124,0.62832),
        SLMaxPhi         = cms.double(0.7854),
        SLMaxEta         = cms.double(-5.2)
    )
))


process.p1 = cms.Path(process.generator*process.VtxSmeared*process.g4SimHits)
#process.outpath = cms.EndPath(process.o1)
EOF
#
#set cfg_in = "SimG4CMS/ShowerLibraryProducer/test/python/runCastorSLMaker_cfg.py"
set cfg_in = "runCastorSLMaker_cfg.template"
set cfg_out = runCastorSLMaker_cfg
set SL_merged
set eta_phi_tag = "7eta-6.6--5.2_5phi0-0.7854"
# create a script for merging
set merge_script="do_merge-`date +%d%b%Y.%H%M%S`.csh"
cat > $merge_script <<EOF
#!/bin/csh
set exec = \$CMSSW_BASE/bin/\$SCRAM_ARCH/CastorShowerLibraryMerger
if (! -e \$exec) then
  echo "\$exec not found. Exiting"
  exit
endif
\$exec \\
EOF
echo "Give the number of events in the phi bin for EM shower"
@ nevtem = "$<"
echo "Give the number of events in the phi bin for HAD shower"
@ nevthad = "$<"
if ($nevtem == 0 && $nevthad == 0) exit

set primId

if ($nevtem > 0 && $nevthad == 0) then
   set simfile = "sim_electron_E";set evtfile = "SL_em_E"
   @ nevt = $nevtem
   set primId = 11
else if ($nevtem == 0 && $nevthad > 0) then
   set simfile = "sim_pion_E";set evtfile = "SL_had_E"
   @ nevt = $nevthad
   set primId = 211
else if ($nevtem > 0 && $nevthad > 0) then
   set simfile = "sim_electron+pion_E";set evtfile = "SL_em+had_E"
   if ($nevtem != $nevthad) then
      echo "Use the same number of events for both showers"
      exit
   endif
   @ nevt = $nevtem
   set primId = "11,211"
endif
echo "Give the energy bin limits (format low_lim,upp_lim; enter to finish)"
set emax
while(1) 
  set bin = "$<"
  if ("$bin" == "") break
  set SL_merged=$SL_merged"`echo $bin|cut -d, -f1`-"
  set sbin=`echo $bin|tr ',' '_'`
  if (("x$emax" != "x")&&("x`echo $bin|cut -d, -f 1`" != "x$emax")) then
     echo "Energy bin not contiguous. Exiting."
     rm -f $SL_merged
     exit
  endif
  set emin=`echo $bin|cut -d, -f 1`
  set emax=`echo $bin|cut -d, -f 2`
  sed -e '/PartID/ s/(.*)/('$primId')/' \
      -e '/fileName/ s/sim_.*root/\/tmp\/'$simfile$sbin$eta_phi_tag'.root/' \
      -e 's/cms.EDProducer.*/cms.EDProducer("FlatRandomEGunProducer",/' \
      -e '/MinE / {s/#//; s/(.*)/('$emin')/}'\
      -e '/MaxE / {s/#//; s/(.*)/('$emax')/}'\
      -e '/^ *Energybins/ s/^ */&#/' \
#      -e 's/^ *M..E = /#&/' \
#      -e '/Energybins/ s/#//; s/(.*)/('$bin')/' \
      -e '/EventNtupleFileName/ s/SL.*_E.*GeV.*.root/'$evtfile$sbin'GeV_'$eta_phi_tag'.root/' \
      -e '/EventNtupleFileName/ s/_[0-9]*events/_'$nevt'events/' \
      -e '/StepNtupleFileName/ s/_E.*GeV.*.root/_E'$sbin'GeV_'$eta_phi_tag'.root/'\
      -e '/nemEvents/ s/=.*$/= cms.int32('$nevtem'),/' \
      -e '/^ *SLemEnergyBins/ s/(.*)/('$emin')/' \
      -e '/nhadEvents/ s/=.*$/= cms.int32('$nevthad'),/' \
      -e '/^ *SLhadEnergyBins/ s/(.*)/('$emin')/' $cfg_in >! ${cfg_out}_E${sbin}.py
   bsub -q 2nw <<EOF
#!/bin/csh
set SL_HOME=$PWD
cd \$SL_HOME
cmsenv
cmsRun ${cfg_out}_E${sbin}.py
EOF
set input_file = `grep EventNtupleFileName ${cfg_out}_E${sbin}.py|sed -e's/^.*SL/SL/' -e's/.root.*/.root/'`
echo $input_file' \' >> $merge_script
end
set SL_merged=$SL_merged$emax
echo $input_file|sed -e's/_E.*GeV/_E'$SL_merged'GeV/' >> $merge_script
chmod +x $merge_script
