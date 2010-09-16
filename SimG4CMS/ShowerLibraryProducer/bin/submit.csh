#!/bin/csh
#
set cfg_in = "SimG4CMS/ShowerLibraryProducer/test/python/runCastorSLMaker_cfg.py"
set cfg_out = runCastorSLMaker_cfg
set SL_merged
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
      -e '/fileName/ s/sim_.*root/\/tmp\/'$simfile$sbin'ppONtrkproj.root/' \
      -e 's/cms.EDProducer.*/cms.EDProducer("FlatRandomEGunProducer",/' \
      -e '/MinE / {s/#//; s/(.*)/('$emin')/}'\
      -e '/MaxE / {s/#//; s/(.*)/('$emax')/}'\
      -e '/^ *Energybins/ s/^ */&#/' \
#      -e 's/^ *M..E = /#&/' \
#      -e '/Energybins/ s/#//; s/(.*)/('$bin')/' \
      -e '/EventNtupleFileName/ s/SL.*_E.*GeV/'$evtfile$sbin'GeV/' \
      -e '/EventNtupleFileName/ s/_[0-9]*events/_'$nevt'events/' \
      -e '/StepNtupleFileName/ s/_E.*GeV/_E'$sbin'GeV/'\
      -e '/nemEvents/ s/=.*$/= cms.int32('$nevtem'),/' \
      -e '/^ *SLemEnergyBins/ s/(.*)/('$emin')/' \
      -e '/nhadEvents/ s/=.*$/= cms.int32('$nevthad'),/' \
      -e '/^ *SLhadEnergyBins/ s/(.*)/('$emin')/' $cfg_in >! ${cfg_out}_E${sbin}.py
   bsub -q 2nd <<EOF
#!/bin/csh
set SL_HOME=/afs/cern.ch/user/m/mundim/scratch0/CMSSW_3_7_0_pre5/src
cd \$SL_HOME
cmsenv
cmsRun ${cfg_out}_E${sbin}.py
#rm -f ${simfile}${sbin}ppONtrkproj.root
EOF
set input_file = `grep EventNtupleFileName ${cfg_out}_E${sbin}.py|sed -e's/^.*SL/SL/' -e's/.root.*/.root/'`
echo $input_file' \' >> $merge_script
end
set SL_merged=$SL_merged$emax
echo $input_file|sed -e's/_E.*GeV/_E'$SL_merged'GeV/' >> $merge_script
chmod +x $merge_script
