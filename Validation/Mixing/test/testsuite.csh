#! /bin/csh
#############################################################################
#                       Test suite for MixingModule                         #
#############################################################################
# loops over bunchcrossings -5 , 3
# for each bunchcrossing the MixingModule is executed, configured with this bunchcrossing only
# then histograms are created and stored in file histos.root
# they may be looked at by executing testsuite.C interactively in root

@ bcrstart=-5
@ bcrend=4
@ i=$bcrstart

while ( $i < $bcrend )
   echo "===================> Step1: executing EDProducer (MixingModule) for $i"
# execute Mixing Module
   /bin/rm /tmp/testsuite1_$i.cfg  >& /dev/null
   sed "s/xxx/$i/" data/testsuite1.cfg >/tmp/testsuite1_$i.cfg
   cmsRun --parameter-set /tmp/testsuite1_$i.cfg
# create histos
   echo "===================> Step2: executing EDAnalyser (TestSuite) to create histos for $i"
    /bin/rm /tmp/testsuite2_$i.cfg  > &/dev/null
    sed "s/xxx/$i/" data/testsuite2.cfg | sed "s/bcrs/$bcrstart/" | sed "s/bcre/$bcrend/" >/tmp/testsuite2_$i.cfg
    cmsRun --parameter-set /tmp/testsuite2_$i.cfg
    @ i++
end
   
echo "===================> Step3: rereading histos with root"
root -b -q testsuite.C

