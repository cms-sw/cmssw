#! /bin/csh
#############################################################################
#                       Test suite for MixingModule                         #
#############################################################################
# part1: detailed testing of MM functionality:
#--------------------------------------------
# loops over bunchcrossings -5 , 3
# for each bunchcrossing the MixingModule is executed, configured with this bunchcrossing only
# then histograms are created and stored in file histos.root
# they may be looked at by executing testsuite.C interactively in root
# or are compared histo by histo
# part2: global physics validation
#---------------------------------

@ bcrstart=-5
@ bcrend=3
@ i=$bcrstart

while ( $i <= $bcrend )
   echo "===================> Step1: executing EDProducer (MixingModule) for bcr $i"
# execute Mixing Module
   /bin/rm /tmp/testsuite1_$i.cfg  >& /dev/null
   sed "s/xxx/$i/" testsuite1.cfg >/tmp/testsuite1_$i.cfg
   cmsRun --parameter-set /tmp/testsuite1_$i.cfg
# create histos
   echo "===================> Step2: executing EDAnalyser (TestSuite) to create histos for bcr $i"
    /bin/rm /tmp/testsuite2_$i.cfg  > &/dev/null
    sed "s/xxx/$i/" testsuite2.cfg | sed "s/bcrs/$bcrstart/" | sed "s/bcre/$bcrend/" >/tmp/testsuite2_$i.cfg
    cmsRun --parameter-set /tmp/testsuite2_$i.cfg
####    cp  histos.root ../data/MMValHistos_$i.root  # for test preparation only!
    echo "===================> Step2a: histogram comparison"
    root -b -p -q DoCompare.C\(\"histos\",\"../data/MMValHistos_$i\"\)
    @ i++
end
   
    echo "===================> Step3: Global comparisons "
    cmsRun globalTest1.cfg  # execute mixing
    cmsRun globalTest2.cfg  # look at results
####    cp  GlobalHistos.root ../data/GlobalHistos.root  # for test preparation only!
    root -b -p -q DoCompare.C\(\"GlobalHistos\",\"../data/GlobalHistos\"\)

    echo "===================> MM Validation finished "
