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
   /bin/rm /tmp/testsuite1_{$i}_cfg.py  >& /dev/null
   sed "s/12345/$i/" testsuite1_cfg.py >/tmp/testsuite1_{$i}_cfg.py
   cmsRun /tmp/testsuite1_{$i}_cfg.py
# create histos
   echo "===================> Step2: executing EDAnalyser (TestSuite) to create histos for bcr $i"
    /bin/rm /tmp/testsuite2_{$i}_cfg.py  > &/dev/null
    sed "s/12345/$i/" testsuite2_cfg.py | sed "s/23456/$bcrstart/" | sed "s/34567/$bcrend/" >/tmp/testsuite2_{$i}_cfg.py
    cmsRun /tmp/testsuite2_{$i}_cfg.py
####    cp  histos.root ../data/MMValHistos_$i.root  # for test preparation only!
    echo "===================> Step2a: histogram comparison"
    root -b -p -q DoCompare.C\(\"histos\",\"../data/MMValHistos_$i\"\)
    @ i++
end
   
    echo "===================> Step3: Global comparisons "
    cmsRun globalTest1_cfg.py  # execute mixing
    cmsRun globalTest2_cfg.py  # look at results
####    cp  GlobalHistos.root ../data/GlobalHistos.root  # for test preparation only!
    root -b -p -q DoCompare.C\(\"GlobalHistos\",\"../data/GlobalHistos\"\)

    echo "===================> MM Validation finished "
