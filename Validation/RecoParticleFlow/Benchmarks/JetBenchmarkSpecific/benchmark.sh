#!/bin/sh

simu=Fast
type=aod

for ((job=0;job<=8;job++));
    do
    case $job in
    0)
	name=QCDDiJet_20_30
	;;
    1)
	name=QCDDiJet_30_50
	;;
    2)
	name=QCDDiJet_50_80
	;;
    3)
	name=QCDDiJet_80_120
	;;
    4)
	name=QCDDiJet_120_160
	;;
    5)
	name=QCDDiJet_160_250
	;;
    6)
	name=QCDDiJet_250_350
	;;
    7)
	name=QCDDiJet_350_500
	;;
    8)
	name=QCDDiJet_500_700
	;;
    esac
	
    filename="..\/..\/test\/"${type}"_"${name}"_"${simu}".root"
    rootname="JetBenchmark_"${name}"_"${simu}".root"
    sed -e "s/==AOD==/${filename}/" benchmark_cfg.py > tmp_cfg.py
    echo "Analysing " $filename "..."
    cmsRun tmp_cfg.py
    rm tmp_cfg.py
    mv JetBenchmark.root $rootname

done

