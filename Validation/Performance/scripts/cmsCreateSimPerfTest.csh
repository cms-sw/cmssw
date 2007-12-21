#!/bin/csh
set verbose

#Adding the environment setting to fix environment problem with cmsRelvalreport
eval `scramv1 runtime -csh`

#Running TimingReport, TimeReport, SimpleMemoryCheck, EdmSize on all 7 candles
#With 50 events each
foreach i (HiggsZZ4LM190 MinBias SingleElectronE1000 SingleMuMinusPt1000 SinglePiMinusPt1000 TTbar ZPrimeJJM700)
mkdir ${i}_TimeSize
cd ${i}_TimeSize
cmsSimulationCandles.pl 50 $i G4no 0123
cmsRelvalreport.py -i SimulationCandles_${CMSSW_VERSION}.txt -t perfreport_tmp -R -P >& ${i}.log
cd ..
end

#Running IgProfPerf, IgProfMem (TOTAL, LIVE, ANALYSE) on 5 ZPrimeJJ events
mkdir ZPrimeJJM700_IgProf
cd ZPrimeJJM700_IgProf
cmsSimulationCandles.pl 5 ZPrimeJJM700 G4no 4567
cmsRelvalreport.py -i SimulationCandles_${CMSSW_VERSION}.txt -t perfreport_tmp -R -P >& ZPrimeJJM700.log
cd ..

#Running ValgrindFCE callgrind and memcheck on 1 ZPrimeJJ event (DIGI only)
mkdir ZPrimeJJM700_Valgrind
cd ZPrimeJJM700_Valgrind
cmsSimulationCandles.pl 1 ZPrimeJJM700 G4no 89
grep -v sim SimulationCandles_${CMSSW_VERSION}.txt >tmp; mv tmp SimulationCandles_${CMSSW_VERSION}.txt
cmsRun ZPrimeJJM700_sim.cfg >& ZPrimeJJM700_sim.log
cmsRelvalreport.py -i SimulationCandles_${CMSSW_VERSION}.txt -t perfreport_tmp -R -P >& ZPrimeJJM700_digi.log
cd ..

#Running ValgrindFCE callgrind and memcheck on 1 SingleMuMinus event (SIM only)
foreach i (SingleMuMinusPt1000)
mkdir ${i}_Valgrind
cd ${i}_Valgrind
cmsSimulationCandles.pl 1 $i G4no 89
grep -v digi SimulationCandles_${CMSSW_VERSION}.txt >tmp; mv tmp SimulationCandles_${CMSSW_VERSION}.txt
cmsRelvalreport.py -i SimulationCandles_${CMSSW_VERSION}.txt -t perfreport_tmp -R -P >& ${i}.log
cd ..
end

