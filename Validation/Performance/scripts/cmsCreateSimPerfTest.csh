#!/bin/csh
set verbose

#Defining the number of events for each test
set TimeSizeNumberOfEvents=100
set IgProfNumberOfEvents=5
set ValgrindNumberOfEvents=0

#Adding the environment setting to fix environment problem with cmsRelvalreport
eval `scramv1 runtime -csh`

#To fix the pie-chart issues until PerfReport3
source /afs/cern.ch/user/d/dpiparo/w0/perfreport2.1installation/share/perfreport/init_matplotlib.sh

#Adding some info for the logfile
date
echo $HOST
pwd
echo $CMSSW_BASE
echo $CMSSW_VERSION
showtags -r

#Adding an independent benchmark of the machine before running
echo "Initial benchmark">cmsScimark2.log
date >> cmsScimark2.log
echo $HOST >> cmsScimark2.log
repeat 10 cmsScimark2 >> cmsScimark2.log
date >> cmsScimark2.log
echo "Initial benchmark">cmsScimark2_Large.log
date >> cmsScimark2_Large.log
echo $HOST >> cmsScimark2_Large.log
repeat 10 cmsScimark2 -large >> cmsScimark2_Large.log
date > cmsScimark2_Large.log

#Running TimingReport, TimeReport, SimpleMemoryCheck, EdmSize on all 7 candles
#With $TimeSizeNumberOfEvents events each
foreach i (HiggsZZ4LM190 MinBias SingleElectronE1000 SingleMuMinusPt1000 SinglePiMinusPt1000 TTbar ZPrimeJJM700)
mkdir ${i}_TimeSize
cd ${i}_TimeSize
cmsSimulationCandles.pl $TimeSizeNumberOfEvents $i G4no 0123
cmsRelvalreport.py -i SimulationCandles_${CMSSW_VERSION}.txt -t perfreport_tmp -R -P >& ${i}.log
cd ..
end

#Running IgProfPerf, IgProfMem (TOTAL, LIVE, ANALYSE) on $IgProfNumberOfEvents ZPrimeJJ events
mkdir ZPrimeJJM700_IgProf
cd ZPrimeJJM700_IgProf
cmsSimulationCandles.pl $IgProfNumberOfEvents ZPrimeJJM700 G4no 4567
cmsRelvalreport.py -i SimulationCandles_${CMSSW_VERSION}.txt -t perfreport_tmp -R -P >& ZPrimeJJM700.log
cd ..

#Running ValgrindFCE callgrind and memcheck on $ValgrindNumberOfEvents ZPrimeJJ event (DIGI only)
mkdir ZPrimeJJM700_Valgrind
cd ZPrimeJJM700_Valgrind
cmsSimulationCandles.pl $ValgrindNumberOfEvents ZPrimeJJM700 G4no 89
grep -v sim SimulationCandles_${CMSSW_VERSION}.txt >tmp; mv tmp SimulationCandles_${CMSSW_VERSION}.txt
cmsRun ZPrimeJJM700_sim.cfg >& ZPrimeJJM700_sim.log
cmsRelvalreport.py -i SimulationCandles_${CMSSW_VERSION}.txt -t perfreport_tmp -R -P >& ZPrimeJJM700_digi.log
cd ..

#Running ValgrindFCE callgrind and memcheck on $ValgrindNumberOfEvents SingleMuMinus event (SIM only)
foreach i (SingleMuMinusPt1000)
mkdir ${i}_Valgrind
cd ${i}_Valgrind
cmsSimulationCandles.pl $ValgrindNumberOfEvents $i G4no 89
grep -v digi SimulationCandles_${CMSSW_VERSION}.txt >tmp; mv tmp SimulationCandles_${CMSSW_VERSION}.txt
cmsRelvalreport.py -i SimulationCandles_${CMSSW_VERSION}.txt -t perfreport_tmp -R -P >& ${i}.log
cd ..

#Adding an independent benchmark of the machine after running
echo "Final benchmark">>cmsScimark2.log
date >> cmsScimark2.log
echo $HOST >> cmsScimark2.log
repeat 10 cmsScimark2 >> cmsScimark2.log
date >> cmsScimark2.log
echo "Final benchmark">>cmsScimark2_Large.log
date >> cmsScimark2_Large.log
echo $HOST >> cmsScimark2_Large.log
repeat 10 cmsScimark2 -large >> cmsScimark2_Large.log
date >> cmsScimark2_Large.log

end

