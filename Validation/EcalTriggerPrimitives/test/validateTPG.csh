echo "===================> Step1: executing EDProducer (TPGSimulation) "
   cmsRun WriteTpg.cfg
# create histos
echo "===================> Step2: executing EDAnalyser (readTP.cfg) to create histos "
    cmsRun ReadTpg.cfg
   
echo "===================> Step3: rereading histos with root"
root -b -q HistoCompare.C
