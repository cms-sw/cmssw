echo "===================> Step1: executing EDProducer (TPGSimulation) "
   cmsRun WriteTPG_cfg.py
# create histos
echo "===================> Step2: executing EDAnalyser (readTPG_cfg.py) to create histos in histos.root "
    cmsRun ReadTPG_cfg.py
   
echo "===================> Step3: rereading histos with root"
root -b -q HistoCompare.C
