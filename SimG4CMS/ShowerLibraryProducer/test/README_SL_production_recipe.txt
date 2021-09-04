(1) 
Use configuration file runHFShowerTMP_cfg.py in directory
/CMSSW_12_*_*/src/SimG4CMS/ShowerLibraryProducer/test/python.
with EDProducer parameters as listed below (here - for electrons of 100 GeV):
        PartID = cms.vint32(11),
        MinEta = cms.double(4),
        MaxEta = cms.double(4),
        MinPhi = cms.double(-3.14159265359),
        MaxPhi = cms.double(3.14159265359),
        MinE   = cms.double(100),
        MaxE   = cms.double(100)
to perform a full simulation of showers in HF.
Do it for all necessary energy points, separately for pions and electrons.

(2)
Incident angle fixed to eta=4 and with random phi. 
The grid of energies is specified in the file 
/CMSSW_12_*_*/src/SimG4CMS/ShowerLibraryProducer/data/fileList.txt.
Energies from the list should be used in the producer parameter list above (1).

(3)
Simulation is performed using special geometry with a starting point at 1000 mm from HF surface, the path filled with air.
To produce a sample of 10k showers for each energy, the number of simulated events should be somewhat bigger, 
since some part of incident particles interacts in the air. 
The latter showers are excluded from final sample and exactly 10k showers are selected and recorded. 
So in production configuration file, number of event should be set this way: 
    process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(11000)
)

(4)
Once 16*2 samples of showers are produced, they should be moved to the directory 
/CMSSW_12_*_*/src/SimG4CMS/ShowerLibraryProducer/python,
where HF ShowerLibrary can be assembled from these input files using writelibraryfile_cfg.py [*]


-----
[*]
Source file /CMSSW_*_*/src/SimG4CMS/ShowerLibraryProducer/plugins/HcalForwardLibWriter.cc 
Configuration file is /CMSSW_12_*_*/src/SimG4CMS/ShowerLibraryProducer/python/writelibraryfile_cfg.py
