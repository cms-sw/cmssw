This is a short description of the content of the ntuple produced by Phase2PixelNtuple.cc

### GEOMETRY 

* subid = 1: TBPX

  layer  = 1..4 

  ladder = 1..28 depending on the layer

  nRowsInDet =  656 layer 1,2

             = 1320 layer 3,4

  nColsInDet =  442 layer 1,2

                442 layer 3,4
		     

* subid = 2: TFPX, TEPX

  side = 1 (-z), 2 (+z)

  disk = 1..12 (disk = 1..8 TFPX; disk = 9..12 TEPX)

  blade = 1..4 TFPX 

        = 1..5 TEPX

  NB: 'blade' corresponds to a ring

  panel = 1 (?)

  nRowsInDet =  656 blade 1,2

             = 1320 blade 3,4,5

  nColsInDet =  442 blade 1,2

                442 blade 3,4,5

### RECHIT
    
* Local position of the SimHit (approx)

    hx [-0.83,+0.83] cm   1x2 modules

       [-1.67,+1.67] cm   2x2 modules

    hy [-2.24,+2.24] cm    

    q = charge of RecHit in e

    DgCharge = charge of pixel in ke

               to get the ADC count (integer): DgCharge*1000./600. or DgCharge*1000./135. 

               Check the value in the pset, e.g.
               
               SimTracker/SiPhase2Digitizer/python/phase2TrackerDigitizer_cfi.py

               process.mix.digitizers.pixel.PixelDigitizerAlgorithm.ElectronPerAdc = cms.double(135.)

               RecoLocalTracker/SiPixelClusterizer/python/SiPixelClusterizer_cfi.py

               process.siPixelClusters.ElectronPerADCGain=cms.double(135.)
