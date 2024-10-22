This is a short description of the content of the ntuple produced by Phase2PixelNtuple.cc

### GEOMETRY 

* `subid` = 1 TBPX                             </br>
	`layer`  = 1..4                         </br>
	`ladder` = 1..28 depending on the layer </br>
	`nRowsInDet` =  656 for `layer` = 1,2   </br>
	`nRowsInDet` = 1320 for `layer` = 3,4   </br>
	`nColsInDet` = 442 for `layer` = 1,2    </br>
	`nColsInDet` = 442 for `layer` = 3,4    
				 
* `subid` = 2: TFPX and TEPX </br>
        `side` = 1 (-z), 2 (+z)  </br>
        `disk` = 1..12 (disk = 1..8 for TFPX; disk = 9..12 for TEPX) </br>
        `blade` = 1..4 TFPX     </br>
        `blade` = 1..5 TEPX     </br>
        NB: the variable named `blade` corresponds to a phase2 ring </br>
        `panel` = 1 (always?)                   </br>
        `nRowsInDet` =  656 for `blade` = 1,2   </br>
        `nRowsInDet` = 1320 for `blade` = 3,4,5 </br>
        `nColsInDet` =  442 for `blade` = 1,2   </br>
        `nColsInDet` =  442 for `blade` = 3,4,5

### RECHIT
    
* Local position of the `SimHit` (ranges below are approx) </br>
        `hx` [-0.83,+0.83] cm  for 1x2 modules             </br>
        `hx` [-1.67,+1.67] cm  for 2x2 modules             </br>
        `hy` [-2.24,+2.24] cm                  
		
* ToF of the `SimHit` corrected for the time-to-det (added in CMSSW 10.5.X</br> 
        `ht` in ns

* Charge and ADC </br>
        `q` = charge of `RecHit` in e               </br>
        `DgCharge` = charge of a single pixel in ke </br>
        NB: to get the ADC count (integer): `DgCharge*1000./600`. or `DgCharge*1000./135.` </br>
        Check always the value in the pset, e.g. </br>
        ```
        SimTracker/SiPhase2Digitizer/python/phase2TrackerDigitizer_cfi.py
        process.mix.digitizers.pixel.PixelDigitizerAlgorithm.ElectronPerAdc = cms.double(135.)
        RecoLocalTracker/SiPixelClusterizer/python/SiPixelClusterizer_cfi.py
        process.siPixelClusters.ElectronPerADCGain=cms.double(135.)
        ```
