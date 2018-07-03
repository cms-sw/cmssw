### Analysis Suite for the Digitizer
There are two analyzers (DQMEDAnalyzer) in the package 
 
* Phase2TrackerMonitorDigi : monitor digi propertied from the collection
* Phase2TrackerValidateDigi : correlates digis with SimHits using DigiSimLink and plots efficiency as a function of Eta, Pt and Phi

Options added in these modules so that they can be configured either for the inner pixel digi collection (PixelDigi) or for the outer tracker digi collection(Phase2TrackerDigi). The switching is done by a configuration parameter

There are two configuration files to run the application where DQM histograms are saved. In the first steo DQM histograms are stored in EDM file and in the second step DQM harvesting is done. At the moment the application is configured for four modules (monitoring + validation for the inner pixel and outer tracker). One can easily modyfy the configuration according to the need.

* DigiTest_cfg.py          (creates and fills histograms and finally write in EDM format)
* DigiTest_Harvest_cfg.py  (harvests the EDM file and writes histograms in a root file )

