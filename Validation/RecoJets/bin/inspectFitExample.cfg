$ -- configBlockIO -- $
$=====================================================================
$
$ input/output files
$
$=====================================================================

  histInput    = analyzeMuon.hist                                     $ hist file, which contains the names of the 
                                                                      $ histograms to be plottet 
  rootInput    = analyzeSemiLeptonicEvents_all.root                   $ root input file(s) corresponding to samples
$                analyzeSemiLeptonicEvents_sig.root                   $ (separated by blanks);

  inputDirs    = muonSample                                           $ directory in which the histograms are kept
$                muonCutKin                                           $ in within the root file; for edm::Analyzers 
                                                                      $ this  corresponds to the module name of the 
                                                                      $ Analyzer in the cfg file (separated by blanks)
  histFilter   = fit                                                  $ list of filter strings; only histograms con-
                                                                      $ taining this stings as substrings are plotted
                                                                      $ (separated by blanks)
  filterOption = begins                                               $ histogram filter option ('begins', 'ends' & 
                                                                      $ 'contains' are supported
  rootOutput   =                                                      $ root output file to write histograms to
                                                                      $
  outputDir    =                                                      $ output directory in root output file 
                                                                      $
  outputLabels = muonSample                                           $ output labels for multiple ps/eps files 
$                muonCutKin                                           $ (separated by blanks)
                                                                      $
                                                                      $
  writePlotsTo = .                                                    $ directory to save the plottet histograms to;
                                                                      $ '.' is the working directory
  writePlotsAs = ps                                                   $ decide wether to write histograms to [ps] or 
                                                                      $ [eps] 

$ -- configBlockHist -- $
$=====================================================================
$
$ canvas and histogram steering
$
$=====================================================================

  xLog        =                                                       $ logs can be declared for each histogram 
                                                                      $ individually; per default they are  
  yLog        =                                                       $ switched off
                                                                      $
  xGrid       =                                                       $ grids can be declared for each histogram
                                                                      $ individually; per default they are 
  yGrid       =                                                       $ switched off

  histScale   =                                                       $ histogram scale; can be steered for each 
                                                                      $ histogram individually
  histMaximum =                                                       $ histogram maximum; can be steered for 
                                                                      $ each histogram individually
  histMinimum =                                                       $ histogram minimum; can be steered for 
                                                                      $ each histogram individually

  histErrors  =    1                                                  $ draw histogram errors for given sample?

  histType    =    1                                                  $ defines wether histogram should be plotted
                                                                      $ as line[0], with poly markers[1] or filled 
                                                                      $ [2]; default is line [0]; can be steered 
                                                                      $ for each sample individually
  histStyle   =    1                                                  $ defines line or fill style for each sample

  histColor   =    2                                                  $ defines line/marker/fill color for each 
                                                                      $ sample
  lineWidth   =    5                                                  $ defines line width for each sample

  markerStyle =   20                                                  $ defines marker style for each sample

  markerSize  =  2.3                                                  $ defines marker size for each sample

                                                                      $ set axes titles of histograms;
  xAxes       =                                                       $ has to be givin in " and has to end with;

  yAxes       =

  legEntries = "all events";                                          $ should contain a legend entry for each 
               "after #mu kin";                                       $ sample in " and separated by ;
                                                                      $
                                                                      $
  legXLeft   = 0.25                                                   $ so far a steering of the legend coord's 
                                                                      $ is only supported globally
  legXRight  = 0.95                                                   $
                                                                      $
  legYLower  = 0.70

  legYUpper  = 0.95


$ -- configBlockFit -- $
$=====================================================================
$
$ histogram fitting
$
$=====================================================================

  fitFunctionType  = 0                                                $ type of fit function to be used (at
                                                                      $ the moment only 0=gauss is implemented)
                                                                      $
  fitFunctionName  = func                                             $ name of the fit function to be used 
                                                                      $ (user defined)
  fitFunctionTitle = Fit(Gauss)                                       $ title of the fit function to be used	
                                                                      $
  fitLowerBound    = 0.                                               $ lower boundary for fit function
                                                                      $
  fitUpperBound    = 2.                                               $ upper boundary for fit function

  targetLabel      = res                                              $ set labels of target histograms
                                                                      $ to contain corresonding fit results
                                                                      $ [cal], [res], [gaussMean] and 
                                                                      $ [gaussSigma] are supported so far;
                                                                      $ more then one label can be set 
                                                                      $ separated by blanks

  titleIndex  = 0 0 0 0 0                                             $ index in *AxesFit vector to be used  
                1 1 1 1 1                                             $ for the corresponding histogram 

  xAxesFit    = "(E_{jet}^{rec}-E_{jet}^{had})/E_{jet}^{had}";        $ vector of x axes title templates for  
                "(E_{t,jet}^{rec}-E_{t,jet}^{had})/E_{t,jet}^{had}";  $ fitted histograms; separated by ';'

  yAxesFit    =                                                       $ vector of x axes title templates for  
                                                                      $ fitted histograms; separated by ';'
