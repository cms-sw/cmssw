INTRODUCTION
============

This small README file is here to guide you in the process of running
the Vertex Validation, slimmed version, on RelVal samples, in case you
want to perform test on tracking and vertexing. The idea here is not
to give you pre-cooked python configuration files, but to teach you
how you could use the most common tool available in CMS to perform the
same. We will mainly use cmsDriver and its powerful option to create
the python cfg that we will run, and das_client to explore and find
suitable samples to run upon. At the end of this page there is the description of other standalone analyzers, configurations and Root macros. Let start with order.

PREREQUISITES
=============

We assume that from this point onward, you have setup a proper CMSSW
area and that you have source its environments, since all the script
that we will be using are available to you only after you performed
such actions.

FIND PROPER SAMPLES
===================

The first thing that we need to do is to find appropriate samples to
run upon. Our suggestion is to start from the RelVal samples that are
regularly produced for every release and pre-release, since this will
avoid all the burden of properly selecting the PU and generation
snippet. In case you want to use anything other than what is available
as RelVal, we assume you are familiar enough with the production
mechanism that you can take care of it alone: no instructions will be
given here.

FIND GEN-SIM-DIGI-RAW-HLTDEBUG samples FOR A SPECIFIC RELEASE
-------------------------------------------------------------

In order to check what samples are available for, e.g. the CMSSW_7_2
release cycle, issue the command

```
das_client.py --query='dataset=/RelValTTbar*/*7_2_0*/*GEN-SIM-DIGI-RAW-HLTDEBUG' --format=plain
```

and pick up the proper dataset among the ones printed directly on the
screen. Here we picked up
/RelValTTbar_13/CMSSW_7_2_0_pre1-PU25ns_POSTLS172_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG.

FIND ALL FILES BELONGING TO A SPECIFIC DATASET
----------------------------------------------

In order to discover which files belong to the selected dataset, you
have to issue the following command (of course you have to change the
dataset name in the query, using the one you discovered in the
previous point...)

```
das_client.py --limit 0 --query='file dataset=/RelValTTbar_13/CMSSW_7_2_0_pre1-PU25ns_POSTLS172_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG' --format=plain | sort -u > gen_sim_digi_raw_files.txt 2>&1
```

This will write the discovered files directly into the ASCII file
gen_sim_digi_raw_files.txt, that will be used as input to the
following cmsDriver commands.

RUN RECO AND VERTEX VALIDATION
==============================

Inn order to run the vertex validation starting from RAW file, you
need to create a proper python cfg. As said, instead of preparing a
pre-cooked one, we think its more useful to give you the cmsDriver
command that will dynamically prepare it for you. To obtain such a cfg
file, issue the following command:

```
cmsDriver.py step3  --conditions auto:run2_mc -n 100 --eventcontent DQM -s RAW2DIGI,RECO,VALIDATION:vertexValidationStandalone --datatier DQMIO --filein filelist:gen_sim_digi_raw_files.txt --fileout step3_VertexValidation.root --customise SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1 --magField 38T_PostLS1
```

This will create the python configuration file **and will
automatically run cmsRun on it. If instead you want to just produce
the configuration, e.g. for inspection and further customization, you
can add the option:

```
--no_exec
```

to the previous command, This command will produce and output file
named step3_VertexValidation,root that will contain all the histograms
produce by the Vertex Validation package. The internal format of the
ROOT file follows the DQMIO rules, to have better performance while
running harvesting.

RUN VERTEX VALIDATION WITHOUT RECO
----------------------------------

It is also possible to re-run only the validation without
reconstruction (e.g. for developing the validation package itself).
For that you need first the list of GEN-SIM-RECO files, i.e. e.g.

```
das_client.py --limit 0 --query='file dataset=/RelValTTbar_13/CMSSW_7_2_0_pre1-PU25ns_POSTLS172_V1-v1/GEN-SIM-RECO' --format=plain | sort -u > gen_sim_reco_files.txt 2>&1
```

The configuration can then be generated with

```
cmsDriver.py step3  --conditions auto:run2_mc -n 100 --eventcontent DQM -s VALIDATION:vertexValidationStandalone --datatier DQMIO --filein filelist:gen_sim_reco_files.txt --secondfilein filelist:gen_sim_digi_raw_files.txt --fileout step3_VertexValidation.root --customise SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1 --magField 38T_PostLS1 --no_exec
```

Note the `secondfilein` parameter for specifying the RAW files for the
"2-files solution".



RUN FINAL HARVESTING TO PRODUCE EFFICIENCY, FAKE, MERGE AND DUPLICATE RATE PLOTS
================================================================================

The outcome of the previous step is not yet suitable to be browsed
using plain ROOT. Moreover all the important plots have not yet been
produce. You need to finalize the processing running the harvesting
sequence. Again, we think it is better to provide you with the
cmsDriver command to do that:

```
cmsDriver.py step4  --scenario pp --filetype DQM --conditions auto:run2_mc --mc  -s HARVESTING:postProcessorVertexStandAlone -n -1 --filein file:step3_VertexValidation.root -no_exec
```
This command will create a final, plain, ROOT file named:
DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root that will contain
all the folders and plots produced by the Vertex Validation package.


FURTHER CUSTOMIZATION
=====================

If you want to customize the default vertex validation sequence, both
the first one and the ones used in harvesting, you need to manually
edit the configuration files produce by the previous cmsDriver
commands. To ease this operation, you can point your browser here:

https://github.com/cms-sw/cmssw/blob/CMSSW_7_2_X/Validation/RecoVertex/python/PrimaryVertexAnalyzer4PUSlimmed_cfi.py

for the first default, and here:

https://github.com/cms-sw/cmssw/blob/CMSSW_7_2_X/Validation/RecoVertex/python/PrimaryVertexAnalyzer4PUSlimmed_Client_cfi.py

for the default used in the harvesting step.

Enjoy.

DETAILED DESCRIPTION OF THE CODE
================================
## Plugins
### AnotherPrimaryVertexAnalyzer
It produces several histograms using a vertex collection as input: the vertex x, y and z  positions, the number of vertices (vs the instantaneous luminosity), the number of tracks per vertex and the sum of the squared pt of the tracks from a vertex (with or without a cut on the track weight), the number of degrees of freedom (also as a function of the number of tracks), the track weights and the average weight and the average values of many of the observables above as a function of the vertex z position. 
Distributions are produced also per run or per fill: the number of vertices and their position as a function of the orbit number and of the BX number. By configuration it is possible to choose among TProfile or full 2D plots.
All these histograms can be filled with a weight to be provided by an object defined in the configuration.
An example of configuration can be found in `python/anotherprimaryvertexanalyzer_cfi.py`.

### AnotherBeamSpotAnalyzer 
`AnotherBeamSpotAnalyzer` is the plugin name which corresponds to the code in `src/BeamSpotAnalyzer.cc`. It produces several histograms to monitor the beam spot position; the name of a beamspot collection has to be provided as input. The histograms are the beam spot position and width and their dependence as a function of the orbit number (one set of histograms per run).
An example of configuration can be found in `python/beamspotanalyzer_cfi.py`.

### BSvsPVAnalyzer
It produces distributions related to the relative position between vertices and the beam spot. It requires a vertex collection and a beam spot collection as input. By configuration it is possible to control whether the comparison has to take into account the tilt of the beamspot. The distributions are the differences of the vertex and beam spot position coordinates, the average of these differences as a function of the vertex z position and, for each run, the dependence of these differences as a function of the orbit number and of the BX number. Configuration parameters have to be used to activate or de-activate those histograms which are more memory demanding.
An example of configuration can be found in `python/bspvanalyzer_cfi.py`.

### MCVerticesAnalyzer
It produces distributions related to the multiplicity of (in-time and out-of-time) pileup vertices (or interactions), to the position of the main MC vertex and to the z position of the pileup vertices. It correlates the average number of pileup interactions with the actual number of pileup interactions. It can be configured to use weights. 
An example of configuration can be found in `python/mcverticesanalyzer_cfi.py`.

### MCVerticesWeight
It is an `EDFilter` which computes an event weight based on the MC vertices z position to reproduce a different luminous region length. It can be configured to reject events or the weight can be used to fill the histograms of `MCVerticesAnalyzer`.
An example of configuration can be found in `python/mcverticesweight_cfi.py`

###MCvsRecoVerticesAnalyzer
It produces histograms to correlate the number of reconstructted vertices with the number of generated vertices or with the average pileup, to correlate the z position of the reconstructed vertices with that of the MC vertices and to check how many times the closest reco vertex to the main MC vertex is the first one in the vertex collection. It can be configured to fill histograms with weights to be provided with `MCVerticesWeight`.
An example of configuration can be found in `python/mcvsrecoverticesanalyzer_cfi.py`

## Configurations
* `test/allanalyzer_example_cfg.py` is a configuration which uses the `AnotherPrimaryVertexAnalyzer`, `AnotherBeamSpotAnalyzer` and `BSvsPVAnalyzer` and that can be used to analyze real data events. It uses VarParsing to pass the input parameters like the input files and the global tag.
* `test/mcverticesanalyzer_cfg.py` an example of configuration which uses the plugins to study the MC vertices
* `test/mcverticessimpleanalyzer_cfg.py` an example of configuration which uses the plugins to study the MC vertices
* `test/mcverticestriggerbiasanalyzer_cfg.py` an example of configuration which uses the plugins to study the MC vertices.

