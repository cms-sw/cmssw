# INTRODUCTION

This brief README file is meant to explain the usage of the
RecoMaterial tools that are present in the
Validation/Geometry/test directory.

# GOALS

The main goal is to provide the users with the necessary
tools to assess the goodness of the material description
used during the reconstruction process (compared to the one
used in the Monte Carlo simulation). In order to do that,
two main ingredients are necessary:

* the material map as derived from the Monte Carlo
  simulation
* the material description as "derived" during the
  reconstruction phase

The procedure to derive the former is extensively documented
in a TWiki page
[here](https://twiki.cern.ch/twiki/bin/viewauth/CMS/TrackerMaterialBudgetValidation).
In the rest of the documentation we will assume that the
user has already followed those instructions and produced
all the required ROOT files.

**NOTA BENE**: it will also be implied that the samples used
to derive the material profile from the Simulation have been
produced **without** any vertex smearing.

The procedure to derive the latter is the subject of the
next few sections.

**REMINDER**: it is a user's responsibility to carefully
check which geometry is loaded (either via GT or via files)
in all the scripts used to derive the material description
from the simulation. As of this PR the default is to use the
PhaseI geometry, while the scripts that are explained below
use the Run2 geometry.

## Material Map used during track reconstruction.

Energy loss and multiple scattering are taken into account
during the patter recognition. In order to have a somewhat
better understanding of the mechanism put in place to do
that, you can read the following [slides]()

The main idea to derive the reco-material map effectively
used during the patter-recognition is to ask to each layer
traversed by the particle, what are its material properties
(as injected by XML files or DB) and simply integrate these
values in eta.

A simple EDAnalyzer (*TrackingRecoMaterialAnalyser*) has
been put in place. The Analyzer will re-fit the tracks that
have been reconstructed during the main reconstruction
phase. The refit is mandatory in order to derive the proper
TSOS at each detector and correct the "effective material"
that the track sees while traversing the detector. The
analyzer can be safely coupled with the standard
reconstruction step.

### Procedure

The steps to be followed in order to derive the
reco-material map are explained below.

#### Sample Generation

```
cmsRun SingleMuPt10_pythia8_cfi_GEN_SIM.py
```

This will generate 1K Single muons (and the corresponding
anti-particle) with Pt=10 and *without* any vertex smearing.
The vertex smearing has been turned off in order to have a
reliable measure of the eta of the track.

#### Sample Digitization

```
cmsRun SingleMuPt10_step2_DIGI_L1_DIGI2RAW_HLT.py
```

#### Sample Reconstruction

```
cmsRun SingleMuPt10_step3_RAW2DIGI_L1Reco_RECO_VALIDATION_DQM.py [geomDB=False]
```

This step will run the regular tracking reconstruction,DQM
and Validation steps, plus the ad-hoc analyzer to derive the
reconstruction material map. The *optional* geomDB=False
flag can be supplied to instruct the reconstruction job to
read the geometry+material description from (possibly local)
XML files in release rather than from the DB. This is
particularly useful to test the changes that the user may
have made to the material description.

#### Sample Harvesting

```
cmsRun SingleMuPt10_step4_HARVESTING.py [geomDB=False]
```

The *optional* geom flag must follow the same prescription
that has been used in the previous step (i.e. be present if
it were in the previous step, absent otherwise)

The final output of the procedure is a plain ROOT file named
DQM_V0001_R000000001__SingleMuPt10_from{DB,LocalFiles}__Run2016__DQMIO.root
according to the flags used while creating it. It contains
the profile of the material description as seen by the
tracks during the patter recognition.

#### Plots Production

In order to have plots comparing the 2 material description
(data vs MC), the user has to run the  following ROOT macro:

```
root -b -q
'MaterialBudget_Simul_vs_Reco.C("DQM_V0001_R000000001__SingleMuPt10_fromLocalFiles__Run2016__DQMIO.root",
"FromLocalFiles")'
```

using the input files according to her needs, and properly
tuning the accompanying label ('FromLocalFiles' in this
case). A bunch of png files will be produced. The one that
compares the 2 material descriptions is names
MaterialBdg_Reco_vs_Simul_%label.png. Take a look at it and
see how things are.
