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
that, you can read the following [slides](://indico.cern.ch/event/512686/contributions/2182630/attachments/1280489/1901936/TrackerDPG_POG_20160527_MR.pdf)

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
reco-material map are included in few shell scripts, that
automatically take care of issuing the correct cmsDriver command for
the different scenarios. The files to be used are:

```
runMaterialDumpAnalyser.sh
runMaterialDumpAnalyser_PhaseI.sh
runMaterialDumpAnalyser_PhaseII.sh
```

Each script accept two command line options, that could be useful for
quick testing:

```
-n XXX
```

to generate only XXX events in place of the default 5K.

```
-g GEOMETRY
```

to select a geometry different from the default one included in the
shell script itself.
