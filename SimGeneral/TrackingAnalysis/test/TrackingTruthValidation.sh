#! /bin/bash

cat $CMSSW_RELEASE_BASE/src/Validation/RecoTrack/test/macro/TrackValHistoPublisher.C | sed \
      -e 's:REF_FILE:TrackingTruthValidationWithoutConditions.root:g' \
      -e 's:NEW_FILE:TrackingTruthValidationWithConditions.root:g' \
      -e 's:REF_LABEL:Without conditions:g' \
      -e 's:NEW_LABEL:With conditions:g' \
      -e "s:REF_RELEASE:$CMSSW_VERSION:g" \
      -e "s:NEW_RELEASE:$CMSSW_VERSION:g" \
      -e 's:REFSELECTION:highPuritySelection:g' \
      -e 's:NEWSELECTION:highPuritySelection:g' \
      -e 's:TrackValHistoPublisher:RelValTTbar:g' \
      -e 's:MINEFF:0.5:g' \
      -e 's:MAXEFF:1.025:g' \
      -e 's:MAXFAKE:0.7:g' \
    > TrackingTruthValidationMacro.C

