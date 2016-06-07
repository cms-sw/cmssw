Tracking validation
===================

The workhorse of the tracking validation is MultiTrackValidator, for
which documentation is available in
https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideMultiTrackValidator.
In addition to tracks it can be used to study seeds.

The plotting tools are documented in
https://twiki.cern.ch/twiki/bin/view/CMS/TrackingValidationMC.


Ntuple
------

There is also an ntuple-version of MultiTrackValidator, called
[TrackingNtuple](plugins/TrackingNtuple.cc). It can be included in any
`cmsDriver.py`-generated workflow containing `VALIDATION` sequence by
including
`--customise Validation/RecoTrack/customiseTrackingNtuple.customiseTrackingNtuple`
argument to the `cmsDriver.py`. The customise function disables all
output modules and replaces the validation sequence with a sequence
producing the ntuple in `trackingNtuple.root` file. If ran without
RECO, it needs both RECO and DIGI files as an input.

For the ntuple content, take a look on the
[TrackingNtuple](plugins/TrackingNtuple.cc) code itself, and an
example PyROOT script for analysis,
[`trackingNtupleExample.py`](test/trackingNtupleExample.py). The
script uses a simple support library
[`ntuple.py`](python/plotting/ntuple.py), but its use is not
mandatory, i.e. you can use the ntuple also "directly". The main
benefit of the library is to provide OO-like interface for all the
links between the objects:
* track <-> TrackingParticle
* track -> seed
* track <-> hit
* seed <-> hit
* glued strip hits -> mono and stereo strip hits
* vertex -> track
* TrackingParticle <-> hit
* TrackingParticle <-> TrackingVertex

By default the ntuple includes hits and seeds, which makes the ntuple
rather large. These can be disabled with switches in
[`trackingNtuple_cff`](python/trackingNtuple_cff.py). Note that to
include seeds you have to run reconstruction as seeds are not stored
in RECO or AOD.