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
* track <-> seed
* track <-> hit
* seed <-> hit
* glued strip hits -> mono and stereo strip hits
* vertex <-> track
* hit <-> SimHit
* TrackingParticle <-> SimHit (but see caveat below)
* TrackingParticle <-> TrackingVertex

By default the ntuple includes hits and seeds, which makes the ntuple
rather large. These can be disabled with switches in
[`trackingNtuple_cff`](python/trackingNtuple_cff.py).

* Note that to include seeds you have to run reconstruction as seeds are
not stored in RECO or AOD.

* Note that to include hits for pileup events you need to use the
"playback mode" of MixingModule. This is because also with this option
also the SimHits are included, and SimHits of pileup events are
available only with this procedure (by default only signal event
(BX=0, event=0) SimHits are available). The playback mode can be
enabled e.g. by passing `--pileup <scenario> --pileup_input <files>`
to `cmsDriver.py`. Both the `<scenario>` and `<files>` need to be
exactly the same that were used for the DIGI-RAW file production. For
more information see
[SWGuideMixingModule](https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideMixingModule#playback_option).
If the "playback mode" is not enabled, an exception will be thrown in
the C++ code for a missing SimHit.


### Caveats

#### RecHit <-> SimHit links

The RecHits (or clusters) are linked to simulation truth
(SimTracks/TrackingParticles) with `PixelDigiSimLink` and
`StripDigiSimLink` objects. In these links there is no information
about SimHits. The RecHits are linked to SimHits in the ntuple code by
using the hit -> TrackingParticle links, and TrackingParticle ->
SimHit links. There is, however, an ambiguity that a TrackingParticle
may have multiple SimHits on a single detector (with same DetId), e.g.
because of delta rays. Currently the "first" SimHit on a detector of a
RecHit linked to the TrackingParticle is linked to the RecHit. The
assignment is a bit random, but there is very little further one can
do without further reco<->sim information.

#### Delta ray SimHits are filtered out

The SimHits with `particleType` of an electron for non-electron
TrackingParticles are filtered out. This case happens e.g. for delta
rays, when the delta ray electron is simulated with the same
SimTrackId as the main particle. As the SimHit -> TrackingParticle
association is based on SimTrackId, the delta ray SimHits get
associated to the TrackingParticle. Further, the SimHits of a
TrackingParticle are sorted by their time of flight for the ntuple, so
the delta ray hits may get interleaved with the main particle hits,
leading to possible confusion (especially when one does not print the
SimHit `particleType` information). Since we are mainly interested of
the main particle SimHits, the delta ray SimHits are removed from the
ntuple.

This SimHit filtering does have the consequence that there can be
cases where a track is matched to TrackingParticle (by
`QuickTrackAssociatorByHits`), but redoing the association by hits in
the ntuple fails, because delta ray SimHit induced a RecHit that was
included in the track.

#### Vertex <-> track links

If the input track collection is different from the one used for
vertex reconstruction, the vertex <-> track links are not created.
Possible use case is making ntuple of the tracks from a single
iteration (without going through `generalTracks`).
