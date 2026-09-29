# Retraining the pixel-track machine-learned models

Everything needed to retrain, check and deploy the models used by the pixel+outer-tracker
track reconstruction iteration. One entry point does the work:

```bash
./retrain_prompt.sh    <step> [options]     # hltPhase2PixelTracksSoAWithStubs
```

Run it with `--help` for the full list. Every step prints each command before running
it, and `--dry-run` prints them without running anything.

## The models

The iteration has three trainable models, trained in this order:

| script | step | model | where it runs | deployed as | to deploy |
|---|---|---|---|---|---|
| prompt | `triplet` | per-triplet gate | inside the track-building kernel | `plugins/alpaka/CATripletDNNWeights_<arm>.h` | rebuild |
| prompt | `gate` | track DNN (loose -> tight) | inside the track classification kernel | `plugins/alpaka/CATrackDNNWeights_<arm>.h` | rebuild |
| prompt | `hp` | final high-purity selector of the iteration | a separate module after reconstruction | `RecoTracker/FinalTrackSelectors/data/PixelTrackTorchHighPuritySelector/<arm>_tree31_<tag>_<date>.bin` | copy the file |

`<arm>` is `prompt`. The first two models are compiled into the kernels, so deploying
them means rebuilding; the selectors are read at run time, so deploying one means copying a file
and pointing the module's configuration at it.

What each step does:

* `dump` reconstructs events and writes a training dataset (one row per track, or per triplet,
  with the truth). `--for triplet`, `--for gate` and `--for hp` choose the dataset; each training
  step produces its own dataset on the fly when it is not in `--work` yet.
* `triplet` trains the per-triplet gate on the triplet dataset and bakes it into its header.
* `gate` trains the track DNN on the `loose` dataset and bakes it into its header.
* `hp` builds a feature cache from the `tight` dataset, trains the final selector, exports it in
  the compact binary the selector loads and stages it in `--work`. `--deploy` also copies it into
  the data directory; the configuration is edited by hand, as printed at the end of the step.

## The final selectors

The final selector is a gradient-boosted forest run by `PixelTrackForestHighPuritySelector@alpaka`
(`useHitFeatures=True`), reading a compact `.bin`. Every model consumes a prefix of the
`PixelTrackFeaturesSoA` column layout, in column order:

| selector | module | configuration file | model file it loads | features | values |
|---|---|---|---|---|---|
| prompt | `hltPhase2PixelTrackTorchHighPuritySelector` (the forest replaces the Torch model under `phase2CAStubs`) | `hltPhase2PixelTrackTorchHighPuritySelector_cfi.py` | `prompt_tree31_ew4m0_20260829.bin` | 31: 17 fit and covariance, 10 hit and stub, `rzChi2`, `meanStubKappa`, `leverArm`, `rMax` | fp32 |

The configuration file is in `HLTrigger/Configuration/python/HLT_75e33/modules/`. One trainer,
`train_merged_forest.py`, produces it; the `hp` step calls it with the recipe of the file
the chain runs today, and stages the result under the trainer's own name
`<arm>_tree<N>_<tag>_<date>.bin` in `--work`. The chain, exactly as the script prints it
(`$W` = `--work`, `$M` = this directory):

**Prompt** (`./retrain_prompt.sh hp --work $W ...`):

```bash
python3 $M/nano_loader.py cache-arm $W/prompt31_cache.npz $W/trackNano_<sample>_tight.root ... --prefix TrkPrompt --label mtv
python3 $M/train_merged_forest.py --cache $W/prompt31_cache.npz --out $W/prompt_forest \
    --feats 31 --label mtv --recall 0.995 --arm prompt --tag wp --date <date> --threads 16 \
    --eff-weight 4.0 --no-fp16 --wp-rule uniform
# staged: $W/prompt_forest/prompt_tree31_wp_<date>.bin
```

Update in `hltPhase2PixelTrackTorchHighPuritySelector_cfi.py`, on the
`_hltPhase2PixelTrackForestHighPuritySelector` producer: `model` (the new file name),
`scoreThreshold` (after the in-situ scan below). `useHitFeatures` stays `True`.

Each `hp` step also leaves `result.json` (every metric, the working point, the feature list and
the full recipe) and `thrmap.txt` (score threshold against true-track recall, with fake
rejection) next to the staged file, and reads the `.bin` back through `read_compact_tree.py`,
re-scoring test rows against the trained booster.

The trainer's inputs and labels, in short: a cache carries the features in the selector's column
order and two labels per track, `y_mtv` (matched to any TrackingParticle: a true track, the
training target) and `y_eff` (matched to a TrackingParticle passing the efficiency selection: the
recall axis). Recall is quoted on the second, fake rejection on the tracks matched to nothing;
the true tracks that fail an efficiency cut are positives in training and on neither axis.

### Working-point rules

`train_merged_forest.py --wp-rule` chooses how the trained scores are turned into a threshold.
Early stopping optimises fake rejection at that threshold, so the rule shapes the model, not
only its cut. `WP_RULE=uniform|global|profile` sets it in the entry script.

* `uniform` (the default): the largest threshold at which the true-track
  recall is at least `--recall` (0.995) in every pT, |eta| and |dxyBS| bin of the trainer's
  standard edges (pT 0.9/1.5/3/10/30, |eta| 1.5/2.5, |dxyBS| 0.1/0.5/1/2/5, each axis open at
  the top). Bins with fewer than 100 true tracks do not constrain it; with no usable bin the
  global rule applies. It depends on nothing outside the run, so the same command means the same
  thing on any training sample, and it protects the sparse tails (high pT, forward, displaced)
  that a single overall quantile lets the populous bins pay for.
* `global`: the threshold at which the overall true-track recall is `--recall`.
* `profile`: match or exceed a reference model bin by bin. `make_profile.py` measures the model
  the configuration loads today at its threshold on this run's own cache, and the working point
  becomes the threshold at which every bin reaches that recall. Use it when the question is "is
  the new model at least as good as the deployed one everywhere"; it ties the result to that
  model and to the sample it is measured on.

Whatever the rule, the threshold in `result.json` is an offline starting point. The deployed
value is chosen by running the reconstruction and the track validation (see "Working points"
below), which every `hp` step says at the end.

### What a retrain reproduces

* Prompt: the deployed file's recipe (31 features, true high-pT tracks up-weighted, fp32
  export) except for the rule: that file was trained against the per-bin profile of its
  predecessor; `WP_RULE=profile` does the same against the file deployed today.

## Order

**Within an iteration: `triplet` -> `gate` -> `hp`.** Every model decides on the population the
previous one left behind, so each dataset must be produced with the upstream models deployed
and compiled in:

* the track DNN trains on quality **`loose`** tracks with the deployed track DNN switched off
  at dump time, because that is the population it decides on -- before it decided;
* the final selector trains on quality **`tight`** tracks, which are the ones the deployed
  track DNN promoted, so that dump runs the full chain;
* regenerate each step's dataset only after the previous step is deployed **and built**.

**After any change to the track fit** (fit refactor, magnetic field, material description):
re-run the `gate` and `hp` steps. Their features are fit outputs,
and a model trained on an older fit is quietly mismatched; `compare_track_dnn_banks.py` scores
the baked track-DNN header directly and makes that visible.

## Inputs

Datasets are produced by reconstructing events and writing out per-track or per-triplet tables.
Pass the events with `--input`, and an already-produced dataset with `--dataset`:

```bash
--input /path/to/dir            # a directory of .root files
--input a.root,b.root           # a comma-separated list
--input files.txt               # a text file with one .root path per line
--dataset /work/trackNano_ttbarPU_loose.root    # skip the dump, use this
```

With several `--sample` names, give each one its own events (otherwise all of them are dumped
from the same files and the sample diversity the trainers assume is not there):

```bash
--sample "ttbarPU displacedPU qcdPU" --input ttbarPU=/data/ttbar,displacedPU=/data/susy,qcdPU=/data/qcd
--sample "ttbarPU displacedPU qcdPU" --input /data/ttbar --input /data/susy --input /data/qcd
```

The named form wins; repeated `--input` options are paired with the `--sample` names in order.
`--events` applies to the dump it is on, so a different event count per stage means one `dump`
per stage, after which the training steps pick the datasets up from `--work`:

```bash
W=/somewhere/work; S="ttbarPU displacedPU qcdPU"
IN="ttbarPU=/data/ttbar,displacedPU=/data/susy,qcdPU=/data/qcd"
./retrain_prompt.sh dump --for triplet --work $W --sample "$S" --input "$IN" --events 100
./retrain_prompt.sh triplet --work $W --sample "$S"       # then rebuild
./retrain_prompt.sh dump --for gate    --work $W --sample "$S" --input "$IN" --events 300
./retrain_prompt.sh gate    --work $W --sample "$S"       # then rebuild
./retrain_prompt.sh dump --for hp      --work $W --sample "$S" --input "$IN" --events 1000
./retrain_prompt.sh hp      --work $W --sample "$S"
```

Every training step is given all the per-sample datasets at once and trains on their union;
the files are named `trackNano_<sample>_<quality>.root` and `tripletNano_<sample>.root`
in `--work`. `--chassis <module>` alone is also enough to
produce a dataset, since a chassis carries its own input file list.

A step with neither option looks for the standard dataset names in `--work`; if they are not
there it stops and says so.

**The base reconstruction configuration.** The two dump configurations
(`trackNano_config.py`, `triplet_dump_cfg.py`) only add tables to an existing reconstruction
process; they do not build one. With `--input` given, the entry script generates that base
configuration once into the working directory with `make_base_config.sh`, which is an ordinary
`cmsDriver.py` call:

```
cmsDriver.py step2 -s L1P2GT,HLT:75e33_trackingOnly --processName HLTX \
  --conditions auto:phase2_realistic_T35 --geometry ExtendedRun4D121 --era Phase2C22I13M9 \
  --procModifiers ngtScouting --filein <files> --no_exec \
  '--output={}' \
  '--inputCommands=keep *, drop *_hlt*_*_HLT, drop triggerTriggerFilterObjectWithRefs_l1t*_*_HLT' \
  --python_filename <out>
```

`--output={}` keeps ConfigBuilder from adding a `PoolOutputModule` and its `output_step`
EndPath: the dump configs write their own NanoAOD file, and without it every dump job would
also write a full GEN-SIM-DIGI-RAW copy of its input. `--inputCommands` drops the products of
an earlier HLT process that RelVal inputs carry, so unqualified `InputTag`s cannot resolve
against them; `make_base_config.sh --input-commands ''` omits it. `ngtScouting` (which
includes `phase2CAStubs` and `pixelTrackMask`) leaves the pixel chain every model is trained
on unchanged and replaces the iterative tracking behind it with a pass-through, so a dump is
cheap; `--modifiers` sets a different set verbatim.

The tracking-only menu runs the whole chain the training data comes from -- the iteration and
its selector -- and skips everything else. Use `--chassis <module>` (or
`NANO_CHASSIS` / `DS_CHASSIS`) to dump on top of a different configuration instead; its
directory must be on `PYTHONPATH`.

**Dataset provenance.** Every track-level dump prints one `[trackNano CHAIN-STATE]` line
recording the state it ran in: whether each in-kernel model was on, at which
threshold, the buffer sizes and the quality. A dataset is only valid for the state on that
line. To dump at a working point that is not in the configuration yet, use the
`NANO_PROMPT_*` overrides documented in `trackNano_config.py`; they are printed
on the same line.

## The build switch for the triplet step

The per-triplet dump is compiled out by default. Its single switch is the last line of
`RecoTracker/PixelSeeding/plugins/alpaka/CATripletDumpMacro.h`; with it commented out the build
is unchanged, which is how production must stay.

```
# before the dump: uncomment '#define CA_TRIPLET_DUMP', then
scram b code-format && scram b -j
#   ... run the triplet dataset step ...
# after the dump: comment it out again, then
scram b code-format && scram b -j
```

The `triplet` step checks the switch and stops with these instructions if it is off. It does
not edit the header.

## Working points: what the tooling derives, and what is deployed

**The threshold a trainer prints is a starting point, not a deployed value.** The deployed
values are chosen by running the reconstruction and the track validation, on axes an offline
scorer cannot see: the efficiency ceiling, the combinatorial load, container occupancy, and the
number of fakes handed to the next stage. So by default each step bakes **the value the chain
runs today** -- a retrain with no options replaces the model and changes nothing else. Use
`--threshold rule` to bake the value the run derived, or `--threshold <x>` for an explicit one.
Each step prints the value the configuration currently carries and where it lives.

| model | the rule the tooling applies | configuration parameter |
|---|---|---|
| triplet gate | largest threshold whose per-track survival stays above a floor in the worst (displacement, momentum) group | `tripletDNNThreshold` on the CA producer; when it is not set there, the value baked into the header applies |
| track DNN | per-track recall -- legitimate here, one decision per track | `trackDNNThreshold` on the CA producer; same fallback |
| final selector | true-track recall in every pT / eta / dxy bin (the `uniform` rule above) | `scoreThreshold` on the selector |

The triplet gate is the one to be careful with. A track leaves about ten real triplets behind,
so a rule stated as "keep 99 % of real triplets" keeps only about 0.99^10 ~ 90 % of the tracks,
which is why the default rule works on per-track survival. The global-recall rule is available
for comparison with `THRESHOLD_RULE=global`; selecting it also prints what the default rule
would have chosen.

## Deploying

Each step ends by printing what it produced, where it goes and what to update. In short:

* **Baked headers** (`CATripletDNNWeights_*.h`, `CATrackDNNWeights_*.h`): written in place by
  the step, then `scram b code-format && scram b -j`. Never build while a `cmsRun` job is
  running -- the shared libraries are replaced in place and running jobs crash. `code-format`
  reformats a freshly baked header, so its checksum changes while its numbers do not; compare
  banks with `compare_track_dnn_banks.py`, not by checksum.
* **Selector models** (`<arm>_tree<N>_<tag>_<date>.bin`): staged in `--work` with today's date
  (`MODEL_NAME` overrides it, `TAG` the tag), then copied into
  `RecoTracker/FinalTrackSelectors/data/PixelTrackTorchHighPuritySelector/` and named in the
  module's configuration. `--deploy` does the copy; editing the configuration is always manual.
  No rebuild. Keeping the date in the name is what makes a model swap visible in the
  configuration diff.

The configurations to edit are in `HLTrigger/Configuration/python/HLT_75e33/modules/`:
`hltPhase2PixelTracksSoAWithStubs_cfi.py` and
`hltPhase2PixelTrackTorchHighPuritySelector_cfi.py` (upstream label; the prompt forest is a
`phase2CAStubs` `toReplaceWith` on it).

## A full pass

```bash
W=/somewhere/work
EVENTS=/path/to/reconstructible/events      # directory, list, or file of paths

# 1. triplet gate  (uncomment CA_TRIPLET_DUMP and rebuild first)
#    2000 events covers the default scoring range [1400,2000) the trainer windows per file;
#    with fewer, retrain_triplet_gate.sh scales the train/score windows to the dataset.
./retrain_prompt.sh triplet --work $W --input $EVENTS --events 2000
#    comment CA_TRIPLET_DUMP out again, then:
scram b code-format && scram b -j

# 2. track DNN  (its dataset is produced under the new gate)
./retrain_prompt.sh gate --work $W --input $EVENTS --events 800
scram b code-format && scram b -j

# 3. final selector  (its dataset is produced under the new track DNN)
./retrain_prompt.sh hp --work $W --input $EVENTS --events 1000
#    copy the staged prompt_tree31_wp_<date>.bin, edit the configuration (model; scoreThreshold
#    after an in-situ scan)
```

`--dry-run` prints every command a step would run without running it. To produce a dataset
without training, use `dump`:

```bash
./retrain_prompt.sh dump --for gate --work $W --input $EVENTS
```

Datasets, caches, models, plots and logs all go to `--work`; nothing is written into the source
tree except the two baked headers, which is where they belong.

The entry script refuses to start next to a running `cmsRun` job, and the heavy passes are
watched against the memory limit of the machine (on the anonymous memory of the control group,
not its total, most of which is reclaimable file cache).

## Comparing model families (prompt)

`./retrain_prompt.sh hp --class both` (or `--class forest`, `--class mlp`) runs a comparison
instead of the deploy path: four candidates trained on one 27-feature cache and one event split
(`prompt_hp_bakeoff.py`), ranked by fake rejection at one recall:

| candidate | family | features |
|---|---|---|
| `mlp17` | neural network, four hidden layers of 64 | 17 fit and covariance |
| `mlp27` | neural network, same shape | 27 -- the same inputs as the forest, so the table separates "a forest is better" from "more features are better" |
| `forest17` | gradient-boosted forest, depth 8 | 17 |
| `forest27` | gradient-boosted forest, depth 8 | 27: the 17 plus 10 hit and stub features |

None of the four is the 31-feature family the chain runs, so the winner is staged for in-situ
checks only, and deploying it is a change of model, to be validated as one.

## Files here

| file | role |
|---|---|
| `retrain_prompt.sh` | entry point for the prompt iteration (triplet gate, track DNN, final selector) |
| `retrain_common.sh` | shared option parsing, environment, dataset dumps, deployed-value lookups, the shared part of the selector training, reports |
| `retrain_triplet_gate.sh` | the `triplet` step: train, score on a disjoint event range, bake the header |
| `retrain_track_dnn.sh` | the `gate` step: train, compare against the bank in use, bake the header |
| `make_base_config.sh` | generate the base reconstruction configuration the dumps run on top of |
| `trackNano_config.py` | reconstruction job writing the track-level dataset (features + truth) |
| `triplet_dump_cfg.py` | reconstruction job writing the per-triplet dataset (needs the dump build) |
| `train_triplet_dnn.py` | trainer for the per-triplet gate, including the scoring tables and the header export |
| `compare_track_dnn_banks.py` | score a baked track-DNN header by reproducing its forward pass, and compare it with another bank or with the model it was baked from |
| `nano_loader.py` | build feature caches from the track-level datasets: `cache-arm` (31 features); `cache`, `cache-ext`, `cache-prompt`, `cache-prompt27` serve the comparison and studies |
| `train_merged_forest.py` | the final-selector trainer: recipe, working-point rules, compact export and read-back |
| `make_profile.py` | measure a deployed selector on a cache, bin by bin, for the `profile` working-point rule |
| `read_compact_tree.py` | read a compact `.bin` and score rows with it, the way the selector kernel does |
| `export_compact_tree.py` | write a booster as the compact binary the selector loads (used by the trainer) |
| `prompt_hp_bakeoff.py`, `build_tree_model.py`, `train_prompt_hp_nano.py`, `nano_train_utils.py` | the prompt model-family comparison: the four candidates, their forest and neural-network trainers, shared helpers |
| `CATrackFeaturesTableProducer.cc`, `TripletFeaturesTableProducer.cc`, `HitTruthTableProducer.cc`, `TrackerLayerId.h` | the table producers the two dump configurations use; built with the package, not run from here |

## Requirements

The CMSSW environment is enough for every step: the entry script sources it itself, and it
carries everything the trainers import. Train the selectors in that environment and no other:
the deployed forests were trained with its xgboost (1.7.5), and a different xgboost version
gives a different tree count and a different rejection at the same recall.
