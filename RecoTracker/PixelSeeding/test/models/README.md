# Retraining the pixel-track machine-learned models

Everything needed to retrain, check and deploy the three models of the pixel + outer-tracker
vector hit track reconstruction (`hltPhase2PixelTracksSoA` under `phase2CAVectorHits`). One entry
point does the work:

```bash
./retrain_prompt.sh <step> [options]
```

Run it with `--help` for the full list of options. Every step prints each command before running
it, and `--dry-run` prints them without running anything.

## How to run a retrain

The three models are trained one after the other, and each one is compiled in (or loaded) before
the dataset of the next is produced. A complete pass is therefore three rounds of *produce the
dataset, train, deploy*:

```bash
cd $CMSSW_BASE/src/RecoTracker/PixelSeeding/test/models
W=/somewhere/with/space/work                       # datasets, models, logs; created if missing
S="ttbarPU displacedPU qcdPU bsPU zpPU"            # the samples (this list is the default)
IN="ttbarPU=/data/ttbar,displacedPU=/data/susy,qcdPU=/data/qcd,bsPU=/data/bs,zpPU=/data/zp"

# 1. per-triplet gate ------------------------------------------------------------------------
#    switch the triplet dump on: uncomment '#define CA_TRIPLET_DUMP' in
#    RecoTracker/PixelSeeding/plugins/alpaka/CATripletDumpMacro.h, then build
scram b -j
./retrain_prompt.sh dump    --for triplet --work $W --sample "$S" --input "$IN" --events 2000
./retrain_prompt.sh triplet --work $W --sample "$S"          # writes CATripletDNNWeights_prompt.h
#    switch the dump off again (comment the #define out), then deploy the new gate:
scram b code-format && scram b -j

# 2. track DNN (its dataset is produced under the new gate) ----------------------------------
./retrain_prompt.sh dump --for gate --work $W --sample "$S" --input "$IN" --events 800
./retrain_prompt.sh gate --work $W --sample "$S"             # writes CATrackDNNWeights_prompt.h
scram b code-format && scram b -j

# 3. final high-purity selector (its dataset is produced under the new track DNN) ------------
./retrain_prompt.sh dump --for hp --work $W --sample "$S" --input "$IN" --events 1000
./retrain_prompt.sh hp   --work $W --sample "$S" --deploy    # stages and copies prompt_tree31_wp_<date>.bin
#    then name the new file in hltPhase2PixelTrackTorchHighPuritySelector_cfi.py and choose
#    scoreThreshold by running the track validation (see "Working points")
```

Things to know before starting:

* **Inputs** are GEN-SIM-DIGI-RAW events of the Phase-2 detector; each sample name needs its own
  files (see "Samples and inputs"). The event counts above are the ones the defaults are sized
  for; any count works, and a small one (a few tens of events of one sample) is enough to check
  that the whole chain runs.
* **A training step produces its own dataset** when it is not in `--work` yet and `--input` is
  given, so `./retrain_prompt.sh gate --work $W --input ...` is steps "dump" and "gate" in one.
  The explicit `dump` is used above because the event count is given per dataset.
* **The datasets of steps 2 and 3 must come from the production build.** In a build with
  `CA_TRIPLET_DUMP` on, the triplet gate's reject is compiled out, so the tracks are not the ones
  the production chain builds; the script checks the switch and stops if it is in the wrong
  state for the dataset asked for.
* **Never build while a `cmsRun` job is running**: the shared libraries are replaced in place and
  running jobs crash. The entry script also refuses to start next to a running `cmsRun`.
* **The pass rebuilds three times.** Building the alpaka code only for the GPU of the local
  machine makes each build several times shorter:
  `scram b enable-gpus:native && scram b clean && scram b -j` before starting, and
  `scram b enable-cuda:reset && scram b enable-rocm:reset && scram b clean && scram b -j` at the
  end to get all backends back.
* **Training runs on a GPU** by default (`--device cuda:0`); `--device` selects another torch
  device. The forest of step 3 is trained by xgboost on the CPU (`XGB_THREADS`, default 16).

Datasets, caches, models, plots and logs all go to `--work`. Nothing is written into the source
tree except the two baked headers, and with `--deploy` the selector file.

## The models

| step | model | where it runs | deployed as | to deploy |
|---|---|---|---|---|
| `triplet` | per-triplet gate | inside the track-building kernel | `plugins/alpaka/CATripletDNNWeights_prompt.h` | rebuild |
| `gate` | track DNN (the loose -> tight promotion) | inside the track classification kernel | `plugins/alpaka/CATrackDNNWeights_prompt.h` | rebuild |
| `hp` | final high-purity selector | a separate module after the reconstruction | `RecoTracker/FinalTrackSelectors/data/PixelTrackTorchHighPuritySelector/prompt_tree31_<tag>_<date>.bin` | copy the file |

The first two models are compiled into the kernels, so deploying them means rebuilding. The
selector is read at run time, so deploying it means copying a file and naming it in the module's
configuration.

What each step does:

* `dump` reconstructs events and writes a training dataset (one row per track, or per triplet,
  with the truth). `--for triplet`, `--for gate` and `--for hp` choose the dataset.
* `triplet` trains the per-triplet gate on the triplet dataset, measures the gate it replaces on
  the same held-out events, chooses the threshold and bakes the header.
* `gate` trains the track DNN on the `loose` dataset, compares it with the bank in use, chooses
  the threshold and bakes the header.
* `hp` builds a feature cache from the `tight` dataset, trains the forest, exports it in the
  compact binary the selector loads and stages it in `--work`. `--deploy` also copies it into the
  data directory; the configuration is edited by hand, as printed at the end of the step.
* `all` runs `triplet` and prints the remaining steps.

### Order

**`triplet` -> `gate` -> `hp`.** Every model decides on the population the previous one left
behind, so each dataset must be produced with the upstream models deployed and compiled in:

* the track DNN trains on quality **`loose`** tracks with the deployed track DNN switched off at
  dump time, because that is the population it decides on, before it decided;
* the final selector trains on quality **`tight`** tracks, which are the ones the deployed track
  DNN promoted, so that dump runs the full chain;
* produce each step's dataset only after the previous step is deployed **and built**.

**After any change to the track fit** (fit code, magnetic field, material description): re-run
the `gate` and `hp` steps. Their features are fit outputs, and a model trained on an older fit is
quietly mismatched; `compare_track_dnn_banks.py` scores the baked track-DNN header directly and
makes that visible.

### The features and their order

Each model identifies a feature by its position, and the same order is written in three places
that must agree: the C++ struct whose `asArray()` feeds the model, the columns of the dataset,
and the feature list of the trainer.

| model | C++ | dataset columns | trainer |
|---|---|---|---|
| triplet gate | `CATripletFeatures::asArray()` (`CATripletDNN.h`) | `TrkPromptTriplet_*` (18 raw quantities) | `BASE_FEATURES + DERIVED` in `train_triplet_dnn.py`, which recomputes the 11 derived ones with the formulas of `CATripletCuts.h` |
| track DNN | `caTrackFeatures::Features::asArray()` (`CATrackFeatures.h`) | `TrkPromptCA_*`, written through the same `fill()` | `FEATS` in `train_disp_nano.py` |
| final selector | the two blocks of `PixelTrackFeaturesSoA.h`, fit block first | `TrkPromptFull_*` (fit block) and `TrkPromptCA_*` (hit block) | `FEATS31` in `nano_loader.py` |

A change of any of these lists needs the other two changed with it and a retrain.

## Samples and inputs

Every model trains on the same five PU200 samples, with the **same number of events from each**:

| tag | sample | what it brings |
|---|---|---|
| `ttbarPU` | top pairs | the bulk: dense jets, the whole momentum range, b-hadron displacement |
| `displacedPU` | long-lived displaced decays | tracks from centimetres out, which nothing else populates |
| `qcdPU` | jets | combinatorics: the highest triplet load per event |
| `bsPU` | a soft dimuon resonance | low-momentum tracks, and a second displacement scale |
| `zpPU` | a heavy dimuon resonance | high-momentum tracks, where the sparse tails live |

Equal event counts are how equal weight per sample is implemented; the trainers do not re-weight
per sample afterwards. The list is the default, so a retrain with no `--sample` trains on all
five; `--sample "..."` (or `SAMPLES`) overrides it. **Give the same `--sample` list to a `dump`
and to the training step that reads it**: the training step looks for one dataset per name.

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
Every training step is given all the per-sample datasets at once and trains on their union; the
files are named `trackNano_<sample>_<quality>.root` and `tripletNano_<sample>.root` in `--work`.
A step with neither `--input` nor `--dataset` looks for those names in `--work`; if they are not
there it stops and says so.

For a sense of scale, 20 TTbar PU200 events give about 4 million triplets (a 200 MB dataset) and
about 40 000 `tight` tracks.

**The base reconstruction configuration.** The two dump configurations (`trackNano_config.py`,
`triplet_dump_cfg.py`) only add tables to an existing reconstruction process; they do not build
one. With `--input` given, the entry script generates that base configuration once into the
working directory with `make_base_config.sh`, which is an ordinary `cmsDriver.py` call:

```
cmsDriver.py step2 -s L1P2GT,HLT:75e33_trackingOnly --processName HLTX \
  --conditions auto:phase2_realistic_T35 --geometry ExtendedRun4D121 --era Phase2C22I13M9 \
  --procModifiers ngtScouting --eventcontent FEVTDEBUGHLT --datatier GEN-SIM-DIGI-RAW \
  --filein <files> -n <events> --no_exec \
  '--output={}' \
  '--inputCommands=keep *, drop *_hlt*_*_HLT, drop triggerTriggerFilterObjectWithRefs_l1t*_*_HLT' \
  --python_filename <out>
```

* `ngtScouting`: under it the HLT menu runs the vector hit pixel chain exactly as under
  `phase2CAVectorHits` (the menu applies the same replacements under either modifier), and the
  iterative tracking behind it becomes a pass-through over the pixel tracks, so a dump is cheap.
  `make_base_config.sh --modifiers` sets a different list verbatim.
* `--output={}` keeps ConfigBuilder from adding a `PoolOutputModule` and its `output_step`
  EndPath: the dump configurations write their own NanoAOD file, and without it every dump job
  would also write a full GEN-SIM-DIGI-RAW copy of its input.
* `--inputCommands` drops the products of an earlier HLT process that RelVal inputs carry, so
  unqualified `InputTag`s cannot resolve against them; `make_base_config.sh --input-commands ''`
  omits it.

Use `--chassis <module>` (the `chassis=<module>` option of the two dump configurations when they
are run by hand) to dump on top of a different configuration instead; its directory must be on
`PYTHONPATH`. A chassis carries its own input file list, so it is enough on its own to produce a
dataset.

**Dataset provenance.** Every track-level dump prints one `[trackNano CHAIN-STATE]` line
recording the state it ran in: whether each in-kernel model was on, at which threshold, the
buffer sizes and the quality. A dataset is only valid for the state on that line. To dump at a
working point that is not in the configuration yet, use the `prompt*` options of
`trackNano_config.py` (`promptGate`, `promptGateOff`, `promptTrackDNN`, `promptTrackThr`,
`promptCaps`, `promptCapsFactor`); they are printed on the same line.

**Truth labels.** The track-level datasets carry two labels per track: `matched` (matched to a
TrackingParticle passing the efficiency selection) and `matchedAny` (matched to any
TrackingParticle, which is what the track validation calls "not a fake"). The trainers target
the second and quote recall on the first. `matchedAny` is written when `trackNano_config.py` runs
with `mtvLabel=1`, which the entry script always passes.

## The build switch for the triplet step

The per-triplet dump is compiled out by default. Its single switch is the last line of
`RecoTracker/PixelSeeding/plugins/alpaka/CATripletDumpMacro.h`; with it commented out the build
is the production one, which is how it must stay.

```
# before the triplet dataset: uncomment '#define CA_TRIPLET_DUMP', then
scram b -j
#   ... ./retrain_prompt.sh dump --for triplet ...
# after it: comment the line out again, then
scram b -j
```

With the switch on, the CA producer writes one row per built triplet and no longer rejects
triplets on the gate's score, so the dataset holds every triplet the cuts accepted. The dataset
also carries the score the kernel computed (`inKernelScore`), so the in-kernel evaluation can be
compared with the trainer's. The scripts check the switch and stop with instructions when it is
in the wrong state: on for the triplet dataset, off for the track-level ones. They do not edit
the header.

## Working points: what the tooling derives, and what is deployed

**Every working point is stated against the model it replaces.** A retrain moves the model and
moves the threshold with it, to the point at which nothing the old model kept is given up
anywhere, so a retrain with no options is the recipe. The two in-kernel banks are deployed at
their rule's point directly. The selector threshold gets one more step, a scan with the track
validation, because the offline scorer cannot measure the combinatorial load, the container
occupancy or the fakes handed to the next stage. `--threshold <x>` (or `THRESHOLD=<x>`,
`BAKE_THR=<x>`) bakes an explicit value in the `triplet` and `gate` steps. Each step prints the
value the configuration currently carries and where it lives.

| model | the rule the step applies | deployed | configuration parameter |
|---|---|---|---|
| triplet gate | the largest threshold at which every populated (\|tip\|, pT) group's per-track survival matches or beats the outgoing bank's, measured on the same held-out events | at this point | `tripletDNNThreshold` on the CA producer; when it is negative there (as configured), the value baked into the header applies |
| track DNN | the largest threshold at which the new bank keeps at least the outgoing bank's fraction of matched tracks in every pT, \|eta\| and \|dxy\| bin (`track_dnn_working_point.py`) | at this point | `trackDNNThreshold` on the CA producer; same fallback |
| final selector | the per-bin recall of the model the configuration loads (the `profile` rule below): a starting point | after the scan | `scoreThreshold` on the selector |

The triplet gate is the one to be careful with. A track leaves about ten real triplets behind,
so a rule stated as "keep 99 % of real triplets" keeps only about 0.99^10 ~ 90 % of the tracks,
which is why the rule works on per-track survival. The global-recall rule is available for
comparison with `THRESHOLD_RULE=global`; selecting it also prints what the default rule would
have chosen.

**Choosing the selector threshold.** Run the reconstruction with the track validation at a few
values of `scoreThreshold` around the trainer's starting point, and take the tightest one whose
efficiency still matches the configuration the run started from, in every region that matters
(not only the integrated number).

### What is deployed today

| model | file or header | threshold |
|---|---|---|
| triplet gate | `CATripletDNNWeights_prompt.h` | 7.218852e-03, baked |
| track DNN | `CATrackDNNWeights_prompt.h` | 3.414618e-02, baked |
| final selector | `prompt_tree31_wp_20260914.bin` | `scoreThreshold` 0.232 |

## The final selector in detail

The selector is a gradient-boosted forest run by `PixelTrackForestHighPuritySelector@alpaka`,
reading a compact `.bin`. Under `phase2CAVectorHits` it replaces the Torch model on the module
`hltPhase2PixelTrackTorchHighPuritySelector`, whose configuration file is
`HLTrigger/Configuration/python/HLT_75e33/modules/hltPhase2PixelTrackTorchHighPuritySelector_cfi.py`.
It reads the first 31 columns of the `PixelTrackFeaturesSoA` layout: 17 fit and covariance
columns, 10 hit and vector hit columns, then `rzChi2`, `meanVectorHitKappa`, `leverArm`, `rMax`.

The `hp` step runs this chain, exactly as the script prints it (`$W` = `--work`, `$M` = this
directory):

```bash
python3 $M/nano_loader.py cache-arm $W/prompt31_cache.npz $W/trackNano_<sample>_tight.root ... \
    --prefix TrkPrompt --label mtv
python3 $M/make_profile.py --cache $W/prompt31_cache.npz --model <the deployed .bin> \
    --threshold <its scoreThreshold> --arm prompt --split test --name deployed_<its name> \
    --out $W/prompt_reference_profile.json
python3 $M/train_merged_forest.py --cache $W/prompt31_cache.npz --out $W/prompt_forest \
    --feats 31 --label mtv --recall 0.995 --arm prompt --tag wp --date <date> --threads 16 \
    --eff-weight 4.0 --no-fp16 --ntrees 1000 --es-rounds 0 --prune-tol 0.002 \
    --wp-rule profile --wp-profile $W/prompt_reference_profile.json --wp-margin 0
# staged: $W/prompt_forest/prompt_tree31_wp_<date>.bin
```

Then update, on the `_hltPhase2PixelTrackForestHighPuritySelector` producer of the configuration
file: `model` (the new file name) and `scoreThreshold` (after the scan). `useHitFeatures` stays
at its default, `True`.

Next to the staged file the step leaves `result.json` (every metric, the working point, the
feature list and the full recipe) and `thrmap.txt` (score threshold against true-track recall,
with fake rejection). It also reads the `.bin` back through `read_compact_tree.py` and re-scores
test rows against the trained booster.

The trainer's inputs and labels, in short: a cache carries the features in the selector's column
order and two labels per track, `y_mtv` (`matchedAny`, the training target) and `y_eff`
(`matched`, the recall axis). Recall is quoted on the second, fake rejection on the tracks
matched to nothing; the true tracks that fail an efficiency cut are positives in training and on
neither axis.

### Working-point rules

`train_merged_forest.py --wp-rule` chooses how the trained scores are turned into a threshold.
The rule shapes the model, not only its cut, because the size of the forest is chosen at that
threshold. `WP_RULE=profile|uniform|global` sets it in the entry script.

* `profile` (**the default**): match or exceed a reference model bin by bin. `make_profile.py`
  measures the model the configuration loads today, at the threshold that configuration carries,
  on this run's own cache; the working point is then the threshold at which the new forest
  reaches that recall in every bin. It is the default because that is the question a retrain has
  to answer: is the new model at least as good as the one it replaces, everywhere, on the sample
  of this run. It needs the deployed file, so a cold start needs one of the other two.
* `uniform`: the largest threshold at which the true-track recall is at least `--recall` (0.995)
  in every pT, |eta| and |dxyBS| bin of the trainer's standard edges (pT 0.9/1.5/3/10/30, |eta|
  1.5/2.5, |dxyBS| 0.1/0.5/1/2/5, each axis open at the top). Bins with fewer than 100 true
  tracks do not constrain it; with no usable bin the global rule applies.
* `global`: the threshold at which the overall true-track recall is `--recall`.

### Forest size

The step fits with a large budget and no early stopping (`--ntrees 1000 --es-rounds 0`) and then
lets the data choose the size: `--prune-tol 0.002` keeps the smallest prefix of the forest whose
fake rejection at the working point, on the validation split, is within 0.002 of the best
prefix. The whole curve is in `result.json` as `size_curve`, so the choice can be read back.
`NTREES`, `ES_ROUNDS` and `PRUNE_TOL` override the three; `PRUNE_TOL=0` keeps every tree. A
different tolerance does not need a refit: `--from-full <out>/model_full.json` starts from the
saved full booster.

### What a retrain reproduces

With no options: 31 features, true high-pT tracks up-weighted (`--eff-weight 4`), full-precision
values in the exported file, the large tree budget with the size chosen by the data, and the
starting threshold taken from the per-bin profile of the model the configuration loads.

## Deploying

Each step ends by printing what it produced, where it goes and what to update. In short:

* **Baked headers** (`CATripletDNNWeights_prompt.h`, `CATrackDNNWeights_prompt.h`): written in
  place by the step, then `scram b code-format && scram b -j`. `code-format` reformats a freshly
  baked header, so its checksum changes while its numbers do not; compare banks with
  `compare_track_dnn_banks.py`, not by checksum. The copy of the header that was replaced is kept
  in `--work` (`CATrackDNNWeights_prompt.h.replaced`) by the `gate` step.
* **Selector model** (`prompt_tree31_<tag>_<date>.bin`): staged in `--work` with today's date
  (`MODEL_NAME` overrides it, `TAG` the tag), then copied into
  `RecoTracker/FinalTrackSelectors/data/PixelTrackTorchHighPuritySelector/` and named in the
  module's configuration. `--deploy` does the copy; editing the configuration is always manual.
  No rebuild. Keeping the date in the name is what makes a model swap visible in the
  configuration diff.

The configurations to edit are in `HLTrigger/Configuration/python/HLT_75e33/modules/`:
`hltPhase2PixelTracksSoAWithVectorHits_cfi.py` (the two in-kernel thresholds, if they are to be
set explicitly) and `hltPhase2PixelTrackTorchHighPuritySelector_cfi.py` (the selector).

### Checking a deployed model

After the rebuild, each model can be checked against its trainer on a freshly produced dataset:

* triplet gate: in a dump build, the `inKernelScore` column of the triplet dataset is the score
  the kernel computed, and `train_triplet_dnn.py eval --header <header>` scores the same rows
  from the header. Nothing here compares the two automatically.
* track DNN: `compare_track_dnn_banks.py <loose datasets> --bank prompt --header <header>`
  reproduces the forward pass of the header on the dataset columns.
* final selector: `read_compact_tree.py` loads a `.bin` and scores rows the way the selector
  kernel does; `make_profile.py` uses it to measure a deployed file on a cache.

## Comparing model families

`./retrain_prompt.sh hp --class both` (or `--class forest`, `--class mlp`) runs a comparison
instead of the deploy path: four candidates trained on one 27-feature cache and one event split
(`prompt_hp_bakeoff.py`), ranked by fake rejection at one recall:

| candidate | family | features |
|---|---|---|
| `mlp17` | neural network, four hidden layers of 64 | 17 fit and covariance |
| `mlp27` | neural network, same shape | 27: the same inputs as the forest, so the table separates "a forest is better" from "more features are better" |
| `forest17` | gradient-boosted forest, depth 8 | 17 |
| `forest27` | gradient-boosted forest, depth 8 | 27: the 17 plus 10 hit and vector hit features |

None of the four is the 31-feature family the chain runs, so the winner is staged for in-situ
checks only, and deploying it is a change of model, to be validated as one.

## Files here

| file | role |
|---|---|
| `retrain_prompt.sh` | entry point (triplet gate, track DNN, final selector) |
| `retrain_common.sh` | shared option parsing, environment, dataset dumps, build-switch checks, deployed-value lookups, the shared part of the selector training, reports |
| `retrain_triplet_gate.sh` | the `triplet` step: train, measure the outgoing bank, score on a disjoint event range, bake the header |
| `retrain_track_dnn.sh` | the `gate` step: train, compare against the bank in use, derive the working point, bake the header |
| `make_base_config.sh` | generate the base reconstruction configuration the dumps run on top of |
| `trackNano_config.py` | reconstruction job writing the track-level dataset (features + truth) |
| `triplet_dump_cfg.py` | reconstruction job writing the per-triplet dataset (needs the dump build) |
| `train_triplet_dnn.py` | trainer of the per-triplet gate, including the scoring tables and the header export |
| `train_disp_nano.py` | trainer of the track DNN (`train`) and its header export (`bake`), for the iteration `--bank` names |
| `compare_track_dnn_banks.py` | score a baked track-DNN header by reproducing its forward pass, and compare it with another bank or with the model it was baked from |
| `track_dnn_working_point.py` | the track DNN's working point: the largest threshold holding the outgoing bank's recall in every bin |
| `nano_loader.py` | build feature caches from the track-level datasets: `cache-arm` (31 features); `cache`, `cache-ext`, `cache-prompt`, `cache-prompt27` serve the comparison and studies |
| `train_merged_forest.py` | the final-selector trainer: recipe, working-point rules, compact export and read-back |
| `make_profile.py` | measure a deployed selector on a cache, bin by bin, for the `profile` working-point rule |
| `read_compact_tree.py` | read a compact `.bin` and score rows with it, the way the selector kernel does |
| `export_compact_tree.py` | write a booster as the compact binary the selector loads (used by the trainer) |
| `prompt_hp_bakeoff.py`, `build_tree_model.py`, `train_prompt_hp_nano.py`, `nano_train_utils.py` | the model-family comparison: the four candidates, their forest and neural-network trainers, shared helpers |
| `CATrackFeaturesTableProducer.cc`, `TripletFeaturesTableProducer.cc`, `HitTruthTableProducer.cc`, `TrackerLayerId.h` | the table producers the two dump configurations use; built with the package (`test/BuildFile.xml`), not run from here |

## Running the dump configurations by hand

The two dump configurations take `cmsRun` command-line options (`VarParsing`); the entry script
passes them, and `cmsRun <configuration> help` lists them with their defaults. For example:

```bash
cmsRun trackNano_config.py chassis=base_reco_cfg inputFiles=a.root,b.root maxEvents=100 \
    minQuality=loose mtvLabel=1 promptTrackDNN=0 promptCaps=big threads=8 outputFile=loose.root
cmsRun triplet_dump_cfg.py chassis=base_reco_cfg inputFiles=a.root maxEvents=100 iteration=prompt
```

Without `inputFiles` a dump reads the input files of its base configuration.

## Environment variables of the scripts

The scripts read these variables when they are set; each default is the value in use. The
command-line options of `retrain_prompt.sh --help` cover the rest.

| variable | default | meaning |
|---|---|---|
| `WORKDIR` | none | working directory, as `--work` |
| `SAMPLES` | `ttbarPU displacedPU qcdPU bsPU zpPU` | sample names, as `--sample` |
| `THREADS` | 8 | reconstruction threads of the dumps, as `--threads` |
| `DEVICE` | `cuda:0` | torch device of the trainers, as `--device` |
| `THRESHOLD` | none (the step's rule) | threshold baked by `triplet` and `gate`, as `--threshold` |
| `MEM_GUARD_GB` | 300 | stop a heavy pass when the anonymous memory of the control group exceeds this |
| `RT_ALLOW_CONCURRENT_CMSRUN` | unset | set to start next to a running `cmsRun` |
| `RT_LABEL` | `mtv` | truth label of the track-DNN trainer, as `--label` |
| `RT_DISP_CAPS_FACTOR` | 4 | enlargement of the displaced CA's buffers in a displaced-arm track dump (`dispCapsFactor`) |
| `MODEL_CLASS` | `forest31` | model family of the `hp` step, as `--class` |
| `WP_RULE` | `profile` | working-point rule of the `hp` step: `profile`, `uniform` or `global` |
| `NTREES` | 1000 (800 in the comparison) | tree budget of the forest |
| `ES_ROUNDS` | 0 | early-stopping rounds of the forest (0 = none) |
| `PRUNE_TOL` | 0.002 | fake-rejection tolerance that chooses the forest size (0 = keep every tree) |
| `RECALL` | 0.995 | true-track recall of the `uniform` and `global` rules |
| `EFF_WEIGHT` | 4.0 | weight of the true high-pT tracks in the forest training |
| `LABEL` | `mtv` | truth label of the forest training |
| `TAG` | `wp` | tag in the staged file name `prompt_tree31_<tag>_<date>.bin` |
| `MODEL_NAME` | today's date | date in the staged file name |
| `XGB_THREADS` | 16 | xgboost threads |
| `TRAIN_EXTRA` | none | further `train_merged_forest.py` options |
| `ARMS` | `mlp17,mlp27,forest17,forest27` | candidates of the comparison; setting it selects the comparison |
| `SELECT_RECALL` | 0.99 | recall at which the comparison ranks its candidates |
| `EPOCHS` | 250 in the comparison, 40 in the `triplet` step | maximum training epochs |
| `DEPTH` | 8 | tree depth of the comparison forests |
| `THR_RECALL` | 0.99 | recall of the threshold a neural-network winner of the comparison is exported with |
| `MAX_ROWS` | all | subsample of the cache used by the comparison |
| `TRAIN_EV` | 1000 | `triplet` step: training events `[0, TRAIN_EV)` of each dataset |
| `TEST_SKIP`, `TEST_EV` | 1400, 600 | `triplet` step: scoring events `[TEST_SKIP, TEST_SKIP + TEST_EV)`; with none of the three set, the window is derived from a smaller dataset |
| `VARIANT` | `bal_up05` | `triplet` step: training recipe |
| `WORKERS` | sized from cores and memory | `triplet` step: ingestion workers |
| `THRESHOLD_RULE` | `worstgroup-survival` | `triplet` step: threshold rule (`global` for the global-recall rule) |
| `SURVIVAL_FLOOR` | 0.90 | `triplet` step: per-track survival held when there is no reference to match |
| `REF_SURVIVAL` | measured on the outgoing header | `triplet` step: json of a reference model to match group by group |
| `BAKE_THR` | none | `triplet` step: explicit threshold to bake, as `--threshold` |
| `EXTRA` | none | `triplet` step: further trainer options |
| `OUTDIR`, `LOGDIR` | `<work>/plots_tripletDNN` | `triplet` step: plots and logs |
| `NAME` | `<bank>_stage1_12f` | `gate` step: tag of the trained model's files |

## Requirements

The CMSSW environment is enough for every step: the entry script sources it itself, and it
carries everything the trainers import (torch, xgboost, uproot, pandas, scikit-learn). Train the
selector in that environment and no other: the deployed forest was trained with its xgboost
(1.7.5), and a different xgboost version gives a different tree count and a different rejection
at the same recall.

The heavy passes are watched against the memory limit of the machine (`MEM_GUARD_GB`, default
300), on the anonymous memory of the control group rather than its total, most of which is
reclaimable file cache.
