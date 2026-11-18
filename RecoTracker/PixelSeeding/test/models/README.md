# Retraining the pixel-track machine-learned models

Everything needed to retrain, check and deploy the three models of the pixel + outer-tracker
vector hit track reconstruction (`hltPhase2PixelTracksSoA` under `phase2CAVectorHits`). One entry
point does the work:

```bash
./retrain_prompt.sh <step> [options]
```

Run it with `--help` for the full list of options. Every step prints each command before running
it, and `--dry-run` prints them without running anything.

The same entry point also retrains the final high-purity selector of the **baseline** pixel chain
(no process modifier), a Torch network; see "The baseline chain's network" below.

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
#    then name the new file in hltPhase2PixelTrackForestHighPuritySelector_cfi.py, choose
#    scoreThreshold by running the track validation (see "Working points"), and add the file
#    to the cms-data repository RecoTracker-FinalTrackSelectors (see "Deploying")
```

Things to know before starting:

* **Inputs** are GEN-SIM-DIGI-RAW events of the Phase-2 detector; each sample name needs its own
  files (see "Samples and inputs"). The event counts above are the ones the defaults are sized
  for; any count works, and a small one (a few tens of events of one sample) is enough to check
  that the whole chain runs.
* **A training step produces its own dataset** when it is not in `--work` yet and `--input` is
  given, so `./retrain_prompt.sh gate --work $W --input ...` is steps "dump" and "gate" in one.
  A dataset already in `--work` is used as it is, so after a rebuild a dataset is produced again
  with `dump`, which always rewrites it; the explicit `dump` above also gives the event count per
  dataset.
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
| `hp` | final high-purity selector | a separate module after the reconstruction | `RecoTracker/FinalTrackSelectors/data/itaca_selectors/prompt_tree31_<tag>_<date>.bin` | copy the file |
| `hp --chain baseline` | final high-purity selector of the baseline chain (Torch network) | a separate module after the reconstruction | `RecoTracker/FinalTrackSelectors/data/PixelTrackTorchHighPuritySelector/pixel_track_classifier_FP16_<date>.pt` | copy the file |

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
| baseline network | the fit block of `PixelTrackFeaturesSoA.h` | `TrkPromptFull_*` | `FEATS17` in `nano_loader.py` |

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
scram b code-format && scram b -j
```

With the switch on, the CA producer writes one row per built triplet and does not reject
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
| final selector, and the baseline network | one threshold at 99.5 % recall of the non-duplicate efficiency tracks ("Working point" below), set on the validation split | after a check with the track validation | `scoreThreshold` on the selector |

The triplet gate is the one to be careful with. A track leaves about ten real triplets behind,
so a rule stated as "keep 99 % of real triplets" keeps only about 0.99^10 ~ 90 % of the tracks,
which is why the rule works on per-track survival. The global-recall rule is available for
comparison with `THRESHOLD_RULE=global`; selecting it also prints what the default rule would
have chosen.

**Checking the selector threshold.** Run the reconstruction with the track validation on
events that the training did not use, with the new model at the trainer's threshold next to
the configuration the run started from, and compare efficiency and fake rate against pT, eta
and pileup. `RECALL` moves the working point if the efficiency has to be traded differently.

### What is deployed today

| model | file or header | threshold |
|---|---|---|
| triplet gate | `CATripletDNNWeights_prompt.h` | 7.218852e-03, baked |
| track DNN | `CATrackDNNWeights_prompt.h` | 3.414618e-02, baked |
| final selector | `prompt_tree31_wp_20261117.bin` (150 trees) | `scoreThreshold` 0.520 |
| baseline network | `pixel_track_classifier_FP16_20261117.pt` | `scoreThreshold` 0.700 |

A selector model is tied to the fit of the tracks it was trained on. Use each model only with
the fit it was trained on, and retrain both models after any change of the fit or of the CA.

## The final selector in detail

The selector is a gradient-boosted forest run by `PixelTrackForestHighPuritySelector@alpaka`,
reading a compact `.bin` from `RecoTracker/FinalTrackSelectors/data/itaca_selectors/`. Its
configuration file is
`HLTrigger/Configuration/python/HLT_75e33/modules/hltPhase2PixelTrackForestHighPuritySelector_cfi.py`.
Under `phase2CAVectorHits` and `ngtScouting` the pixel tracking sequence
(`HLTPhase2PixelTracksAndVerticesSequence_cfi.py`) puts it in place of the Torch model, with the
module label `hltPhase2PixelTrackTorchHighPuritySelector`.
It reads the first 31 columns of the `PixelTrackFeaturesSoA` layout: 17 fit and covariance
columns, 10 hit and vector hit columns, then `rzChi2`, `meanVectorHitKappa`, `leverArm`, `rMax`.

The `hp` step runs this chain, exactly as the script prints it (`$W` = `--work`, `$M` = this
directory):

```bash
python3 $M/nano_loader.py cache-arm $W/prompt31_cache.npz $W/trackNano_<sample>_tight.root ... \
    --prefix TrkPrompt --label mtv
python3 $M/train_merged_forest.py --cache $W/prompt31_cache.npz --out $W/prompt_forest \
    --feats 31 --label mtv --arm prompt --tag wp --date <date> --threads 16 \
    --recall 0.995 --pos-scale 4 --region-pt-weight 4 --eff-weight 8 \
    --recall-axis A_nodup --wp-rule global --tailw 0 --no-fp16 \
    --ntrees 150 --es-rounds 0 --prune-tol 0 --dup-as-fake 1
# staged: $W/prompt_forest/prompt_tree31_wp_<date>.bin
```

Then update, on the `_hltPhase2PixelTrackForestHighPuritySelector` producer of the configuration
file: `model` (the new file name) and `scoreThreshold` (after the scan). `useHitFeatures` stays
`True`.

Next to the staged file the step leaves `result.json` (every metric, the working point, the
feature list and the full recipe) and `thrmap.txt` (score threshold against true-track recall,
with fake rejection). It also reads the `.bin` back through `read_compact_tree.py` and re-scores
test rows against the trained booster.

The trainer's inputs and labels, in short: a cache carries the features in the selector's column
order and two labels per track, `y_mtv` (`matchedAny`, the training target) and `y_eff`
(`matched`, the recall axis). Recall is quoted on the second, fake rejection on the tracks
matched to nothing; the true tracks that fail an efficiency cut are positives in training and on
neither axis.

### Working point

One `scoreThreshold` per model: the threshold at which the recall of the non-duplicate
efficiency tracks is `RECALL` (0.995), set on the validation split. One threshold removes almost
as many fakes as the best possible set of per-pT thresholds at the same overall recall (0.3 to
0.7 points of rejection less), and it needs no change of the selector. A recall floor in every
pT, |eta| and |dxyBS| bin is set by the hardest bins (|dxyBS| 0.1-0.5 cm, pT above 10 GeV), where
separation is weakest: at 99.5 % it keeps 4 to 8 times more fakes above 10 GeV. Matching the
replaced model bin by bin keeps that model's operating point in every bin and gave the smallest
gains. `train_merged_forest.py --wp-rule` still has those two rules for studies; the entry script
always uses `global`, and `train_torch_hp_dnn.py` has only that one.

### Forest size

The step fits `EXPORT_TREES` trees (150, the deployed size) directly. Size against cost and gain
(measured 2026-11-17 on TTbar PU200): the selector costs about 14 ms per event on the serial
backend plus 0.09 ms per tree, and 0.08 / 0.16 / 0.42 ms per event of GPU kernels for 150 / 300 /
700 trees; each doubling of the size removed about 12 % of the remaining fakes.

A saved full forest is cut without a refit with `train_merged_forest.py --from-full
<out>/model_full.json --export-trees <n>`: the first n trees of a boosted forest are the forest a
fit of n rounds gives.

### The training recipe (both model families)

The forest and the baseline network are trained the same way:

| step | what it does | option (environment variable) |
|---|---|---|
| target | a track matched to a TrackingParticle is true (`matchedAny`) | always |
| duplicates | a duplicate (a second track of a particle another track found) is a negative of the loss | `--dup-as-fake 1` (`DUP_AS_FAKE`, 0 = off) |
| class balance | true tracks weighted by fake fraction / (1 - fake fraction) | always |
| true-track scale | every true track x 4 | `--pos-scale 4` (`POS_SCALE`) |
| pT region weight | every track, true and fake, x (1 + 4 log10(max(pT, 0.9 GeV) / 0.9 GeV)) | `--region-pt-weight 4` (`REGION_PT_WEIGHT`) |
| efficiency-track pT weight | the non-duplicate efficiency tracks x (1 + A log10(...)), A set per chain (below) | `--eff-weight A` (`EFF_WEIGHT`) |
| forest second pass | off | `--tailw 0` (always) |
| forest size | 150 trees, the deployed size | `--ntrees 150` (`EXPORT_TREES`) |
| working point | one threshold at 99.5 % recall of the **non-duplicate** efficiency tracks | `--wp-rule global --recall 0.995 --recall-axis A_nodup` (`RECALL`; the network trainer always uses this rule and axis) |

**The recall axis.** The track validation counts a particle as found once, so a duplicate track
adds no efficiency. Duplicates score low; counting them in the recall pulls the threshold down
and keeps fakes for nothing (19 % more fakes for the vector hit forest in the study that set
this rule). The recall is therefore measured on the non-duplicate efficiency tracks.

**Duplicates as fakes.** Training duplicates as negatives raises the duplicate rejection at the
working point from about 14 % to 17-20 % and costs no fake rejection and no recall. Most
duplicates are good tracks of a particle that already has one, so the selector cannot remove
many: the duplicate rate is set by the CA's cleaning.

**The efficiency-track pT weight `EFF_WEIGHT`, per chain.** The region weight alone lowers the
high-pT recall of the vector hit forest (about 1 point below the model it replaced, above
10 GeV). A pT weight on the true tracks only restores it, at the cost of some high-pT fakes. The
rule: the smallest A at which the recall at 10-30 GeV and above 30 GeV, at the working point,
reaches the deployed model's within statistics, or stops rising if it never does, measured with
the deployed size (the per-pT tables of `result.json`). Values (2026-11-17, `RT_EFF_WEIGHT_VH`
and `RT_EFF_WEIGHT_BASELINE` in `retrain_common.sh`): vector hit forest 8 (the plateau, 0.2-0.4
points below the old forest), baseline network 1. Re-measure them after a change of the fit or
of the CA, which change how the model scores high-pT tracks.

How the rest was chosen: the six combinations of pT weight (none, efficiency tracks only, every
track) and true-track scale (x1, x4) were trained for both model families and ranked by fake
rejection at 99.5 % recall on the validation split. The x4 scale gives the forest about 0.5
points of rejection; the forest's former two-pass tail re-weight had a band that covered every
true track, so it was only a uniform x4, and the x4 scale reproduces that forest exactly.
Training weights flat in pT, or in pT x |eta|, within each class (the reweighting of the CMS
flavour taggers) were worse for both families: 0.8 to 1.4 points less fake rejection for the
forest, 0.4 to 1.2 for the network, and no better separation at high pT. More high-pT QCD in the
training sample helped at high pT.

With no options a retrain reproduces: 31 features (17 for the network) and this recipe,
full-precision values in the exported forest, and the threshold at 99.5 % recall of the
non-duplicate efficiency tracks.

## The baseline chain's network

Without a process modifier the chain runs `PixelTrackTorchHighPuritySelector@alpaka` on the
module `hltPhase2PixelTrackTorchHighPuritySelector`, with a TorchScript file. The baseline chain
has no vector hits and no in-kernel models, so its retrain is one round:

```bash
W=/somewhere/with/space/work_baseline     # its own directory: the datasets differ from the
                                          # vector hit chain's
./retrain_prompt.sh dump --chain baseline --for hp --work $W --sample "$S" --input "$IN" --events 1000
./retrain_prompt.sh hp   --chain baseline --work $W --sample "$S" --deploy
#    then name the new file and its scoreThreshold on hltPhase2PixelTrackTorchHighPuritySelector
#    (hltPhase2PixelTrackTorchHighPuritySelector_cfi.py)
```

* **Dataset.** `--chain baseline` generates its own base configuration,
  `base_reco_baseline_cfg.py` (`make_base_config.sh --modifiers none`). On a chain without vector
  hits `trackNano_config.py` leaves out the `TrkPromptCA` tables and converts the tracks with a
  copy of the chain's own `hltPhase2PixelTracks`. The trackingOnly menu without modifiers runs the
  full iterative tracking behind the pixel tracks, so a baseline dump is slower than a vector hit
  one.
* **Model.** `train_torch_hp_dnn.py` holds the network as the deployed file has it: the 17 fit
  features; log10(|x| + 1e-8) on ten of them, a clamp on eight, then standardisation; four hidden
  layers of 64 with LayerNorm, ReLU and dropout 0.05; a sigmoid output of shape [N, 1]; FP16
  parameters. The step first runs `train_torch_hp_dnn.py check`, which loads the deployed weights
  into these classes and requires the TorchScript code of every submodule and the scores to be
  identical to the deployed file. A retrain recomputes the numbers only: the clamp limits (the
  0.1 % and 99.9 % quantiles of each clamped feature), the mean and spread, and the weights.
* **Training.** Target `matchedAny`, the recipe of "The training recipe" above (the trainer's
  own defaults), Adam with early stopping on the validation fake rejection at `RECALL`; the cache
  is `nano_loader.py cache-arm --feats 17`.
* **Working point.** One threshold at `RECALL` of the non-duplicate efficiency tracks, on the test
  split, with the exported FP16 file scored in half precision as the producer does. It is in
  `baseline_dnn/result.json` with its per-bin tables, next to the deployed file at its own
  threshold and at the same recall.

## Deploying

Each step ends by printing what it produced, where it goes and what to update. In short:

* **Baked headers** (`CATripletDNNWeights_prompt.h`, `CATrackDNNWeights_prompt.h`): written in
  place by the step, then `scram b code-format && scram b -j`. `code-format` reformats a freshly
  baked header, so its checksum changes while its numbers do not; compare banks with
  `compare_track_dnn_banks.py`, not by checksum. The copy of the header that was replaced is kept
  in `--work` (`CATrackDNNWeights_prompt.h.replaced`) by the `gate` step.
* **Selector model** (`prompt_tree31_<tag>_<date>.bin`): staged in `--work` with today's date
  (`MODEL_NAME` overrides it, `TAG` the tag), then copied into
  `RecoTracker/FinalTrackSelectors/data/itaca_selectors/` (a network of the baseline chain:
  `RecoTracker/FinalTrackSelectors/data/PixelTrackTorchHighPuritySelector/`) and named in the
  module's configuration. `--deploy` does the copy; editing the configuration is always manual.
  No rebuild. Keeping the date in the name is what makes a model swap visible in the
  configuration diff. A release takes that data directory from the cms-data repository
  `RecoTracker-FinalTrackSelectors`, so the new file is also added there; the configuration
  change goes into CMSSW.

The configurations to edit are in `HLTrigger/Configuration/python/HLT_75e33/modules/`:
`hltPhase2PixelTracksSoAWithVectorHits_cfi.py` (the two in-kernel thresholds, if they are to be
set explicitly), `hltPhase2PixelTrackForestHighPuritySelector_cfi.py` (the forest of the vector
hit chain) and `hltPhase2PixelTrackTorchHighPuritySelector_cfi.py` (the network of the baseline
chain).

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
| `train_disp_nano.py` | trainer of the track DNN (`train`) and its header export (`bake`); the `gate` step runs it with `--bank prompt` |
| `compare_track_dnn_banks.py` | score a baked track-DNN header by reproducing its forward pass, and compare it with a model `.pt` (the one it was baked from, or a newly trained one) |
| `track_dnn_working_point.py` | the track DNN's working point: the largest threshold holding the outgoing bank's recall in every bin |
| `nano_loader.py` | build feature caches from the track-level datasets: `cache-arm` (31 features); `cache`, `cache-ext`, `cache-prompt`, `cache-prompt27` serve the comparison and studies |
| `train_merged_forest.py` | the final-selector trainer: recipe, working-point rules, compact export and read-back |
| `train_torch_hp_dnn.py` | the baseline chain's network: `check` against the deployed file, `train` and FP16/FP32 export, working point |
| `make_profile.py` | measure a deployed selector on a cache, bin by bin (the `profile` rule of `train_merged_forest.py`, for studies) |
| `read_compact_tree.py` | read a compact `.bin` and score rows with it, the way the selector kernel does |
| `export_compact_tree.py` | write a booster as the compact binary the selector loads (used by the trainer) |
| `prompt_hp_bakeoff.py`, `build_tree_model.py`, `train_prompt_hp_nano.py`, `nano_train_utils.py` | the model-family comparison: the four candidates, their forest and neural-network trainers, shared helpers |
| `CATrackFeaturesTableProducer.cc`, `TripletFeaturesTableProducer.cc`, `HitTruthTableProducer.cc`, `TrackerLayerId.h` | the table producers the two dump configurations use; built with the package (`test/BuildFile.xml`), not run from here |

Several tools also accept options for a process that has a second, displaced iteration and a
merged collection of both iterations, which this release does not have: `--bank displaced`
(`retrain_triplet_gate.sh`, `retrain_track_dnn.sh` and their trainers), `--arm disp|merged`
(`train_merged_forest.py`, `make_profile.py`), the `TrkDisp*` and `TrkMerged*` tables,
`nano_loader.py cache-merged` and the `disp*` options of the dump configurations. The entry
script never passes them: everything it runs and deploys is the prompt iteration.

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
| `CHAIN` | `vh` | pixel chain, as `--chain`: `vh` or `baseline` |
| `THREADS` | 8 | reconstruction threads of the dumps, as `--threads` |
| `DEVICE` | `cuda:0` | torch device of the trainers, as `--device` |
| `THRESHOLD` | none (the step's rule) | threshold baked by `triplet` and `gate`, as `--threshold` |
| `MEM_GUARD_GB` | 300 | stop a heavy pass when the anonymous memory of the control group exceeds this |
| `RT_ALLOW_CONCURRENT_CMSRUN` | unset | set to start next to a running `cmsRun` |
| `RT_LABEL` | `mtv` | truth label of the track-DNN trainer, as `--label` |
| `RT_DISP_CAPS_FACTOR` | 4 | enlargement of the displaced CA's buffers in a displaced-arm track dump (`dispCapsFactor`), for a process with a displaced iteration |
| `MODEL_CLASS` | `forest31` | model family of the `hp` step, as `--class` |
| `NTREES` | 800 | tree budget of the comparison forests |
| `RECALL` | 0.995 | recall of the non-duplicate efficiency tracks at the `hp` working point |
| `POS_SCALE` | 4 | every true track x this, after the class balance (forest and network) |
| `REGION_PT_WEIGHT` | 4 | pT weight of every track, true and fake (forest and network) |
| `EFF_WEIGHT` | per chain (vh 8, baseline 1) | pT weight of the non-duplicate efficiency tracks only (forest and network) |
| `DUP_AS_FAKE` | 1 | loss weight of duplicates trained as negatives (0 = duplicates are true tracks) |
| `EXPORT_TREES` | 150 | number of trees of the forest |
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
