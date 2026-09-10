# The Socratic Method

## Cross-resolution teacher supervision for papyrus surface segmentation

**Authors:** to be confirmed<br>
**Affiliations:** to be confirmed<br>
**Status:** working draft, September 2026<br>
**Repository:** <https://github.com/ubc-nvining/socratic_method>

> This Markdown file mirrors the canonical LaTeX source under
> `submissions/2026-09/src/`. The PDF built from `paper.tex` is authoritative.

### Abstract

Coarse-resolution papyrus surface segmentation fails in two opposing ways:
faint sheets disappear, while nearby wraps merge into blobs. We present a
teacher-student method in which an approximately 2.399 um native-fine Villa
model interrogates the released 9.362 um M7 segmenter. The resulting student is
still an ordinary, single M7 residual-encoder nnU-Net at inference.
Fine-teacher predictions enter the coarse frame as soft occupancy with an
explicit validity field, and the student is held to M7 by one masked
function-space term that acts where the teacher is silent or confidently agrees.
Trained over twelve scrolls for a full 250,000-exposure schedule, the student
reaches 0.6808 macro-scroll Dice on a frozen 689-row benchmark against released
M7's 0.5541 under matched inference, and 0.7130 with eight-way mirror test-time
augmentation at a preregistered operating threshold. We also report what did not
survive. An earlier form of this method carried five further training terms —
medial crest recall, a teacher-background separation shell, one-sided M7
preservation, and fixed-route then dynamic widest-path connectivity — together
with a global parameter trust region. Given a corpus of adequate breadth and no
trust projection, every one of those terms is measurably neutral or harmful. The
trust region itself was roughly 82 times smaller than the displacement the
successful recipe requires, and its saturation is what concealed that fact for
as long as it did. Those negative findings, and the measurement failures behind
them, are reported in the appendices rather than omitted.

### 1. Introduction

The objects behind this work are the Herculaneum papyri, a library of scrolls
carbonized by the AD 79 eruption of Mount Vesuvius. The rolls survive but are too
brittle to open, leaving their writing trapped inside tightly wound coils.
Synchrotron micro-CT records the internal structure without touching the object;
recovering a page then requires locating the papyrus sheet in three dimensions
and virtually unwrapping it to a plane. One scroll is a single sheet, metres
long, visiting many adjacent turns, so pieces that are close in CT may be far
apart on the page.

Segmentation is the first geometric commitment in a scroll-unwrapping pipeline.
A missed surface interrupts a sheet that may otherwise be traceable over a long
distance. An overgrown prediction is at least as dangerous: it can weld nearby
turns of the scroll into one component and make a locally plausible surface
globally wrong. A useful training procedure must therefore grow real faint
structure and remove false material at the same time. A single overlap score
does not express that tension particularly well.

The Vesuvius Challenge provides another source of information. A model operating
on finer CT can resolve local evidence that is ambiguous after downsampling, but
the deployed segmentation and geometry stack still consumes the coarser grid.
This suggests a cross-resolution dialogue. The fine model proposes a soft
occupancy; the coarse M7 model contributes the function that already works; an
explicit loss term exposes where those views are inconsistent.

This is the origin of the name. In the quoted description of Socrates, a speaker
who professes little knowledge reveals inconsistencies in an expert's account
and thereby points toward a better one. Here the fine teacher is not deployed as
a superior replacement. It asks local questions of the expert coarse model.
The answer is a revised M7 architecture that stands alone at inference.

> “I neither know nor think that I know.” — Socrates, in Plato's *Apology* 21d
>
> **SOCRATES:** I am wiser than this man; he fancies he knows something,
> although he knows nothing—<br>
> **DARRYL, SOCRATES' FRIEND:** *fuck him up socrates*<br>
> — leon (@leyawn), April 8, 2015

#### A method arrived at by subtraction

This paper describes a smaller method than the one we first built, and that
difference is a substantial part of its content. An earlier version supervised
the student with seven weighted terms: soft occupancy and Dice, medial crest
recall, a teacher-background separation shell, a masked M7 function anchor,
one-sided M7 positive preservation, and a dynamic widest-path connectivity term,
all under a global parameter trust region. Each term was a plausible local
answer to an observed failure, and the stack as a whole did improve on released
M7. When the corpus was later widened from one scroll to twelve, the trust
projection removed, and the declared schedule actually run to completion,
single-term ablation showed that five of the six auxiliary terms were neutral or
harmful, and that the surviving one — the M7 function anchor — accounted for the
measurable gain. The reduced method is described in the body; the discarded
terms are in Appendix A, and the trust region in Appendix B.

Our contributions are:

1. a hash-pinned and machine-executable 2 um-to-9 um supervision recipe;
2. an objective reduced by single-term ablation to soft occupancy and one masked
   M7 function anchor, reported together with the five discarded terms and the
   measurements that discarded them;
3. an evaluation protocol that runs the declared schedule to completion,
   quantifies patch-sampling uncertainty by paired cluster bootstrap, and
   measures seed noise with a full-horizon replicate before crediting any
   difference; and
4. a separately measured additive 2D postprocess that reconnects only short gaps
   passing geometric and cross-plane objections (Appendix C).

> **Figure 1 - method overview.** File:
> `figures/figure_1_method_overview.png`. The rendered schematic dates from the
> earlier seven-term configuration, so the medial and separation losses and the
> post-update trust projection it depicts are no longer part of the method.

### 2. Cross-resolution student

#### 2.1 Architecture and initialization

The student is a six-stage 3D residual-encoder nnU-Net with feature counts 32,
64, 128, 256, 320, and 320. It accepts one CT channel and predicts background
and surface logits. The network has 102,349,770 trainable parameters. Every
parameter is retrained, but initialization is accepted only when the released
M7 checkpoint has SHA-256
`17465b77591b794638e671f1a9f79c4cf1e79821f302e6fc235e3725e5da7d7e`.
The check is fail-closed and includes the original segmentation heads.

The fine teacher is used in corpus construction, not inference. The deployed
network consumes a coarse CT block and emits the raw student's logits. There is
no fine-teacher forward pass and no probability blend with M7 at deployment.
This boundary matters both scientifically and operationally: an improvement
cannot be attributed to an unreported ensemble, and inference retains the
ordinary M7 memory and integration shape.

#### 2.2 Corpus

The training corpus contains 16,200 registered 192-cubed rows: 15,432 training
and 768 validation, drawn from twelve training scrolls and two held-out
validation scrolls. Of the training rows, 10,072 carry anti-aliased soft teacher
targets and 6,128 are human-labelled rows passed through as hard targets. We
draw 250,000 sample exposures from this schedule; "250,000" counts repeated
draws, not unique rows.

Corpus breadth is the single largest lever we measured, and it is worth stating
plainly because it displaced most of our loss engineering. An earlier
configuration trained on 4,096 rows from one scroll. Widening to twelve scrolls,
with no other change to the objective, moved the frozen benchmark further than
any auxiliary loss term we designed.

#### 2.3 Projected supervision

The fine teacher provides one target in the student's physical frame: soft
occupancy `q` in [0,1], obtained by an anti-aliased pullback from the 2.399 um
grid, with an independent validity mask recording where the teacher is defined
at all. Occupancy estimates how much of a 9.362 um voxel is filled by papyrus,
and retaining it as a fraction rather than a hard vote preserves the
partial-volume evidence at a sheet boundary that motivated the cross-resolution
setup.

**The occupancy ceiling, and why it is a corpus property.** A thin but real
continuation commonly occupies only 25–45% of a coarse voxel. Soft cross-entropy
is correctly calibrated when it drives such a voxel toward p = 0.25–0.45, and the
result may then sit below the segmentation threshold. Read as a property of the
loss, this says occupancy supervision has a ceiling and needs a separate
sheet-existence target; that reading produced the medial crest term of Appendix
A. Read as a property of the data, it says something narrower: on a
single-scroll schedule, the fraction of continuation-band voxels the student can
push over a usable threshold is small. The second reading is the correct one. On
the present corpus the same diagnostic does not fire, and the crest term,
measured there, does not help. We retain occupancy alone.

#### 2.4 Loss

The objective is soft cross-entropy at weight 1.0, soft Dice at weight 1.0, and
a masked M7 function anchor at weight 0.5. Every other term we have implemented
carries weight zero, and there is no parameter-space projection.

Both occupancy terms take the fractional target rather than a thresholded label,
and both are reduced per sample before averaging over the samples that contain
any supervised voxel. The per-sample reduction is deliberate: a dense
human-labelled patch and a sparse native-teacher patch otherwise contribute in
proportion to their valid voxel counts, which lets one supervision source
dominate a mixed corpus.

The anchor holds the student to the released function wherever the teacher is
silent or confidently agrees, and leaves it free everywhere else. Its mask
anchors unknown space, carves a radius-two corridor around known teacher
positives back out of that unknown region, and inside known space acts only
where M7 and the teacher agree **and** the teacher is confident, meaning
`q <= 0.1` or `q >= 0.5`. Without that last restriction, a partial-volume voxel
at `q = 0.3` with M7 below its own decision boundary counts as agreement on
background, and the anchor then pulls down exactly the continuation band the
occupancy target is trying to raise. The KL direction is
`KL(Ber(p_M7) || Ber(p_student))`, which penalizes the student for abandoning
M7's mass rather than for adding its own where M7 is unopposed.

Both occupancy terms and the anchor are applied at the standard nnU-Net decoder
outputs with normalized weights proportional to 1, 1/2, 1/4, and so on, with
zero weight on the coarsest output.

#### 2.5 Optimization

We train with SGD at learning rate 1e-3, Nesterov momentum 0.99, weight decay
3e-5, gradient clipping at 12, and a polynomial decay of exponent 0.9 declared
over the full 250,000-exposure horizon. Batch size is three with no accumulation,
under bfloat16 autocast. Seed 1203 fixes schedule and augmentation randomness.

Declaring the schedule at its true length matters for comparability. The learning
rate at a given exposure depends on the declared horizon, so a run stopped early
is a different optimizer trajectory from a run declared short, and milestone
scores taken while the rate is still near its initial value describe a hot random
walk rather than a converged model. Every arm in this study, including the
ablations of Appendix A, declares the same horizon so that matched milestones are
comparable even when an arm is terminated early.

There is no trust region and no post-update projection. See Appendix B.

### 3. Evaluation protocol

Our primary measurement is macro-scroll Dice on a frozen 689-row benchmark drawn
from PHerc0814 and PHerc1451: the unweighted mean of the per-scroll pooled Dice.
The manifest, its row set and its threshold grid were fixed before this study and
are hash-checked at evaluation time.

Three protocol choices matter. The operating threshold is declared rather than
fitted per candidate: T = 0.35 is the primary endpoint and T = 0.30 the single
preregistered secondary point, chosen a priori as the calibrated argmax at 21 of
25 milestones; a per-candidate calibrated threshold is a diagnostic only.
Uncertainty is measured by paired cluster bootstrap, with support-anchor chunks
as sampling units and identical resample weights for a candidate and its
reference; such an interval describes patch sampling and says nothing about seed
noise, which we measure separately. A promoted candidate must additionally pass a
blind six-cube morphology gate on PHerc1447 and a frontier veto, whose bridge
allowance is interpolated at matched skeleton coverage with no extrapolation.

### 4. Results

**Milestone behaviour and seed noise.** All 25 milestones completed their frozen
audits and were gate-eligible at T = 0.35. Over the eleven late milestones at or
beyond 150,000 exposures, the mean fixed-threshold macro Dice is 0.6613 with a
standard deviation of 0.0107, and adjacent milestones swing by several times that
deviation in both directions. The preregistered late-selection rule chooses
200,000 exposures at 0.680787; for context rather than selection, the global
maximum is 0.6816 and the exact endpoint 0.6693.

A second full-horizon replicate under an identical recipe with a different seed
differs from the primary run by a root-mean-square of 0.0295 across the 25
matched milestones, falling to 0.0191 over the late band. This is why we do not
claim duration superiority: the gap between our selected milestone and an early
milestone of the same recipe is smaller than the difference two seeds produce.

**Comparison with released M7**, frozen 689-row benchmark, macro-scroll Dice:

| Model | Inference | Macro Dice |
|---|---|---|
| Released M7 | T = 0.35 | 0.5541 |
| Released M7 | TTA, T = 0.30 | 0.5875 |
| Released M7 | TTA, calibrated T = 0.20 | 0.5926 |
| Ours (200k) | T = 0.35 | 0.6808 |
| Ours (200k) | TTA, T = 0.35 | 0.7038 |
| Ours (200k) | TTA, T = 0.30 | **0.7130** |

The matched no-TTA gain is 0.1266 macro Dice, a 22.9% relative improvement.
Inference runs the raw student alone, with no teacher forward pass and no blend.

**Inference-time levers.** Two levers that change no weights were preregistered
before scoring, with the rule that a claim above 0.70 requires a positive paired
lower bound after Holm correction and non-negative deltas on both scrolls.
Eight-way mirror TTA at T = 0.30 reaches 0.7130 [0.7012, 0.7248], a paired gain
of +0.0230 [+0.0195, +0.0265] over the same weights without TTA, positive on both
scrolls. Uniform weight averages over late milestones reduce variance without
shifting the level; none beat the selected milestone beyond the seed band, and we
report rather than promote them. At the shipped operating point the blind
six-cube gates and the frontier veto pass: foreground ratio 0.799, with 4,945
reference-relative bridge pixels against a 7,983 ceiling at skeleton coverage
0.782. A Kaggle-style composite on the same rows rises from 0.6401 to 0.6521,
with topology from 0.3880 to 0.4016, meeting the 0.3776 advisory that the no-TTA
configuration missed.

**Held-out human labels.** On PHerc0500P2, the one held-out set with human rather
than model-derived labels, every variant scores between 0.476 and 0.497
composite, and released M7 with TTA at its calibrated threshold remains ahead at
0.5015. These are absolute scores only, because released M7 saw those labels
during its own training; we report them because a flat result on the only
human-labelled holdout is a real limit on what the frozen benchmark establishes.

**Figure provenance.** Every rendered panel in this draft — the teaser, the
registered examples, the failure cases and both galleries — shows the earlier
v31 student at T = 0.45, not the model in the table above. Those panels have not
been regenerated. They remain informative about the qualitative failure modes the
method addresses, but should not be read as depicting the shipped weights.
Regeneration is outstanding release work.

### 5. Limitations

Fine-teacher predictions are model-derived soft supervision rather than human
ground truth, and both gate corpora are likewise model-derived, so our guards
constrain a checkpoint without certifying it. The one held-out set with human
labels is flat across every variant. Milestone-to-milestone variation is
comparable to the differences selection rules are asked to resolve, and one
replicate is a weak estimate of a variance. The ablations of Appendix A are
negative results under one corpus, one optimizer and one horizon; they do not
establish that no version of those objectives could help elsewhere, and the
medial tail floor in particular remains untested at scale. Test-time augmentation
improves the shipped number at eight times the inference cost. Rendered figures
show the earlier student. The postprocessors of Appendix C keep provenance pins
at the earlier operating threshold and must be re-qualified before being combined
with the shipped inference recipe. Finally, exact replay requires large,
separately licensed artifacts.

### 6. Conclusion

A fine-resolution teacher can productively question a coarse expert when its
evidence is represented as soft occupancy and its authority is bounded by a
masked function anchor. Under matched inference the resulting student improves on
released M7 by 0.1266 macro-scroll Dice on a frozen benchmark, and by more with
an inference-time augmentation that leaves the weights untouched.

The larger lesson is subtractive. We began with seven weighted training terms,
each a defensible answer to an observed failure, and finished with three. What
removed the others was not a better idea but a better measurement: a corpus wide
enough for a shape prior to be worth less than data, a schedule run to its
declared length, and the removal of a parameter-space constraint that had been
quietly making every ablation uninformative. The discarded terms were not shown
to be wrong by argument. They were shown to be wrong by measuring them in a
regime where they could have been right.

---

## Appendix A. What we also tried

The objective has three terms. An earlier version had seven. This appendix
records what those terms were, why each seemed justified, and what happened when
they were measured in a regime where they could have been right.

| Term | Weight | Status |
|---|---|---|
| Soft cross-entropy | 1.0 | retained |
| Soft Dice | 0.25 | retained, reweighted to 1.0 |
| M7 function anchor | 0.5 | retained |
| Medial crest recall | 1.0 | retired |
| Separation shell | 2.0 | retired |
| M7 preservation | 1.0 | retired |
| Dynamic connectivity | 0.03125 | retired |
| Trust region | — | retired (Appendix B) |

### A.1 Why the stack survived as long as it did

The terms were not retained because they had been shown to work. They were
retained because, in the regime where they were developed, nothing could be shown
either way, and a term that shows no measurable effect is easy to keep.

First, the parameter-space trust region was saturated: the post-update projection
was active on essentially every optimizer step, so no term could move the
function far enough to demonstrate either benefit or harm. Appendix B treats this
in detail because it is the load-bearing failure.

Second, the promotion gate rewarded the behaviour the terms produced. Its
dominant criterion required that at least 95% of the M7 voxels already judged
teacher-correct survive within a one-voxel neighbourhood of the student's
prediction. That is a retention test, and preservation and the anchor are
retention terms; a thinner, better student fails it. No shipped term scored above
the no-term baseline on this gate, and the only intervention that moved it was an
inference-time blend with M7, later withdrawn as confounded. A companion
anti-blob check compounded the problem by pinning its reference to an older
prediction grid with a foreground cap, bounding any candidate at roughly 39% of
the foreground of a previously accepted model, so growth was scoreable only as a
violation.

Third, the training corpus had narrowed to 4,096 rows from a single scroll. A
shape prior is most attractive exactly when data is scarce, and least testable for
the same reason.

A fourth, procedural point belongs here. The terms were added incrementally and
evaluated as a stack against the previous stack, so what was tuned was the
interaction, not the terms. Leave-one-out arms were specified for five of them and
never trained. Preservation is the clearest casualty: it was always paired with
the separation shell that offsets its cost, so the question of what it does alone
was never asked until it was asked directly.

### A.2 Single-term ablations

Once the corpus was widened and the trust region removed, each candidate term
could be added individually to a common baseline. Arms trained from released M7
under the body's optimizer, all declaring the same horizon and terminated at the
same milestone, differing only in the named flag:

| Arm | Macro Dice |
|---|---|
| Occupancy only (CE + Dice) | 0.6542 |
| Occupancy only, seed replicate | 0.6455 |
| + separation shell 0.5 | 0.6349 |
| + M7 function anchor 0.5 | **0.6769** |

The anchor is the one term that clears its own noise floor, and it is the one
term retained. The separation arm lands below the baseline. The two
occupancy-only arms differ by more than most effects we were trying to detect,
which is why every later comparison is expressed against a measured seed band.

### A.3 Separation shell

The shell placed background cross-entropy on a two-voxel dilation of the
teacher-supported positives, intersected with confidently empty teacher space:
a local fence against unsupported girth and short cross-wrap welds.

Measured as a dial, it does what it claims. Raising its weight from 2 to 3 cut
reference-relative bridge pixels by roughly half, and cost about sixteen points
of reference skeleton recall against a tolerance of one; lowering it toward zero
inverted both effects. What the sweep does not show is any movement of the
underlying trade. Compared at matched coverage rather than at a fixed threshold, a
model trained with the shell sits on the same bridge-versus-coverage curve as one
trained without it. The shell selects a point on that curve, and so does the
segmentation threshold, which is free. We retired it because a training-time term
that duplicates a deployment-time dial should be the dial.

### A.4 Medial crest recall

The crest term supervised sheet existence separately from amount of material. The
fine teacher's medial surface was extracted with the center–radius view of LSMAT,
projected to the coarse grid with a max operator so a thin center survives
downsampling, and rewarded through a per-sample recall term evaluated only on
crest voxels. Slice-wise radii were deliberate: pairing 2D centers with a 3D
distance transform would substitute spheres for disks and collapse the medial
surface toward a curve.

Its premise was the occupancy ceiling, which is corpus-dependent. On the wide
corpus the diagnostic that was supposed to trigger the term does not fire, so it
was never triggered for a wide-corpus arm of its own. Where it was measured,
raising its weight reduced bridge burden by roughly a fifth but cost five points
of skeleton recall against the same one-point tolerance, and the rendered panels
showed it refilling broad mass rather than extending thin lines. Global recall
pressure is the wrong shape for the problem: it buys length by buying width.

A focused variant survives as untested rather than refuted. Instead of rewarding
all crest voxels, it applies a one-sided hinge to the weakest fifth of them, so
gradient vanishes once a selected voxel clears a floor. It had the best
burden-versus-recall behaviour of the medial family in its one trial, and it
cannot be evaluated on the present corpus at all: it requires a medial sidecar on
every training row, and the human-labelled rows can never carry one.

### A.5 M7 positive preservation

Preservation applied a one-sided foreground cross-entropy to voxels where the
frozen M7 predicted foreground inside a two-voxel corridor around teacher
positives, to stop de-blobbing from deleting M7-supported sheet. It had the best
surviving motivation of any retired term: our students beat M7 on overlap and on
merge behaviour but lose to it on topology on the one human-labelled holdout,
which is the deficit preservation is shaped to fix. It was also the only term
never measured in isolation at scale, so we measured it.

Three arms were preregistered against the shipped recipe, each declaring the full
horizon and terminating at a common milestone: the original hard form at its
shipped weight; a one-sided soft-floor form that resists only where the student
has fallen below M7; and, conditional on a rim-thickening trigger, the hard form
paired with a reduced separation shell. Kill criteria, a seed-derived tie margin
and the thickening trigger were fixed before any arm ran.

All three were rejected, and the failure mode was the one the preregistration
named. Pooled paired deltas against the shipped recipe over five matched
milestones were -0.0480 for the hard form, -0.1117 for the soft floor and -0.0228
for the shell-paired arm, every interval excluding zero. Each failed the blind
six-cube morphology gate at two consecutive milestones on interior fraction and
maximum thickness. Foreground ran between 1.3 and 2.3 times the reference where
the shipped recipe runs at 0.71, and reference-relative bridge pixels ran an
order of magnitude above it. The frontier veto could not be evaluated at all: at
the coverage these models reach they lie beyond the end of the reference curve,
which is itself the finding. None of the arms was inert — the term contributed
between 13% and 22% of the committed training loss — so these are measured
negatives rather than absences.

The shell-paired arm is the informative one. It halves the damage on every axis
and still fails every gate. Read together, the three arms say that preservation's
cost is paid in girth, and that the separation shell of the original stack
existed to pay it. Reconstructing that pair means reintroducing a term retired in
A.3 for duplicating a free deployment-time dial.

Evidence: [`recipes/f0_ablations_20260910`](../../../recipes/f0_ablations_20260910/README.md).

### A.6 Dynamic connectivity

The connectivity term asked, existentially, for some viable route through an
audited teacher-medial corridor joining two or more disconnected M7 components.
Its precursor prescribed one minimum-off-axis route and raised the weakest tenth
of its logits; that version tied the incumbent at the smallest weight and degraded
ordinary validation monotonically at larger ones, which motivated replacing a
prescribed path with a bottleneck on whichever path the student currently
supports. The mechanism is sound and remains the most interesting of the retired
terms.

It is nonetheless inapplicable to the corpus that produced our result, for a
structural reason rather than a measured one. An event requires a full-volume M7
prediction to define the component-contact pins, and that prediction exists for
exactly one training scroll. The frozen atlas contains 149 events, all from that
scroll; on the wide corpus roughly three percent of rows carry any event, so the
term is close to inert by construction and its weight cannot be interpreted.
Producing the missing predictions is days of inference; until then the honest
status is untested at breadth, not refuted.

### A.7 Terms proposed and rejected later

A hardened Dice term, taking the thresholded label the benchmark actually scores
in addition to the soft target, lost to the shipped recipe at both matched
milestones with intervals excluding zero, and carries a documented confound:
human rows are already binary, so their Dice weight doubles. Dropping the
human-labelled rows entirely lost at the earlier milestone and won narrowly at the
later one, inside the seed band, and was not promoted.

### A.8 The retired configuration, for reference

The completed duration ladder of the seven-term configuration on its
single-scroll corpus reached a best calibrated macro-scroll Dice of 0.58293 at
7,168 samples, a gain of roughly 0.02 over released M7. The present method gains
0.1266 on the frozen benchmark. The entire seven-term stack was worth about a
sixth of what corpus breadth and a completed schedule were worth, which is the
ratio we would most like a reader to take away.

## Appendix B. A cautionary note on trust regions

The retired configuration projected the student's complete parameter vector back
into a relative-L2 ball around the released M7 parameters after every optimizer
update, with radius 0.0027535422. We describe it separately from the retired loss
terms because its consequences were not local to itself: while it was active it
made the other terms unmeasurable, and every ablation verdict recorded under it
is uninformative.

**Why it seemed reasonable.** The premise was conservative and, stated
abstractly, sound. The released M7 is a working expert; the fine teacher is
locally more informative but globally incomplete and model-derived; a student free
to move anywhere in parameter space can follow a confident teacher into a region
where it is wrong, and the loss will not notice because the teacher supplies the
targets. The radius was not arbitrary either: it was derived from an earlier
measurement of how far the function could safely depart, then inherited across
several subsequent versions. The mistake was not the mechanism. It was retaining a
radius across a change of objective, corpus and optimizer without re-measuring
what the new configuration required, and then reading the resulting absence of
movement as an absence of effect.

**What it actually did.** The projection was active for 97.7% of updates in the
first evaluation interval and 100% thereafter, with a mean pre-projection step
slightly larger than the radius itself. Roughly a quarter of every optimizer step
was being discarded. The ball was not a guardrail that occasionally caught an
excursion; it was a boundary the run rode continuously, which makes the radius a
hyperparameter of the method rather than a safety margin.

The decisive number came from an observation rather than an experiment. Once a
wide-corpus run was trained without any projection, we measured where it ended
up. The recipe that produced the model in this paper sits at roughly 67 times the
shipped radius after 30,000 exposures and 82 times after 50,000, and keeps
moving. The displacement the successful method requires is nearly two orders of
magnitude outside the ball earlier versions were confined to. No amount of loss
design inside that ball could have found this model, because the model is not in
it.

**The second-order damage.** A saturated constraint silently converts every
comparison made under it into a comparison of projections. Two arms differing in
one loss term will produce nearly identical functions if the projection removes
most of the difference, and the correct reading of "these two runs are
indistinguishable" is then "this experiment had no power", not "this term does
nothing". Our own record contains the clean example: a carefully matched pair
testing one mask refinement of the M7 anchor produced probability fields
differing by fractions of a percent at every threshold, and the run logs show
why — the anchor's own loss component was mechanically near zero, because the
function never left M7's neighbourhood. That pair was recorded as a null result
for the refinement. The refinement is in the shipped objective today, and it is
one of the three mask properties the term needs to work. This is the mechanism by
which a saturated trust region manufactures a stack of loss terms.

**What we would do instead.** We do not conclude that parameter-space trust
regions are a bad idea; we conclude that an unmeasured one is, and that the
measurement is cheap. Log the projection's activity fraction and the
pre-projection step norm from the first interval, and treat sustained saturation
as a failed run rather than a successful guard. Before trusting any ablation,
measure the unconstrained displacement of the same recipe and compare it with the
radius; if the free run ends up far outside, the ablation was run at zero power
and its verdict should be discarded rather than recorded. Re-derive the radius
whenever the objective, corpus or optimizer changes. And where the underlying
worry is following a model-derived teacher into its own errors, prefer a
constraint expressed in function space and restricted to where the teacher is
silent or confidently agrees — which is what the M7 anchor does, and which
survived ablation when the parameter-space constraint did not.

## Appendix C. Additive 2D curve fitting

Segmentation and geometric repair are separate stages, and this appendix is
separated from the body for the same reason: the postprocess runs after
thresholding, changes no weights, and is measured on its own terms.

After thresholding, the native C postprocessor skeletonizes each z-plane with
Zhang-Suen thinning, prunes short spurs, labels components, and extracts
endpoints with outward tangents. It builds endpoint pairs under one symmetric
score and accepts them greedily in deterministic score order while keeping each
endpoint unique.

Candidate pairs must pass successively more expensive objections: excluded or
clipped endpoints, same-component closure, reach, tangent facing and opposition,
umbilicus-aware radial displacement, adjacent-plane evidence for long joins,
score, Bezier arc ratio, a third-component merger margin, an adjacent-sheet
corridor check, and intersection or near-contact with already accepted joins.
Endpoint and connection tracks across z then supply persistence evidence for a
second matching round. Only sufficiently supported connections are painted as
cubic Bezier disk strokes. Painting is additive and every new pixel is auditable.

In a calibrated 4x5x5 study the fitter made 481 joins and added 12,697 pixels.
High-resolution precision was 92.5% overall and 98.4% for joins no longer than
six pixels. We observed no full-turn fusion and measured a 21% reduction in
atlas overlap. A larger 21-cube census measured 98.3% weighted precision.
High-resolution connectivity is confirmation, however, not a proof against
cross-wrap joins; the radial gate remains the primary cross-wrap certificate.

The bridge scanner reports thin-neck weld candidates but cutting is off by
default because erasure breaks the additive contract. An experimental
mesh-stage slab splitter separated zero of 246 fused runs. It remains in the
source tree as a measured negative result and is not part of the recommended
pipeline.

Three further opt-in stages are implemented in the repository and evaluated in
their own records: a checked repair profile, a probability-guided continuity
tool that reaches beyond the native fitter's 21-pixel limit, and a stage that
extends supported curves into explicit 3D patches. All three share this
appendix's contract — additive, opt-in, separately measured, no model change —
and all three currently pin the earlier operating threshold, so each must be
re-qualified before being combined with the shipped inference recipe.

> **Figure 3 - measured line-fitter decisions.** File:
> `figures/figure_3_line_fitter_examples.png`.

### Reproducibility and release

The repository contains the full Python package snapshot, machine-readable
recipes, exact command arguments, hashes and byte sizes for the staged inputs,
the observed environment lock, original research plans and drivers, and the
isolated native line fitter. A small portability shim maps the source
experiment's absolute Windows root to a local artifact mirror at read time. It
does not rewrite the provenance-bound JSON or JSONL bytes.

The large corpora, released M7 weights, Villa teacher material, and student
checkpoints are not appropriate ordinary Git payloads. Model and dataset cards
plus a safetensors exporter are staged for artifact hosting, but no upload
should occur until ownership and upstream terms are resolved. The repository
uses an explicit license hold rather than guessing a permissive license.

### References

The SIGGRAPH source bibliography is authoritative. In particular, the medial
representation cited in Appendix A is Rebain et al., “LSMAT: Least Squares
Medial Axis Transform,” *Computer Graphics Forum* 38(6), 2019,
[doi:10.1111/cgf.13599](https://doi.org/10.1111/cgf.13599). Final artifact
metadata must also include the Vesuvius Challenge data and task citation,
released M7 model, Villa teacher implementation/model, nnU-Net and
dynamic-network-architectures, knowledge distillation, Zhang--Suen thinning,
and the public records used for the release.
