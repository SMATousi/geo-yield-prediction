# Expert-Validated Statement Pretraining — v1

**Status:** Approved design, 2026-09-28. Not implemented.

This document specifies an optional addition to the field-scale System B
pretraining framework. The existing five objectives remain in place. The new
component uses general agronomic statements to condition cross-modal prediction
before supervised fine-tuning on measured yield.

## 1. Purpose and approved scope

A model drafts general statements about modality types and their relationships.
A human agronomic expert validates every statement admitted to the training
library. A frozen text encoder embeds the approved statements, and those
embeddings provide additional context to an auxiliary sensor-prediction task.

The approved v1 decisions are:

- Statements describe general agronomic relationships, not individual field
  observations or sample captions.
- The drafting model receives modality descriptions and relationships, not field
  sensor measurements or measured yield.
- Every training statement must be approved by a human expert.
- Knowledge contributes a soft, weighted auxiliary loss.
- Yield prediction uses sensor inputs only. Text is used during pretraining.
- V1 requires statements only: no executable conditions, numerical rule
  thresholds, logical predicates, or explicit statement-satisfaction scorer.

The longer-term intent is to encourage representations consistent with expert
knowledge connecting multiple modalities. V1 implements **statement-conditioned
cross-modal pretraining**, not direct enforcement of conditional agronomic rules.
Its effectiveness must be measured rather than inferred from expert approval.

## 2. Statement library

Store the library as a versioned JSONL artifact. Each record contains:

| Field | Meaning |
|---|---|
| `id` | Stable statement identifier. |
| `text` | The complete expert-approved natural-language statement. |
| `modalities` | Model modality names involved in the statement. |
| `review_status` | `draft`, `approved`, or `rejected`. Only `approved` is eligible. |
| `reviewer_id` | Identifier of the expert who approved this version. |
| `reviewed_at` | Approval timestamp. |
| `version` | Version of this statement. Editing approved text requires review again. |
| `generator` | Optional provenance for the drafting model and prompt version. |

The modality tags support routing and do not turn the statement into a structured
rule. They identify which inputs the text discusses; they contain no conditions,
thresholds, or numerical consequences. Require at least two distinct, recognized
modality names for this cross-modal objective.

Statement approval concerns agronomic meaning, including any qualifications
written in the text. No per-field statement review or executable rule review is
required in v1. Reject malformed records and duplicate identifiers; exclude
unapproved records from the training library.

## 3. Text encoding

Use a pretrained text encoder with frozen parameters and evaluation-mode behavior.
Choose its model and tokenizer during implementation; this spec does not select a
provider or model. Cache embeddings of the approved library rather than re-encode
unchanged statements on every training step.

Record the library version/hash, text encoder identifier and revision, tokenizer,
and preprocessing settings with each experiment. These identify the actual
knowledge input and allow a cache to be rebuilt or invalidated after changes.

The text encoder consumes statements only. It does not generate sample-specific
claims from sensor data during training.

## 4. Auxiliary prediction task

For an eligible field sample and target modality:

1. Encode the observed modalities using the existing native-resolution encoders.
2. Hold the target modality out of the predictor's sensor inputs. Its actual
   encoding supplies the reconstruction target only.
3. Select approved statements tagged with the target and at least one remaining
   observed modality. Average their cached text embeddings to obtain one
   conditioning vector for this prediction direction.
4. Fuse the remaining sensor tokens with the existing backbone and mean-pool its
   latents into a field representation.
5. Concatenate that representation with the text conditioning vector. A dedicated
   auxiliary MLP for the target modality predicts its pooled sensor embedding.
6. Compare the prediction with the detached, mean-pooled target embedding.

For shared sensor dimension `D` and text dimension `D_text`, the auxiliary head
maps `(B, D + D_text)` to `(B, D)`. Gradients update the remaining sensor encoders,
the fusion backbone, and the auxiliary head. The frozen text encoder and detached
target branch receive no gradients from this objective. Other pretraining
objectives continue to train the modality encoders.

The first version predicts pooled embeddings, consistent with the current
cross-modal objective. It does not reconstruct spatially distinct sensor pixels.

### Eligibility and missing modalities

Compute this objective only for sample/target pairs with:

- an actually observed target modality;
- at least one actually observed conditioning modality; and
- at least one relevant approved statement.

Availability is evaluated per sample. A learned missing token is not an observed
target or a qualifying conditioning source. Missing target rows do not contribute
to the loss. If no sample/target pair is eligible, return a zero knowledge loss and
report zero eligible pairs.

Selection by modality tags does **not** establish that a statement's agronomic
conditions hold in a particular field. The text provides context to the predictor;
v1 has no mechanism to test or activate those conditions explicitly.

## 5. Soft training objective

Use mean squared error in the shared sensor embedding space. Average over embedding
dimensions and eligible samples for each target direction, then average the active
directions:

```text
target_m = stop_gradient(mean_pool(sensor_encoder_m(input_m)))
prediction_m = knowledge_head_m(concat(fused_other_sensors, relevant_text))
L_knowledge = mean_over_active_directions(MSE_over_eligible_rows(prediction_m, target_m))
L_total = L_existing_five_objectives + lambda_knowledge * L_knowledge
```

Expose `lambda_knowledge` as an experiment configuration value; choose its value
using validation rather than fixing an unsupported default. The feature is
optional and disabled by default. When disabled, preserve the existing objective
behavior and perform no text encoding or additional fusion passes.

Use a separate auxiliary head from ordinary cross-modal prediction so the
statement-conditioned component can be enabled and ablated independently. Report
the unweighted knowledge loss, its weighted contribution, and eligible pair count.
Sample prediction directions if necessary to keep the added fusion cost bounded.

## 6. Fine-tuning and inference

Transfer the sensor encoders and fusion backbone to supervised yield fine-tuning.
Discard the text encoder, cached statements, and knowledge prediction heads from
the yield inference path. Loading a fine-tuned model and making predictions must
require no statement library, text input, or text-model access.

Measured yield remains the ground truth for supervised fine-tuning and evaluation.
The statement library supplies pretraining context; it supplies no yield labels.

## 7. Evaluation and acceptance

Compare three primary pretraining conditions:

1. The existing five objectives without knowledge conditioning.
2. The existing objectives plus approved statement conditioning.
3. The same added objective with shuffled statement assignments.

Also compare the added prediction head with its text input removed or replaced by
a constant vector. This separates the benefit of additional prediction training
from the benefit of statement content. Use comparable training budgets, the same
sensor data, spatial splits, fine-tuning protocol, and evaluation metrics.

Measure held-out cross-modal reconstruction error and downstream yield-map
performance. Report uncertainty across repeated runs where practical. Shuffled
text and no-text controls are necessary because the predictor may learn to ignore
the statements. Useful expert knowledge is not, by itself, evidence that this
conditioning mechanism transfers it into sensor representations.

Acceptance requires:

- Tests exclude unapproved statements and missing target rows from the objective.
- Tests verify that holding out a target prevents its embedding from entering the
  predictor and that text-encoder parameters remain frozen.
- Tests cover empty libraries, no eligible pairs, and heterogeneous availability.
- Disabling the feature preserves the existing pretraining path.
- Fine-tuning and inference run without text artifacts.
- Reproducible real-data comparisons establish whether the addition helps.
  Synthetic runs establish executability only.

Do not report logical rule satisfaction or violation rates for v1: it has no
executable definition of either. Report a lack of measured benefit or ignored text
as an experimental result, not as successful knowledge transfer.

## 8. Implementation placement and later work

Implement this as an optional extension of `MultimodalSelfSupervisedPretrain`.
The statement library and embedding cache belong to the pretraining data/config
path, rather than the supervised field-sample contract.

Real unlabelled field ingestion is a prerequisite for meaningful evaluation
(roadmap Phase 3). Integrate and profile the experimental objective with the
pretraining and transfer work in Phase 4. No implementation is claimed by this
document.

Later versions may explore explicit conditions, measurable consequences,
applicability checks, confidence-weighted rules, and statement-satisfaction
scoring. Those require a separate design decision and are outside the approved v1
scope.
