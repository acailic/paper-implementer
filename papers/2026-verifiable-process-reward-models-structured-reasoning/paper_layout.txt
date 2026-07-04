                                             Beyond Outcome Verification: Verifiable Process Reward Models for
                                                                  Structured Reasoning

                                                                  Massimiliano Pronesti1,2 , Anya Belz2 , Yufang Hou1,3
                                                               1
                                                                 IBM Research Europe - Ireland, 2 Dublin City University,
                                                               3
                                                                 IT:U Interdisciplinary Transformation University Austria
                                                                 Correspondence: massimiliano.pronesti@ibm.com, yufang.hou@it-u.at




                                                               Abstract                            mathematical evaluation (Guo et al., 2025; Lam-
                                                                                                   bert et al., 2025). By grounding supervision in
                                             Recent work on reinforcement learning with
arXiv:2601.17223v1 [cs.CL] 23 Jan 2026




                                                                                                   objective verifiers rather than learned reward mod-
                                             verifiable rewards (RLVR) has shown that large        els, RLVR avoids many of the alignment failures
                                             language models (LLMs) can be substantially
                                                                                                   associated with neural reward hacking and has pro-
                                             improved using outcome-level verification sig-
                                             nals, such as unit tests for code or exact-match      duced state-of-the-art performance in code gen-
                                             checks for mathematics. In parallel, process          eration and mathematical reasoning (Wang et al.,
                                             supervision has long been explored as a way to        2025b; Zhang and Zuo, 2025; Guo et al., 2025;
                                             shape the intermediate reasoning behaviour of         Yang et al., 2025a).
                                             LLMs, but existing approaches rely on neural             However, outcome-only RLVR provides rewards
                                             judges to score chain-of-thought steps, leaving       solely at the terminal step of reasoning, offering
                                             them vulnerable to opacity, bias, and reward          no guarantees about whether the model followed a
                                             hacking. To address this gap, we introduce
                                                                                                   valid intermediate process. To address this lim-
                                             Verifiable Process Reward Models (VPRMs),
                                             a reinforcement-learning framework in which           itation, several extensions augment RLVR with
                                             intermediate reasoning steps are checked by           structural or auxiliary signals, such as masked-and-
                                             deterministic, rule-based verifiers. We apply         reordered self-supervision (Wang et al., 2025c) or
                                             VPRMs to risk-of-bias assessment for medical          self-verification mechanisms (Zeng et al., 2025).
                                             evidence synthesis, a domain where guideline-         These works strengthen RLVR but still operate
                                             defined criteria and rule-based decision paths        fundamentally at the level of outcome verifica-
                                             enable programmatic verification of reason-
                                                                                                   tion. Most importantly, none of the above meth-
                                             ing traces. Across multiple datasets, we find
                                             that VPRMs generate reasoning that adheres            ods provide a fully verifiable form of process su-
                                             closely to domain rules and achieve substan-          pervision, and existing approaches that score in-
                                             tially higher coherence between step-level de-        termediate Chain-of-Thought (CoT) steps rely on
                                             cisions and final labels. Results show that           neural judges (Lightman et al., 2024; Zhang et al.,
                                             VPRMs achieve up to 20% higher F1 than state-         2025; Zou et al., 2025) which reintroduce opacity,
                                             of-the-art models and 6.5% higher than verifi-        bias, and opportunities for reward hacking (Amodei
                                             able outcome rewards, with substantial gains in       et al., 2016; Skalse et al., 2022).
                                             evidence grounding and logical coherence.
                                                                                                      To date, no work has demonstrated that pro-
                                                                                                   cess rewards themselves can be made verifiable
                                         1   Introduction                                          on tasks whose structure admits deterministic sym-
                                                                                                   bolic checking. Yet such a capability would be
                                         Large language models (LLMs) have made remark-            highly desirable: if intermediate reasoning steps
                                         able progress in complex natural language process-        can be validated explicitly, then reinforcement
                                         ing tasks, including reasoning, planning, and struc-      learning could optimise not only for correct out-
                                         tured decision making (Brown et al., 2020; Ope-           comes, but also for transparent, logically sound
                                         nAI, 2024, 2025). Reinforcement learning with             reasoning. Crucially, verifiable step-level rewards
                                         verifiable rewards (RLVR) has recently emerged            would eliminate the aforementioned problems as-
                                         as a robust alternative to preference-based rein-         sociated with neural process rewards.
                                         forcement learning, enabling LLMs to improve                 This leaves a key open problem: can reinforce-
                                         using reward signals derived from deterministic           ment learning be used to train models whose entire
                                         checks such as program test suites or exact-match         reasoning trajectory is rewarded only when each
                                                                    Reasoning process

                        Input
                                                        Step 1                 Step 2                Step 3                Step 4               Output


                                                   Randomization         Randomization            Sequence                Baseline
                                                                                                                                                Risk: Low
                                                      reported?             method?            predictability?           Imbalance?



                      Study on an
                    intervention to
                    improve patient                    Reported                Random            Unpredictable                None
                     recovery after
                        surgery.
                                                        Label                  Label                 Label                  Label



                                                 Randomization                             Sequence                      Predictable            ? Moderate
                                                                          Random
                                                     method                             predictability
                                   Reported                       Non Random
                                                                                                         Unpredictable                             - High
                  Randomization                                                                                                        Likely
                     reported                                                               - High                          Baseline
                                                                                                                           imbalance
                                  Not Reported                                                                                         None

                                                                                                                                                    + Low
                                                 ? Moderate



                         Step                       Label                       Final Outcome




Figure 1: Illustration of the verifiable reasoning setup for risk-of-bias assessment (type A: bias arising from the
randomisation process). Top: given an input study x, the model produces a structured reasoning trace Y =
(o1 , . . . , oT ) with step-level labels (ℓ1 , ℓ2 , ℓ3 , ℓ4 ), each corresponding to a guideline-defined assessment question.
Bottom: the corresponding rule-based decision tree, which deterministically maps each combination of step-level
labels to low (+), high (-), or moderate (?) risk.


intermediate step satisfies domain-defined, rule-                                           neural process-reward baselines, and pretrained
based criteria? To address this gap, we introduce                                           LLMs prompted for Chain-of-Thought (CoT). Our
Verifiable Process Reward Models (VPRMs), a                                                 results show that VPRM-trained models achieve
reinforcement-learning framework in which each                                              substantially higher accuracy and more coherent
reasoning step is assessed by a deterministic veri-                                         reasoning traces, showing that verifiable process su-
fier grounded in explicit task guidelines. VPRMs                                            pervision offers a more reliable optimization signal
provide fine-grained, step-level reward signals that                                        for both result correctness and process soundness.
complement outcome-level verification, guiding                                                  In summary, our contributions are as follows: (i)
optimisation toward reasoning traces that are both                                          we propose a verifiable process reward framework
correct and aligned with domain logic. Crucially,                                           that integrates deterministic step-level verification
we prove that under mild assumptions, VPRMs                                                 with reinforcement learning for dense, interpretable
offer theoretical guarantees that gradient-based up-                                        supervision over reasoning trajectories (Section 3);
dates assign positive expected weight to correct                                            (ii) we show that, under mild reward-separation
reasoning trajectories and negative weight to incon-                                        assumptions, verifiable process rewards encourage
sistent ones, thereby encouraging sound reasoning.                                          correct reasoning (Section 3.3, Appendix A); and
   To evaluate this framework, we consider a chal-                                          (iii) we validate the approach on risk-of-bias assess-
lenging, real-world structured-reasoning task: risk-                                        ment in medical systematic reviews, demonstrating
of-bias (RoB) assessment in clinical systematic re-                                         significant improvements over outcome-only and
views. In this setting, studies must be evaluated for                                       neural process-reward baselines (Section 4).
susceptibility to systematic error (Chandler et al.,
                                                                                             2       Preliminaries
2019), and domain guidelines prescribe a rigid se-
quence of reasoning steps and decision rules that                                            2.1       Policy Optimisation Algorithms
make the task uniquely amenable to verifiable pro-                                           Group Relative Policy Optimization (GRPO)
cess supervision. Figure 1 illustrates the verifiable                                        GRPO (Shao et al., 2024) is a widely-adopted
reasoning process for assessing randomisation bias,                                          group-based reinforcement learning method that
one of the RoB domains defined in the Cochrane                                               optimises a policy by comparing multiple candi-
RoB tool for randomised trials (Sterne et al., 2019).                                        date completions for the same input.
  Across multiple models and RoB domains, we                                                   For each passage x, G candidate completions
compare VPRMs against outcome-only RLVR,                                                     {yi }G
                                                                                                  i=1 ∼ πold (· | x) are sampled from the refer-
ence policy πold to encourage robustness and diver-            whether a model output satisfies task-specific con-
sity. These completions are scored using a reward              straints. This approach is particularly effective for
model. The raw rewards Ri are then normalised                  verifiable tasks with clear notions of correctness,
across the group:                                              such as mathematical problem solving, program
                                                               synthesis, or logical reasoning. The reward is com-
                 Ri − E[Rj ]                                   puted deterministically by a verifier and does not
       Ai =       p          ,      j ∈ {i, ..., G}
                    V[Rj ]                                     rely on learned preference models. In its simplest
                                                               form, the reward is binary:
 where E[Rj ] and V[Rj ] are respectively the mean
                                                                         (
and variance of the rewards for the group of re-                           1 if the output is verified as correct,
sponses. The policy is optimised using a clipped,               R(y) =
                                                                           0 otherwise.
KL-regularised objective that encourages agree-
ment with high-reward behaviours while maintain-               This setting provides scalable, reliable supervision
ing proximity to a reference model πref :                      with minimal ambiguity.

                      G       |yi |                           2.3    Systematic Reviews
                   1 X 1 X              
 LGRPO (θ) = E                       min pi,t (θ)Ai ,          Systematic reviews provide a principled framework
                  G i=1 |yi | t=1
                                                             for aggregating empirical evidence through pre-
        clip(pi,t (θ), 1 − ε, 1 + ε)Ai − β KL[πθ ∥ πref ]      defined search strategies, explicit inclusion crite-
                                                               ria, and reproducible synthesis pipelines (Chan-
where β governs the regularisation strength and                dler et al., 2019). Their value lies in reducing
pi,t (θ) is the token-level probability ratio defined          subjective judgment in evidence collection and
as follows:                                                    enabling structured comparison across heteroge-
                          πθ (yi,t | x, yi,<t )                neous studies. One of the gold-standard reposito-
           pi,t (θ) =                                          ries of systematic reviews is the Cochrane library (),
                         πθold (yi,t | x, yi,<t )
                                                               which contains curated reviews adhering to strict
Dynamic sAmpling Policy Optimization (DAPO)                    evidence-synthesis and bias-assessment protocols.
Building on GRPO, DAPO (Yu et al., 2025) re-
                                                               2.4    Risk of Bias Assessment
moves the KL penalty, introduces a clip-higher
mechanism, incorporates dynamic sampling, ap-                  Risk of bias assessment is a core component of sys-
plies a token-level policy gradient loss, and adopts           tematic reviews, providing structured evaluations
overlong reward shaping. The key improvement                   of methodological flaws in primary studies that di-
is dynamic sampling, by over-sampling and filter-              rectly inform evidence synthesis and the credibility
ing out prompts with the accuracy equal to 1 and               of review conclusions. Assessments are structured
0, leaving all prompts in the batch with effective             into a fixed set of domains of bias, focusing on
gradients, avoiding dampening the gradient signals             different aspects of trial design, conduct and re-
for model training with a larger variance in the gra-          porting (Chandler et al., 2019; Sterne et al., 2019).
dient. This leads to the following maximisation                Within each domain, information about features
objective:                                                     of the trial that are relevant to risk of bias is col-
                                                               lected and mapped to judgments of low, medium
                                |yi |
                               G X                             or high risk. This domain-based assessment iden-
                    1          X              
   LDAPO (θ) = E PG                        min pi,t (θ)Ai ,    tifies issues such as selection bias, measurement
                       i=1 |yi | i=1 t=1
                                               
                                                               bias, and selective reporting, enabling subsequent
         clip(pi,t (θ), 1 − εH , 1 + εL )Ai         ,          evidence synthesis to appropriately weight studies
                                                               not only in terms of their results, but also in terms
                 s.t. 0 < |{yi | is_equivalent(yi , a)}| < G
                                                               of their risk of bias. Additional details on the differ-
This ensures that for the same input, the sampled              ent bias domains and the corresponding assessment
set contains both correct and incorrect answers.               reasoning process are provided in Appendix B.

2.2   Rule-based Reward Modeling                               3     Verifiable Process Reward Models
In rule-based reward modeling, the reward signal               We introduce Verifiable Process Reward Models
is defined by explicit, hand-crafted rules that verify         (VPRMs), a framework for process supervision
                          Verifiable Outcome Rewarding                              Verifiable Process Rewarding


                                               RCT                                                    RCT




                                                                                                                         step score: 1.0
                                                                                                                         step label score: 0.66
                                                                               <think>
                         <think>                                               Step 1: Identify_randomization_report
                         The paper reports that the                            ...                                               Step
                         randomization method used was                         Answer: reported                                 Label
                         "random," and the sequence generation
                         appears to follow standard protocols.                 Step 2: Classify_randomization_method
                                                                               ...                                               Step
                         However, there are some
                                                                               Answer: random                                   Label
                         inconsistencies in the sequence
                         pattern, and the method of                            Step 3: Assess_sequence_predictability
                         randomization was not explicitly                      ...
                                                                                                                                 Step
                         detailed.                                             Answer: unclear
                                                                                                                                Label
                         </think>
                         <answer>             Verification                     </think>
                           risk: high                               Low        <answer>
                                                                                risk:low                                       Low
                         </answer>
                                                                  score: 0.0   </answer>
                                                                                                                           score: 1.0



                                     Reasoning                Ground truth                  Reasoning                         Ground truth




Figure 2: Comparison between verifiable outcome rewards (left), which evaluates only the final risk label, and
verifiable process rewards (right), which additionally verifies each reasoning step and its associated label.


in which intermediate reasoning steps are evalu-                               where wtn , wtl ≥ 0 are preset weights.
ated by deterministic, externally checkable verifiers                             A terminal outcome reward rlabel evaluates
rather than learned neural judges. The framework                               whether the final risk value predicted from the full
is presented in the context of risk assessment tasks,                          reasoning trace matches the gold risk value. The
which naturally admit structured reasoning steps,                              full verifiable process reward is then
discrete labels, and rule-based transitions that can                                                                    T
be validated against domain guidelines.
                                                                                                                        X
                                                                                               R(Y ; x) =                     rt (Y ; x) + rlabel
                                                                                                                        t=1
3.1    Reasoning Trajectories and Steps
For an input x, the model produces a reasoning tra-                              As illustrated in Figure 2, this reward is fully
jectory Y = (o1 , . . . , oT ) with a stochastic policy                        computable using deterministic, rule-based checks,
                                                                               making all components of the reasoning trajectory
                               T
                               Y                                               verifiable.
           πθ (Y | x) =               πθ (ot | o<t , x).
                               t=1                                             3.3         Reward Separation and Optimisation
   Each step t contains two discrete outputs: (i) a                                        Guarantee
step identifier st ∈ S and (ii) a step label ℓ̂t ∈ Lt ,                        An interesting consequence of combining VPRMs
which represents the model’s answer for that step.                             with GRPO or DAPO is that the resulting opti-
Domain guidelines specify, for each prefix Y≤t ,                               misation dynamics exhibit a clear structure: rule-
the gold step identifier s⋆t and gold step label ℓ⋆t                           consistent trajectories are, in expectation, pushed
obtained by applying the rule-based logic of the                               in a beneficial direction.
task.                                                                             Let R(Y ) denote the verifiable process reward
                                                                               assigned to a response y, and let G be the number
3.2    Verifiers and Process Rewards
                                                                               of responses sampled for the input x. Let C be the
Correctness is evaluated by two bounded scoring                                event that Y is correct according to the rule-based
functions:                                                                     task semantics. Define the conditional expectations
                snt (st , s⋆t ),          slt (ℓ̂t , ℓ⋆t ),                        µc := E[R(Y ) | C],                               µi := E[R(Y ) | C c ].
each mapping a model output and its corresponding
                                                                                 We assume the following mild conditions (see
gold value into [0, 1]. These provide a positive
                                                                               Appendix A for the full formal statement and dis-
reward signal when the model selects the correct
                                                                               cussion): (i) R(Y ) has finite variance, (ii) correct
step identifier and the correct label according to the
                                                                               reasoning chains receive strictly larger expected
task rules. The instanteneous step-level verifiable
                                                                               reward than incorrect ones (µc > µi ), and (iii) a
reward is then defined as:
                                                                               sufficiently large G to ensure stable gradient up-
      rt (Y ; x) = wtn snt (st , s⋆t ) + wtl slt (ℓ̂t , ℓ⋆t ),                 dates.
 Theorem 1. Under the above hypotheses, the               Dataset                  Train   Test   Total   Avg tokens

 expected GRPO and DAPO advantage E[Â(Y )]               C OCHRANE F OREST E XT   2651     295   2946     13,596.9
                                                          C OCHRANE F OREST          –     1846   1846     12,722.8
 satisfies                                                RoBBR Cochrane            774     906   1680      9,084.6
                                                          RoBBR Non-Cochrane         –     2489   2489      7,940.7
      E[Â(Y ) | C] > 0,     E[Â(Y ) | C c ] < 0.
                                                         Table 1: Datasets statistics. Train/test split only ap-
 Thus, both GRPO and DAPO assign positive                plies to C OCHRANE F OREST E XT and RoBBR Cochrane.
 expected weight to correct reasoning trajectories       C OCHRANE F OREST and RoBBR Non-Cochrane are
 and negative weight to incorrect ones, raising the      used for testing.
 likelihood of correct reasoning in expectation.
                                                         4.3   Experimental Setup
  This follows from the theoretical results pre-
sented by Wen et al. (2025). A proof for both            Training and Evaluation Datasets. For training,
GRPO and DAPO objectives is provided in Ap-              we use the corpus constructed with the methods
pendix A.                                                from Section 4.1, allocating 2,651 instances for
                                                         training and 295 for validation. We also include the
4     Experiments                                        774 training instances from the RoBBR Cochrane
                                                         split (Wang et al., 2025a). All training data are
4.1    Training Dataset Creation
                                                         augmented with step-level labels (Section 4.2).
The first stage of our methodology involved the             For evaluation, we consider three datasets. The
acquisition of a high-quality, human-aligned corpus      first is C OCHRANE F OREST (Pronesti et al., 2025b),
suitable for training reward models.                     which contains 1,846 instances drawn from 48
   To this end, we build on the C OCHRANE F OR -         Cochrane Systematic Reviews and 202 forest plots.
EST (Pronesti et al., 2025a) and C OCHRANE F OR -        The second consists of the two test sets from the
EST E XT (Pronesti et al., 2025b) datasets, which        RoBBR benchmark: RoBBR Cochrane, which con-
provide two essential components for our task: (i)       tains 906 datapoints originating from 204 papers in-
the forest plots extracted from Cochrane systematic      cluded in 58 Cochrane reviews; and RoBBR–Non-
reviews, and (ii) the full-text primary studies corre-   Cochrane, which contains 2,489 datapoints drawn
sponding to every trial included in those plots. The     from 496 non-Cochrane reviews that collectively
inclusion of full papers is critical, as risk-related    assess 496 papers. Dataset statistics are sum-
signals often depend on methodological details           marised in Table 1, while per-risk-type statistics
available only in the complete manuscripts.              are reported in Table 6 (Appendix).
   From these corpora, we retain exclusively the
forest plots that contain an associated risk-of-bias     Evaluation Metrics. We evaluate all models
map. Each such plot establishes an explicit corre-       on the main prediction task using Accuracy and
spondence between its set of included studies and        macro–F1, computed over the discrete risk labels.
their study-level bias assessments. We therefore            In addition, for analyses (Section 4.6), we re-
define a single instance as a paper-risk pair consist-   port Coherence for the VPRM-trained models, de-
ing of a full-text study and its aligned risk-of-bias    fined as the proportion of datapoints for which the
definition and label extracted from the map. The         model’s predicted risk is consistent with the con-
resulting dataset comprises 2,946 instances drawn        clusion implied by its own intermediate reason-
from 104 systematic reviews, totalling 4M tokens.        ing. Formally, let r̂i ∈ R denote the final risk
                                                         value predicted by the model for datapoint i, and
4.2    Synthetic Data Annotation for Structural          let ℓ̂i,1 , . . . , ℓ̂i,T denote the sequence of step-level
       Reasoning Processes                               labels produced along the corresponding reasoning
Following the methodology described by Pronesti          trace. Let D : L1 × · · · × LT → R be a fixed, ex-
et al. (2025b), we enrich our dataset with step-level    ternally specified decision function mapping step-
labels for RL using LlaMa 3.1 405B (Grattafiori          level labels to a risk value. In our setting, this is the
et al., 2024) with the system prompt shown in Fig-       set of macros used in the RoB2 tool (Sterne et al.,
ure 4 (Appendix), temperature of 0.7 and 2,048           2019) (See Figure 1). The coherence indicator for
tokens generation limit. An example data instance        datapoint i is then
is provided in Table 8 (Appendix); a human verifi-                                 n                         o
cation of the generated annotations in Appendix D.                   Ci := 1 r̂i = D(ℓ̂i,1 , . . . , ℓ̂i,T )
and the dataset-level Coherence is given by             serves as a reasonable baseline for comparison. In
                                                        addition, we train a policy using MedPRM (Yun
                                 N
                             1 X                        et al., 2025) as a reward model, one of the first
             Coherence :=        Ci
                             N                          open-source PRMs for general medical reasoning.
                                i=1

By construction, Coherence measures the degree to       4.4   Main Results
which the model’s final conclusions are internally      Comparison with Pretrained Baselines. Ta-
consistent with the reasoning signals expressed in      ble 2 presents a performance comparison of pre-
its own intermediate steps.                             trained and fine-tuned language models on the
Training Setup. We conduct our training us-             C OCHRANE F OREST and RoBBR benchmarks.
ing a compact instruct models of recent release:        Across all datasets, models trained with verifi-
Qwen2.5-7B (Yang et al., 2025b). We study two           able rewards substantially outperform pretrained
methodological regimes: supervised fine-tuning          models, including large reasoning-enabled sys-
(SFT) with reasoning traces augmentation and            tems. Reinforcement learning with verifiable out-
reinforcement learning (RL) with verifiable re-         come rewards already yields strong gains over
wards. SFT is conducted for 5 epochs with a per-        supervised fine-tuning, while incorporating veri-
device batch size of 1, a learning rate of 5 × 10−5 ,   fiable process rewards consistently leads to fur-
and the AdamW optimiser (Loshchilov and Hut-            ther improvements in both accuracy and macro-F1.
ter, 2017). For RL, we investigate two policy-          On C OCHRANE F OREST, Qwen2.5-7B trained with
optimisation algorithms, GRPO (Shao et al., 2024)       VPRM achieves the best overall performance, and
and DAPO (Yu et al., 2025), combined with two           similar improvements are observed on both RoBBR
reward types: verifiable outcome reward and our         Cochrane and Non-Cochrane. The latter result in-
verifiable process reward approach. All RL config-      dicates that the benefits of verifiable process super-
urations are trained for 3 epochs with a learning       vision generalise beyond the training distribution,
rate of 1 × 10−6 , per-device batch size 1, and 16      rather than exploiting dataset-specific regularities.
sampled generations per batch. Further implemen-        Comparison with Neural PRMs. Table 3 com-
tation details are provided in Appendix F.              pares verifiable process rewards against neural
Model Baselines. To validate our results, we            process-reward baselines. In all settings, mod-
compare a range of open- and closed-source mod-         els also receive the same verifiable outcome re-
els, with and without reasoning capabilities. mod-      ward; the comparison isolates only the effect of
els are evaluated in zero-shot settings with prompt     the process-level supervision. While neural PRMs
and hyperparameters shown in Appendix C. We in-         substantially improve over outcome-only train-
clude three main model families: Qwen 2.5 (Yang         ing, they are consistently outperformed by VPRM.
et al., 2025b), Llama 3.1 (Grattafiori et al., 2024),   This performance gap suggests that learned step-
and Granite 3.1 (Granite Team, 2024). In addition,      level judges introduce noise and misalignment that
we benchmark the distilled Qwen and Llama mod-          limit their effectiveness, whereas deterministic,
els derived from DeepSeek-R1 (Guo et al., 2025).        guideline-based verification provides a cleaner and
Lastly, we include one closed-source and two open-      more reliable optimisation signal. These results
source models from OpenAI (OpenAI et al., 2025).        support the central claim that verifiable process re-
   To contextualise the effectiveness of our verifi-    wards offer a stronger and more robust alternative
able reward formulation, we also evaluate neural        to neural process supervision for complex, struc-
process-reward baselines. At present, no pretrained     tured reasoning tasks.
PRM exists for risk-of-bias assessment or, more
broadly, for non-mathematical scientific reasoning.     4.5   Ablation Studies
Therefore, to approximate a general-purpose PRM,        To assess the contribution of different components
we follow prior work on using LLMs as step-level        of our verifiable reward formulation, we conduct
judges, prompting a model to assign correctness         two ablation studies: the structure of process super-
scores to each reasoning step (Song et al., 2025).      vision and the inclusion of an outcome-level reward.
This setup has been shown to deliver competitive        For process supervision, we compare a steps-only
process-level feedback in domains where explicit        reward that verifies whether the model follows the
PRM training data is unavailable, and therefore         correct sequence of reasoning steps, irrespective
                                             C OCHRANE F OREST RoBBR Cochrane RoBBR Non-Cochrane
      Model                         Think
                                             Acc      F1       Acc     F1     Acc       F1
               Pretrained LLMs
      GPT-4-0125                      ✗      52.4     41.6      56.0       47.9     47.8         42.3
      GPT-OSS-20B                     ✓      61.4     43.9      56.4       50.3     46.3         42.8
      GPT-OSS-120B                    ✓      67.1     49.8      59.5       51.0     48.8         44.2
      Qwen2.5-7B                      ✗      32.9     31.6      35.8       34.1     36.4         34.5
      Qwen2.5-14B                     ✗      39.0     35.1      37.0       35.5     35.4         32.5
      Qwen2.5-72B                     ✗      51.3     42.1      56.1       51.0     47.5         43.6
      Llama-3.1-8B                    ✗      36.4     30.6      34.5       32.1     36.4         32.5
      Llama-3.1-70B                   ✗      38.8     30.2      49.5       40.0     42.5         38.9
      Llama-3.1-405B                  ✗      68.4     45.5      59.4       44.0     52.5         39.8
      DeepSeek-Qwen-7B                ✓       –        –         –          –        –            –
      DeepSeek-Qwen-14B               ✓      33.3     19.2      35.8       23.5     35.4         23.5
      DeepSeek-Qwen-32B               ✓      40.8     35.9      44.9       40.4     46.4         41.3
      DeepSeek-Llama-8B               ✓       –        –         –          –        –            –
      DeepSeek-Llama-70B              ✓      44.2     33.5      57.3       41.2     48.3         42.7
      Granite-3.1-3B                  ✗      24.4     23.6      22.2       21.8     13.7         14.9
      Granite-3.1-8B                  ✗      24.7     22.0      35.8       31.6     33.2         28.2
      Granite-4.0-h-small (32B)       ✗      48.2     33.5      45.4       41.2     40.9         33.1
                 Our Models
      Qwen2.5-7B-SFT                  ✓      45.1     36.9      38.6       32.4     38.3         31.9
      Qwen2.5-7B-GRPO                 ✓      81.5     70.2      63.1       58.0     56.8         45.1
      Qwen2.5-7B-DAPO                 ✓      76.8     57.3      60.2       45.4     55.8         43.6
      Qwen2.5-7B-GRPO-VPRM            ✓      87.9     76.7      65.2       58.5     60.7         47.2
      Qwen2.5-7B-DAPO-VPRM            ✓      79.2     60.6      60.7       48.9     57.1         45.3

Table 2: Evaluation results across models on three datasets, reporting Accuracy and macro-F1. “–” denotes
unparsable or inconclusive outputs. Best results are bolded; second-best are underlined.


     Method                           Acc      F1                Setting                       Acc      F1
     Neural PRMs                                                 w/o Outcome Reward
     Qwen2.5-7B-GRPO-PRM-GPT-OSS      78.2    56.1
     Qwen2.5-7B-GRPO-MedPRM           76.8    53.4                 Steps-only process reward   34.4     32.3
                                                                   Full VPRM                   40.2     35.3
     Verifiable Rewards
     Qwen2.5-7B                       32.9    31.6               w/ Outcome Reward
     Qwen2.5-7B-GRPO                  81.5    70.2                 Steps-only process reward   83.1     71.8
     Qwen2.5-7B-GRPO-VPRM             87.9    76.7                 Full VPRM                   87.9     76.7

Table 3: Performance comparison between neural            Table 4: Ablation study on outcome and process reward
judges, rule-based rewarding and verifiable process re-   components on C OCHRANE F OREST. We report Accu-
warding on C OCHRANE F OREST.                             racy (Acc) and F1; lower blocks compare models with
                                                          and without outcome reward, and columns compare
                                                          steps-only supervision to the full VPRM formulation.
of the correctness of their content, against the full
VPRM, which additionally evaluates the correct-
ness of each step and enforces consistency with the       steps-only variant, demonstrating the importance
guideline-defined decision structure. For outcome         of verifying not just the presence but also the cor-
supervision, we train each variant both with and          rectness and logical composition of intermediate
without a verifiable outcome reward.                      reasoning steps. When the outcome reward is in-
   Table 4 shows that removing the outcome reward         cluded, performance improves markedly, and the
leads to substantial performance degradation, indi-       full VPRM consistently achieves the best results,
cating that step-structure verification alone is insuf-   showing that combining verifiable outcome super-
ficient to reliably optimize the task. Nevertheless,      vision with fine-grained, correctness-aware process
even in this setting, the full VPRM outperforms the       rewards yields the strongest learning signal.
4.6    Analyses                                                              Model                                Coherence          CA

Impact of Thought Process. Table 5 reports Co-                               GPT-OSS-120B                               36.2         28.5
                                                                             Qwen2.5-72B                                44.3         24.9
herence on the C OCHRANE F OREST testset for the                             Llama-3.1-405B                             50.7         27.1
models trained with VPRMs compared with pre-                                 Qwen2.5-7B-GRPO-VPRM                       89.5         75.0
trained baselines prompted to output process labels.                         Qwen2.5-7B-DAPO-VPRM                       80.1         69.4
In addition, we report Coherent Accuracy (CA),
defined as the accuracy restricted to coherent in-       Table 5: Coherence scores for Qwen models trained
                                                         with DAPO and GRPO using verifiable process rewards
stances. That is, among the datapoints for which
                                                         compared with pretrained LLMs on C OCHRANE F OR -
the model’s final prediction is consistent with the      EST. Best results are bolded; second-best underlined.
decision implied by its own reasoning steps (i.e.,
Ci = 1), we measure the proportion whose final                                       Process Reward   Accuracy Reward     Thought Format Reward
                                                                            1
predicted label is also correct. Formally, letting ŷi
and yi denote the predicted and gold labels respec-                        0.8




                                                         Training Reward
tively, we define
                                                                           0.6

                        1[Ci = 1 ∧ ŷi = yi ]
               PN
                  i=1                                                      0.4
        CA =
                              N
                                                                           0.2
CA therefore quantifies the reliability of the                                   0    200 400 600 800 1,000 1,200 1,400 1,600 1,800 2,000
model’s predictions conditioned on coherence.                                                         Step
   Results show that pretrained models exhibit
low coherence and very low CA, indicating that           Figure 3: Reward dynamics. Format rewards plateau
                                                         early, while accuracy and process rewards improve grad-
even when their step-level reasoning appears self-
                                                         ually, indicating that LLMs quickly learn structure but
consistent, it rarely leads to correct final judg-       continue to refine quality.
ments. In contrast, VPRM-trained models achieve
both substantially higher coherence and high CA,
demonstrating that they not only follow the de-          guide reasoning trajectories (Wang et al., 2025c;
cision logic faithfully but also produce accurate        Zeng et al., 2025; Wen et al., 2025). While
conclusions when they do so, suggesting improved         effective, these methods remain centered on
robustness and interpretability.                         terminal-outcome verification. In parallel, process-
                                                         supervision approaches score intermediate reason-
Reward Dynamics. Figure 3 shows that process             ing steps using neural judges (Lightman et al.,
and correctness rewards follow closely aligned tra-      2024; Zhang et al., 2025; Zou et al., 2025), provid-
jectories: both rise sharply early, stabilize within     ing dense feedback but relying on non-verifiable
the same oscillatory range, and peak at similar          model-based evaluations prone to bias and reward
points. This alignment indicates that improved           hacking (Amodei et al., 2016; Skalse et al., 2022).
step-level reasoning directly improves final-label       We bridge this gap by combining RLVR with step-
correctness. In contrast, the thought-format reward      wise rule-based verification, enabling transparent
saturates quickly and remains flat, contributing lit-    and verifiable process supervision.
tle once formatting is learned. Overall, the strongly
correlated shapes of the process and correctness         6                 Conclusion
curves highlight that VPRM training drives coher-
                                                         In this paper, we introduce verifiable process re-
ent, mutually reinforcing gains in intermediate and
                                                         wards that integrate deterministic step-level veri-
final reasoning behaviour.
                                                         fication with reinforcement learning, provide the-
5     Related Work                                       oretical guarantees under mild assumptions, and
                                                         demonstrate substantial empirical gains on risk-of-
For a more detailed overview of related work,            bias assessment in medical systematic reviews.
refer to Appendix E. Prior work extends rein-               Our results indicate that verifiable process super-
forcement learning with verifiable rewards (RLVR)        vision is a practical and robust approach to induc-
beyond outcome-only supervision by adding                ing reliable reasoning behaviour in large language
structural signals, such as masked-and-reordered         models, opening the door to broader applications
self-supervision or self-verification modules that       in structured scientific and decision-making tasks.
Limitations                                               Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song,
                                                            Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma,
While VPRMs offer strong guarantees for struc-              Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai
tured reasoning tasks, several limitations remain.          Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao,
                                                            Zhuoshu Li, Ziyi Gao, Aixin Liu, and 180 others.
First, the approach relies on the existence of de-
                                                            2025. Deepseek-r1: Incentivizing reasoning capa-
terministic, domain-specific rules; tasks lacking           bility in LLMs via reinforcement learning. Preprint,
well-defined intermediate reasoning steps may not           arXiv:2501.12948.
benefit directly. Second, our empirical evaluation is
currently focused on risk-of-bias assessment; gener-      Jiajie Huang, Honghao Lai, Weilong Zhao, Danni Xia,
                                                             Chunyang Bai, Mingyao Sun, Jianing Liu, Jiayi Liu,
alisation to other domains, particularly open-ended          Bei Pan, Jinhui Tian, and 1 others. 2025. Large
reasoning tasks, remains to be established.                  language model–assisted risk-of-bias assessment in
   Additionally, the approach assumes that the               randomized controlled trials using the revised risk-
model can produce reasoning traces in a format               of-bias tool: Usability study. Journal of Medical
                                                             Internet Research, 27:e70450.
compatible with the verifiers; misalignment be-
tween model output and verifier expectations could        Hugging Face. 2025. Open R1: A fully open reproduc-
reduce reward effectiveness, especially in the con-         tion of deepseek-r1.
text of smaller models. Finally, while VPRMs re-
                                                          Changkai Ji, Bowen Zhao, Zhuoyao Wang, Yingwen
duce reliance on neural reward models, they do not          Wang, Yuejie Zhang, Ying Cheng, Rui Feng, and
fully eliminate other sources of model bias or errors       Xiaobo Zhang. 2025. Robguard: Enhancing llms
arising from incomplete guidelines. Addressing              to assess risk of bias in clinical trial documents. In
these challenges will be critical for deploying ver-        Proceedings of the 31st International Conference on
                                                            Computational Linguistics, pages 1258–1277.
ifiable process supervision in broader, real-world
applications.                                             Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying
                                                           Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gon-
                                                           zalez, Hao Zhang, and Ion Stoica. 2023. Efficient
References                                                 memory management for large language model serv-
                                                           ing with pagedattention. In Proceedings of the 29th
Dario Amodei, Chris Olah, Jacob Steinhardt, Paul           symposium on operating systems principles, pages
  Christiano, John Schulman, and Dan Mané. 2016.           611–626.
  Concrete problems in ai safety. arXiv preprint
  arXiv:1606.06565.                                       Honghao Lai, Jiayi Liu, Chunyang Bai, Hui Liu, Bei
                                                            Pan, Xufei Luo, Liangying Hou, Weilong Zhao,
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie            Danni Xia, Jinhui Tian, and 1 others. 2025. Lan-
  Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind          guage models for data extraction and risk of bias
  Neelakantan, Pranav Shyam, Girish Sastry, Amanda          assessment in complementary medicine. npj Digital
  Askell, and 1 others. 2020. Language models are           Medicine, 8(1):74.
  few-shot learners. Advances in Neural Information
  Processing Systems (NeurIPS), 33.                       Nathan Lambert, Jacob Morrison, Valentina Pyatkin,
                                                            Shengyi Huang, Hamish Ivison, Faeze Brahman,
Jacqueline Chandler, Miranda Cumpston, Tianjing             Lester James Validad Miranda, Alisa Liu, Nouha
   Li, Matthew J. Page, and Vivian A. Welch. 2019.          Dziri, Xinxi Lyu, Yuling Gu, Saumya Malik, Victoria
   Cochrane handbook for systematic reviews of inter-       Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le
  ventions. Hoboken: Wiley, 4.                              Bras, Oyvind Tafjord, Christopher Wilhelm, Luca
                                                            Soldaini, and 4 others. 2025. Tulu 3: Pushing fron-
Abel Corrêa Dias, Viviane Pereira Moreira, and João         tiers in open language model post-training. In Second
  Luiz Dihl Comba. 2025. Robin: A transformer-based         Conference on Language Modeling.
  model for risk of bias inference with machine reading
  comprehension. Journal of Biomedical Informatics,       Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harri-
  166:104819.                                               son Edwards, Bowen Baker, Teddy Lee, Jan Leike,
                                                            John Schulman, Ilya Sutskever, and Karl Cobbe.
IBM Granite Team. 2024. Granite 3.0 language mod-           2024. Let’s verify step by step. In The Twelfth Inter-
  els. URL: https://github.com/ibm-granite/granite-         national Conference on Learning Representations.
  3.0-language-models.
                                                          Ilya Loshchilov and Frank Hutter. 2017. Decoupled
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri,          weight decay regularization. In International Confer-
  Abhinav Pandey, Abhishek Kadian, Ahmad Al-                 ence on Learning Representations.
  Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten,
  Alex Vaughan, and 1 others. 2024. The LLaMa 3           OpenAI. 2024. Introducing openai o1.          https://
  herd of models. arXiv preprint arXiv:2407.21783.          openai.com/it-IT/o1/.
OpenAI. 2025. Introducing gpt-5. https://openai.           Jianyou Wang, Weili Cao, Longtian Bao, Youze Zheng,
  com/it-IT/gpt-5/.                                           Gil Pasternak, Kaicheng Wang, Xiaoyue Wang, Ra-
                                                              mamohan Paturi, and Leon Bergen. 2025a. Measur-
OpenAI, Sandhini Agarwal, Lama Ahmad, Jason                   ing risk of bias in biomedical reports: The RoBBR
  Ai, Sam Altman, Andy Applebaum, Edwin Ar-                   benchmark. In Proceedings of the 2025 Conference
  bus, Rahul K. Arora, Yu Bai, Bowen Baker, Haim-             on Empirical Methods in Natural Language Process-
  ing Bao, Boaz Barak, Ally Bennett, Tyler Bertao,            ing, pages 3220–3248, Suzhou, China. Association
  Nivedita Brett, Eugene Brevdo, Greg Brockman, Se-           for Computational Linguistics.
  bastien Bubeck, Che Chang, and 107 others. 2025.
  gpt-oss-120b & gpt-oss-20b model card. Preprint,         Yiping Wang, Qing Yang, Zhiyuan Zeng, Liliang Ren,
  arXiv:2508.10925.                                          Liyuan Liu, Baolin Peng, Hao Cheng, Xuehai He,
Massimiliano Pronesti, Joao H Bettencourt-Silva, Paul        Kuan Wang, Jianfeng Gao, Weizhu Chen, Shuohang
 Flanagan, Alessandra Pascale, Oisín Redmond, Anya           Wang, Simon Shaolei Du, and yelong shen. 2025b.
 Belz, and Yufang Hou. 2025a.          Query-driven          Reinforcement learning for reasoning in large lan-
 document-level scientific evidence extraction from          guage models with one training example. In The
 biomedical studies. In Proceedings of the 63rd An-          Thirty-ninth Annual Conference on Neural Informa-
 nual Meeting of the Association for Computational           tion Processing Systems.
 Linguistics (Volume 1: Long Papers), pages 28034–
 28051, Vienna, Austria. Association for Computa-          Zhen Wang, Zhifeng Gao, and Guolin Ke. 2025c.
 tional Linguistics.                                         Masked-and-reordered self-supervision for reinforce-
                                                             ment learning from verifiable rewards. arXiv preprint
Massimiliano Pronesti, Michela Lorandi, Paul Flanagan,       arXiv:2511.17473.
  Oisín Redmond, Anya Belz, and Yufang Hou. 2025b.
  Enhancing study-level inference from clinical trial      Xumeng Wen, Zihan Liu, Shun Zheng, Zhijian Xu,
  papers via reinforcement learning-based numeric rea-       Shengyu Ye, Zhirong Wu, and 1 others. 2025. Rein-
  soning. In Proceedings of the 2025 Conference on           forcement learning with verifiable rewards implicitly
 Empirical Methods in Natural Language Processing,           incentivizes correct reasoning in base llms. arXiv
  pages 30345–30361, Suzhou, China. Association for          preprint arXiv:2506.14245.
  Computational Linguistics.
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu,            An Yang, Anfeng Li, Baosong Yang, Beichen Zhang,
  Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan             Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao,
  Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024.                Chengen Huang, Chenxu Lv, Chujie Zheng, Day-
  DeepSeekMath: Pushing the limits of mathemati-             iheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao
  cal reasoning in open language models. Preprint,           Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41
  arXiv:2402.03300.                                          others. 2025a. Qwen3 technical report. Preprint,
                                                             arXiv:2505.09388.
Joar Skalse, Nikolaus Howe, Dmitrii Krasheninnikov,
  and David Krueger. 2022. Defining and characteriz-       An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui,
  ing reward gaming. Advances in Neural Information          Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu,
  Processing Systems, 35:9460–9471.                          Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jian-
                                                             hong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang,
Mingyang Song, Zhaochen Su, Xiaoye Qu, Jiawei Zhou,          Jingren Zhou, Junyang Lin, Kai Dang, and 23 oth-
  and Yu Cheng. 2025. PRMBench: A fine-grained               ers. 2025b. Qwen2.5 technical report. Preprint,
  and challenging benchmark for process-level reward         arXiv:2412.15115.
  models. In Proceedings of the 63rd Annual Meeting
  of the Association for Computational Linguistics (Vol-   Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xi-
  ume 1: Long Papers), pages 25299–25346, Vienna,            aochen Zuo, YuYue, Weinan Dai, Tiantian Fan, Gao-
  Austria. Association for Computational Linguistics.        hong Liu, Juncai Liu, LingJun Liu, Xin Liu, Haibin
Jonathan AC Sterne, Jelena Savović, Matthew J.              Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan
  Page, Roy G. Elbers, Natalie S. Blencowe, Isabelle         Tong, Chi Zhang, Mofan Zhang, and 17 others. 2025.
  Boutron, Christopher J. Cates, He Cheng, Mark S.           DAPO: An open-source LLM reinforcement learning
  Corbett, Sandra M. Eldridge, Miguel A. Hernán,             system at scale. In The Thirty-ninth Annual Confer-
  Sally Hopewell, Asbjørn Hróbjartsson, Diana R. Jun-        ence on Neural Information Processing Systems.
  queira, Peter Jüni, Jamie J. Kirkham, Toby Lasser-
  son, Tianjing Li, Ann McAleenan, and 8 others. 2019.     Jaehoon Yun, Jiwoong Sohn, Jungwoo Park, Hyunjae
  RoB 2: a revised tool for assessing risk of bias in         Kim, Xiangru Tang, Daniel Shao, Yong Hoe Koo,
  randomised trials. BMJ, 366:l4898.                          Ko Minhyeok, Qingyu Chen, Mark Gerstein, Michael
                                                              Moor, and Jaewoo Kang. 2025. Med-PRM: Medical
Simon Šuster, Timothy Baldwin, and Karin Verspoor.            reasoning models with stepwise, guideline-verified
  2024. Zero-and few-shot prompting of generative             process rewards. In Proceedings of the 2025 Con-
  large language models provides weak assessment of           ference on Empirical Methods in Natural Language
  risk of bias in clinical trials. Research Synthesis        Processing, pages 16565–16582, Suzhou, China. As-
  Methods, 15(6):988–1000.                                    sociation for Computational Linguistics.
Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Good-       Assumption A3 (Concentration of group statis-
   man. 2022. STar: Bootstrapping reasoning with rea-    tics). Let R and S denote the empirical mean
   soning. In Advances in Neural Information Process-
                                                         and standard deviation of rewards within a sampled
   ing Systems.
                                                         group. As the group size G → ∞,
Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu,
 Keqing He, Zejun MA, and Junxian He. 2025.
                                                                   p               p         p
 SimpleRL-zoo: Investigating and taming zero rein-             R→
                                                                − m,            S→
                                                                                 − σ :=       Var[R(Y )].
 forcement learning for open base models in the wild.
 In Second Conference on Language Modeling.              A.2    Normalised advantages
Jixiao Zhang and Chunsheng Zuo. 2025. GRPO-LEAD:         GRPO uses the trajectory-level normalised advan-
   A difficulty-aware reinforcement learning approach    tage
   for concise mathematical reasoning in language mod-                           R(Y ) − R
   els. In Proceedings of the 2025 Conference on Em-                   Â(Y ) =            .
   pirical Methods in Natural Language Processing,                                   S
   pages 5642–5665, Suzhou, China. Association for       DAPO constructs token-level advantages by scaling
   Computational Linguistics.                            the same trajectory advantage:
Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen                          Âi,t = ci,t Â(Yi ),
  Zhang, Runji Lin, Bowen Yu, Dayiheng Liu, Jin-
                                                         where ci,t ≥ 0 and P 1|Yi | i,t ci,t = 1. Follow-
                                                                                       P
  gren Zhou, and Junyang Lin. 2025. The lessons of
                                                                                 i
  developing process reward models in mathematical       ing Wen et al., 2025 and without loss of generality,
  reasoning. In Findings of the Association for Compu-
  tational Linguistics: ACL 2025, pages 10495–10516,     we consider a policy gradient update
  Vienna, Austria. Association for Computational Lin-                    G
  guistics.                                                            1X
                                                               ∇J(θ) ≈     Â(Yi )∇θ log πθ (Yi | x).
                                                                       G
Jiaru Zou, Ling Yang, Jingwen Gu, Jiahao Qiu, Ke Shen,                       i=1
   Jingrui He, and Mengdi Wang. 2025. Reasonflux-        A.3    Proof of Theorem 1
   PRM: Trajectory-aware PRMs for long chain-of-
   thought reasoning in LLMs. In The Thirty-ninth        We prove the result for Â(Y ); the DAPO case fol-
   Annual Conference on Neural Information Process-      lows immediately by the nonnegativity of the con-
   ing Systems.                                          stants ci,t .
A     Theoretical Analysis                                  By Assumption A3 and Slutsky’s theorem,

This section provides the proof of Theorem 1,                           R(Y ) − R d R(Y ) − m
                                                                 Â(Y ) =           −
                                                                                    →           .
which is an extension of prior results on reinforce-                         S               σ
ment learning with verifiable rewards (RLVR) in          Taking conditional expectations yields
base LLMs (Wen et al., 2025).                                                            µc − m
                                                                                       G→∞
                                                                       E[Â(Y ) | C] −−−−→      ,
A.1    Setup and assumptions                                                                σ
                                                                                    G→∞ µi − m
Let R(Y ) denote the verifiable process reward as-                E[Â(Y ) | C c ] −−−−→        .
                                                                                            σ
signed to a response Y , and let G be the number           Substituting m = pµc + (1 − p)µi gives
of responses sampled for the input x. Let C denote
the correctness event for a trajectory Y . Define                  µc − m = (1 − p)(µc − µi ) > 0,
                                                                         µi − m = −p(µc − µi ) < 0,
    µc := E[R(Y ) | C],      µi := E[R(Y ) | C c ],
                 p := P(C),                                establishing the sign separation:

   and let m := pµc + (1 − p)µi be the uncondi-                              E[Â(Y ) | C] > 0,
tional expected reward.                                                     E[Â(Y ) | C c ] < 0.
Assumption A1 (Verifiability and finite vari-              For DAPO,
ance). For fixed (x, r), the verifiable reward
R(Y ) is a real-valued random variable with finite                 E[Âi,t | C] = ci,t E[Â(Y ) | C] > 0,
mean and nonnegative variance σY > 0.                            E[Âi,t | C c ] = ci,t E[Â(Y ) | C c ] < 0.
Assumption A2 (Reward gap). Correct reason-                 Thus, both GRPO and DAPO apply positive ex-
ing trajectories have higher probabilities to induce     pected weight to correct traces and negative weight
correct answers: µc > µi .                               to incorrect ones, proving Theorem 1.
A.4    Verifiable Outcome Reward as a Special          cealment protects against selection bias, whereas
       Case of VPRMs                                   inadequate or unclear procedures raise concerns.
                                        (r)
Define the degenerate label spaces Lt = ∅ for          C. Blinding of participants and personnel This
all t < T . Then rt (·) = 0 for t < T and              domain considers whether participants and those
                                                       administering interventions were aware of group
          R(Y ; x, r) = rlabel (Y ; x, r).
                                                       assignments. Reviewers establish whether blind-
Hence a Verifiable Outcome Reward Model is ex-         ing was reported, whether it involved participants,
actly the case of a VPRM with no intermediate          personnel, or both, and whether the blinding ap-
verifiable labels. All the results above apply: they   proach was likely to have been effective. Lack of
reduce to the original GRPO/DAPO statements            blinding, or ineffective procedures, may influence
where the scalar reward depends only on final out-     participants’ behaviour or care delivery and thus
come statistics.                                       introduce performance bias.

B     Risk of Bias Assessment                          D. Blinding of outcome assessment Assessors
                                                       may also be influenced by knowledge of treatment
Risk-of-bias estimation evaluates the extent to        allocation. Reviewers check whether outcome as-
which study findings may be systematically dis-        sessors were blinded and whether blinding was
torted. The process is organised into a set of do-     likely to minimise biased measurement. Absence
mains that correspond to common sources of bias        of blinding or unclear reporting raises the possibil-
in randomized trials. For each domain, reviewers       ity that assessments were influenced by expecta-
extract relevant information from the study report     tions or prior beliefs.
and translate it into qualitative judgments about
the presence and potential impact of bias. Modern      E. Incomplete outcome data This domain eval-
assessment tools, such as RoB 2.0 (Sterne et al.,      uates the extent and handling of missing data.
2019), increasingly leverage automated decision        Reviewers consider how much data is missing,
rules to standardise these judgments. Algorithm 1      whether reasons for missingness are reported and
provides an example of a macro for risk of type A      plausible, and whether the analysis appropriately
using the labels defined in this paper, which illus-   accounts for missing data. High or unexplained
trates how extracted steps are mapped to specific      attrition, or inadequate handling strategies, can pro-
risk levels. Below we outline the main domains         duce biased estimates of effect.
considered in our work, together with the typical      F. Selective reporting Selective reporting bias
reasoning steps involved.                              arises when outcomes are reported inconsistently
A. Random sequence generation This domain              with the study protocol or when unplanned out-
assesses whether the method used to generate the       comes are introduced. Reviewers check whether
allocation sequence was truly random. Reviewers        a protocol is available, compare planned and re-
first check whether the study reports how random-      ported outcomes, and assess whether omissions or
ization was carried out. If so, they evaluate the      additions suggest selective emphasis. Clear corre-
nature of the method (e.g., computer-generated se-     spondence indicates low risk; discrepancies raise
quence versus quasi-random methods such as alter-      concern.
nation) and judge whether the sequence could have      In addition to these core domains, we also consider
been predicted. Clearly reported and genuinely ran-    supplementary aspects relevant to internal validity:
dom procedures indicate low risk; quasi-random or      similarity of baseline outcomes (G), similarity of
non-random procedures, or a lack of information,       baseline characteristics (H), and risk of contamina-
increase concern.                                      tion between study arms (I). These domains capture
B. Allocation concealment Here the question is         further sources of potential bias arising from imbal-
whether the assignment to treatment groups was         ances at baseline or from unintended exposure to
shielded from those enrolling participants. Review-    interventions across groups.
ers determine whether concealment was reported
                                                       C    Prompts
and whether the method (e.g., sealed opaque en-
velopes, central allocation) prevented foreknowl-      The prompts used for synthetic data annotation and
edge of upcoming assignments. Adequate con-            for training are shown in Figure 4 and 6, respec-
Algorithm 1 RoB A Macro                                                      Prompt for synthetic data annotation
1: procedure P REDICT L ABEL -A(steps)
2:    if steps[I DENTIFY R ANDOMIZATION R EPORT] = N OT R EPORTED then       Articles: {articles}
3:          return M ODERATE
4:    end if                                                                 Your task is to produce a structured reasoning trace for the
5:    if steps[C LASSIFY R ANDOMIZATION M ETHOD] = N ON R ANDOM then         following risk of bias domain to justify the ground truth value.
6:          return H IGH
7:    end if                                                                 Comparison: {comparison}
8:    if steps[A SSESS S EQUENCE P REDICTABILITY] = P REDICTABLE then        Outcome: {outcome}
9:          return M ODERATE                                                 Bias: {bias_id} – {bias_definition}
10:     end if                                                               Ground_truth: {bias_value}
11:     if steps[BASELINE I MBALANCE] = L IKELY then
12:          return H IGH                                                    You must follow the structured reasoning procedure defined
13:     end if                                                               for each risk-of-bias domain (A–I). For every domain, you must use
14:     return L OW                                                          the exact step names and allowable categorical labels listed below:
15: end procedure
                                                                             {steps_and_labels}

 Dataset                  A     B     C     D     E     F     G    H    I    Follow this exact output structure:
 C OCHRANE F OREST E XT   498   498   498   498   498   273   61   61   61   {{
 C OCHRANE F OREST        330   330   330   330   330   112   28   28   28   "step_name": "step_label",
 RoBBR Cochrane           125   125   198   133   206   119    0    0    0   "step_name_rationale": "your detailed rationale",
 RoBBR Non-Cochrane       412   474   467   472   478   186    0    0    0   ...
                                                                             (repeat for all steps required by the bias domain)
                                                                             }}
           Table 6: Datasets statistics per risk type.

                                                                             Figure 4: Prompt for synthetic data annotation.
tively. For training, a temperature of 0.7 and 2,048
tokens as maximum output length are used.                                    Steps and labels

                                                                             A — Random sequence generation
D     Silver Steps and Labels Manual                                         Identify_randomization_report → reported | not_reported
      Verification                                                           Classify_randomization_method → random | non_random
                                                                             Assess_sequence_predictability → unpredictable | predictable
                                                                             Baseline_imbalance → likely | none
To assess the quality of the automatically generated
                                                                             B — Allocation concealment
reasoning steps and silver labels used for VPRM                              Identify_concealment_report → reported | not_reported
training, we selected a random sample of 20 in-                              Determine_concealment_method → adequate | inadequate
                                                                             Assess_possibility_of_foreknowledge → no | possible
stances from the full dataset and manually eval-
                                                                             C — Blinding of participants and personnel
uated their correctness. The evaluation was con-                             Identify_blinding_report → reported | not_reported
ducted by two master’s students in NLP familiar                              Assess_blinding_status → participants | personnel | both | none
                                                                             Evaluate_blinding_effectiveness → effective | ineffective
with the task. For each instance, annotators in-
                                                                             D — Blinding of outcome assessment
spected the complete step-level reasoning trace and                          Identify_outcome_blinding_report → reported | not_reported
                                                                             Assess_assessor_blinding → yes | no
verified two properties: (i) whether the sequence                            Evaluate_blinding_effect_on_measurement → no | possible
of steps constituted a valid decision path for the
                                                                             E — Incomplete outcome data
target risk-of-bias domain, according to Cochrane’s                          Quantify_missing_data → none | low | high
                                                                             Identify_missing_data_reason → adequate | inadequate |
domain-specific guidelines; and (ii) whether each                            not_reported
step and label was valid given the underlying paper.                         Assess_handling_of_missing_data → appropriate | inappropriate
                                                                             Estimate_bias_due_to_missing_data → unlikely | likely
For each item, we recorded whether the overall
                                                                             F — Selective reporting
reasoning trace was coherent, and for each individ-                          Identify_protocol_availability → available | not_available
ual step we recorded whether the step and its label                          Compare_outcomes_reported → all | partial | none
                                                                             Detect_unexpected_outcomes → none | added
were valid.                                                                  Evaluate_reporting_selectivity → no | possible | yes

                                                                             G — Baseline outcomes similar
                 Metric                      Fraction                        Identify_baseline_outcomes_report → reported | not_reported
                                                                             Compare_baseline_outcomes → similar | different
                 Coherent instances          100.0 %                         Evaluate_impact_of_differences → likely_impact | unlikely_impact
                 Correct steps               100.0 %
                                                                             H — Baseline characteristics similar
                 Correct labels               96.7%                          Identify_baseline_characteristics_report → reported | not_reported
                                                                             Compare_baseline_characteristics → similar | different
                                                                             Evaluate_impact_of_differences → likely_impact | unlikely_impact
Table 7: Manual verification of 20 randomly sampled
silver-labelled reasoning traces.                                            I — Contamination
                                                                             Identify_contamination_risk_report → reported | not_reported
                                                                             Assess_contamination_possibility → possible | unlikely
                                                                             Assess_contamination_impact → likely_impact | unlikely_impact
   Table 7 summarises the proportion of coherent
instances, correct steps, and correct labels observed
in this manual evaluation. Results of the manual
                                                                                            Figure 5: Steps and labels.
verification show that the automatically generated
    Prompt for training and inference                                     able to outcome-only RL, they depend on model-
    Articles: {articles}                                                  generated evaluations and therefore inherit issues
                                                                          of opacity, bias, and reward hacking (Amodei et al.,
    Question: Based on the given article, what is the risk of
    bias for the following Comparison and Outcome?                        2016; Skalse et al., 2022). Crucially, their interme-
    Comparison: {comparison}
    Outcome: {outcome}                                                    diate rewards are not verifiable.
    The bias you have to assess is defined as follows: {bias}                Taken together, these works highlight two re-
    You must follow the structured reasoning procedure defined for each
    risk-of-bias domain (A–I). For every domain, you must use the exact   maining gaps: existing approaches lack (i) verifia-
    step names and allowable categorical labels listed below:
                                                                          bility of intermediate rewards and (ii) fine-grained
    {steps_and_labels}
                                                                          step-level supervision grounded in deterministic
    Follow this exact structure for your reasoning:                       rules. To date, no method provides reinforce-
    <think>
    Step 1: step_name                                                     ment learning over reasoning trajectories where
    ...your thought process here...
    Answer: step_label                                                    every step is evaluated by an externally check-
                                                                          able verifier. Verifiable Process Reward Models
    Step 2: step_name
    ...your thought process here...                                       (VPRMs) address this gap by combining the ro-
    Answer: step_label
                                                                          bustness of RLVR with step-wise, rule-based veri-
    (repeat for all steps required by the bias domain)                    fication, which not only enables transparent, struc-
    </think>
    <answer>                                                              turally aligned reasoning but also removes the op-
    risk: high | low | moderate
    </answer>                                                             portunities for reward hacking inherent in neural
                                                                          process rewards.
 Figure 6: Prompt for VPRM-training and -inference.                       RoB Assessment and Automated Evidence Eval-
                                                                          uation. Prior work on automated RoB assess-
                                                                          ment has largely relied on supervised modelling or
steps and labels used for VPRM training are of                            prompted LLMs. Transformer-based systems such
consistently high quality. All inspected traces fol-                      as RoBIn (Dias et al., 2025) frame RoB inference
low the correct decision structure, and step-level                        as a machine reading comprehension task and train
labels are almost always accurate, providing a solid                      classifiers directly on annotated evidence. Other ap-
ground for model training.                                                proaches enhance pretrained LLMs with retrieval
                                                                          or auxiliary decision heads, as in RoBGuard (Ji
E    Related Work                                                         et al., 2025). Several studies investigate LLM
Reinforcement Learning and Verifiable Re-                                 prompting for RoB assessment, reporting limited
wards Several extensions of reinforcement learn-                          reliability when models operate without explicit
ing with verifiable rewards (RLVR) go beyond                              procedural constraints (Huang et al., 2025; Šuster
outcome-only supervision by enriching the reward                          et al., 2024). Likewise, analyses of LLM-based
signal with structural information. Masked-and-                           critical appraisal highlight dependence on model
reordered self-supervision provides auxiliary sig-                        pretraining and prompt sensitivity rather than veri-
nals encouraging coherent intermediate reason-                            fiable optimisation (Wang et al., 2025a; Lai et al.,
ing (Wang et al., 2025c), while self-verification                         2025). Across these methodologies, existing sys-
methods add progress-estimation or critique mod-                          tems employ prompting or supervised fine-tuning,
ules that guide models toward more reliable rea-                          but none leverage reinforcement learning for RoB
soning trajectories (Zeng et al., 2025). Theoretical                      assessment. Our work is, to our knowledge, the
analyses further show that verifiable rewards influ-                      first to introduce RL-based training in this domain.
ence trajectory selection in predictable ways, steer-
                                                                          F   Hyperparameters and APIs
ing models toward high-success modes under veri-
fiable criteria (Wen et al., 2025). These approaches                      We executed all the experiments either via API or
strengthen RLVR but remain fundamentally cen-                             on our own cluster. We used the paid-for OpenAI
tered on terminal-outcome verification.                                   API to access GPT-4. On the other hand, we hosted
   Complementary lines of research pursue pro-                            and trained the open-source models used in this
cess supervision, scoring CoT steps using neural                          paper on a distributed cluster.
judges (Lightman et al., 2024; Zelikman et al.,                             SFT is performed for 5 epochs with a batch size
2022; Zhang et al., 2025; Zou et al., 2025). While                        of 1 (due to the large size of the input data) us-
such methods provide dense feedback unavail-                              ing a learning rate of 5 × 10−5 and the AdamW
optimiser (Loshchilov and Hutter, 2017).
   For the RL setups, we adopt the GRPO (Shao
et al., 2024) and DAPO (Yu et al., 2025) algo-
rithms, training for 3 epochs with a learning rate
of 1 × 10−6 , per-device batch size 1, and 16 sam-
pled generations per batch. Both training protocols
leverage gradient accumulation with 8 accumula-
tion steps. All experiments are conducted using the
Open-R1 framework (Hugging Face, 2025) on 8
NVIDIA A100 GPUs, each equipped with 80GB
of memory. Models have been served for inference
with the vLLM framework (Kwon et al., 2023).

G       Scientific Artefacts and Licensing
In this work, we used the following scientific arte-
facts. LLaMa 3.1 is licensed under a commercial
license1 . GPT-4 is licensed under a commercial li-
cense2 . Qwen2.5 is licensed under the Apache 2.0
license3 . Granite 3.1 is licensed under the Apache
2.0 license4 . DeepSeek models are licensed under
the MIT license5 . Mining text and data from the
Cochrane library is permitted for non-commercial
research through the Wiley API.6 . The usage of the
listed artefacts is consistent with their licenses.




    1
     https://llama.meta.com/doc/overview
    2
     https://openai.com/policies/terms-of-use
   3
     https://qwenlm.github.io/blog/qwen3
   4
     https://www.ibm.com/architectures/
product-guides/granite-31
   5
     https://api-docs.deepseek.com/news/
news250120
   6
     https://www.cochranelibrary.com/help/access
       Figure 7: Example of forest plot containing three studies and their corresponding risk of bias map.




                                                      Hawkey 2015
        Text              Autologous Hematopoetic Stem Cell Transplantation for Refractory Crohn Disease: A
                          Randomised Clinical Trial

                          Importance Case reports and series suggest hematopoietic stem cell transplantation
                          (HSCT) may benefit some patients with Crohn disease.
                          ...
        Outcome           Clinical Remission
        Comparison        Stem Cell vs Placebo or control
        Bias type         A
        Bias name         Random sequence generation (selection bias)
        Bias definition   Bias arising from the randomization process
        Bias value        +
        Annotation        "Identify_randomization_report": "reported",
                          "Identify_randomization_report_rationale": "The methods section of the trial explicitly
                          states that participants were allocated to groups using a computer-generated random
                          sequence. The authors describe the randomisation process in sufficient detail, indicating
                          that a statistician who was not involved in participant recruitment generated the
                          sequence.",
                          "Classify_randomization_method": "random",
                          "Classify_randomization_method_rationale": "A computer-generated sequence (e.g.,
                          using a random number generator or statistical software) is a truly random method. It does
                          not rely on any deterministic or quasi-random technique such as alternation, date of birth,
                          or admission order, therefore it is classified as ’random’.",
                          "Assess_sequence_predictability": "unpredictable",
                          "Assess_sequence_predictability_rationale": "Because the sequence was generated by a
                          computer algorithm and concealed from the investigators enrolling participants (allocation
                          was performed via sealed, opaque envelopes prepared after the sequence was generated),
                          neither the investigators nor the participants could foresee the upcoming allocation. This
                          makes the sequence unpredictable.",
                          "Baseline_Imbalance": "none",
                          "Baseline_Imbalance_rationale": "Baseline tables show that demographic and clinical
                          characteristics (age, sex, disease severity, etc.) are comparable across groups with no
                          statistically significant differences, indicating that the randomisation produced balanced
                          groups and is unlikely to have introduced bias."

Table 8: The corresponding entry in C OCHRANE F OREST E XT for Hawkey 2015 in Figure 7. Rationales are not used
for RL training.
