## Overview

In previous steps, my colleagues built high‑performance models, such as CatBoost and XGBoost, that can predict how an audience will score a robot’s dance based on metrics like “Rhythm” and “Storytelling.” But there’s a problem: although knowing a dance will get, let’s say, a score of 4/5 is useful, it doesn’t tell a choreographer why. Our models are initially “black boxes.” The central goal of this work is to look inside that box, so we can identify the exact changes required for improving the choreography.

Hence, we used SHAP analysis. The core idea of SHAP is to take a single prediction for a single dance and explain exactly how each feature—like the number of movements in the performance—contributed to that final score.

We designed our SHAP analysis as a four‑step pipeline:

1. **Phase 1**: Extract the raw SHAP values and other foundational information.  
2. **Phase 2**: Generate plots to enhance interpretability of those SHAP values.  
3. **Phase 3**: Evaluate the robustness and validity of our approach.  
4. **Phase 4**: Aggregate and synthesize results into human‑interpretable outputs.

---

## Phase 1: Extracting Base Information

Phase 1 focuses on extracting the base information we need. The entire SHAP pipeline relies on this foundation. We take a sample of 500 instances from the test set and calculate the SHAP values for those samples.

We sample the test set because applying our method to the entire set would be too expensive in terms of computation time and resources. One might ask whether this random sampling is robust—this question is addressed later in Phase 3.

This phase produces three outputs:

- **background.csv**: Contains all the foundational information we need.  
- **raw_<target>.npy**: Derived from the background file and shows raw SHAP values for later steps.  
- **interaction values**: Used in multiple subsequent phases to capture feature interactions.

### SHAP Values Scatter Plot

![Scatter plot of SHAP values on the x‑axis (–1 to +1) vs. feature values, with a color bar indicating feature magnitude](https://raw.githubusercontent.com/rzsoli/Evaluations_on_Robotic_Choreographies/main/shap/readme/1.png)

On the x‑axis, we have the SHAP values: +1 indicates maximum positive effect on the target, –1 indicates maximum negative effect, and 0 means neutral influence.

Each point represents a sample from the test choreographies. The color bar reflects the change in the feature value—showing how the feature’s magnitude varies across samples.

For example, you can see that **nMovements** significantly influences the SHAP value. There’s a noticeable cluster of red points near –1, suggesting that an increase in nMovements generally has a negative effect on the target score.

Now, take a look at **movementsTransitionsDuration**. An average value in this feature tends to have a positive influence. However, very low average values show a slight negative effect, while very high values can have either a strong positive or negative influence, depending on the context.

---

## Phase 2: Feature Effects Analysis

Phase 2 is about answering **how** and **when** features matter. We create two sets of plots: SHAP dependence plots and PDP/ICE plots.

### SHAP Dependence Plots

![SHAP dependence plot for nMovements vs. SHAP value, colored by timeDuration](https://raw.githubusercontent.com/rzsoli/Evaluations_on_Robotic_Choreographies/main/shap/readme/2.png)

In this example, the y‑axis shows the SHAP value for **nMovements**, indicating its effect on the target score; the x‑axis shows raw nMovements values; and the color gradient represents **timeDuration**. You can see:

- Below ~16 movements, nMovements generally has a positive effect.  
- Above ~16 movements, it tends to have a negative effect.  
- Lower timeDuration (blue points) clusters around a SHAP value of zero, but as timeDuration increases, the influence deviates more strongly.

### PDP/ICE Plots

![PDP/ICE plot showing predicted score vs. timeDuration for individual samples](https://raw.githubusercontent.com/rzsoli/Evaluations_on_Robotic_Choreographies/main/shap/readme/3.png)

PDP (Partial Dependence) and ICE (Individual Conditional Expectation) plots show the average and individual effects of a single feature on predictions. For each choreography sample, we hold all other features constant and vary **timeDuration**:

As shown, setting timeDuration between **105** and **140** tends to yield the highest predicted score.

### Example: Music BPM & Movements Interaction

![SHAP dependence plot showing interaction between music BPM and number of movements](https://raw.githubusercontent.com/rzsoli/Evaluations_on_Robotic_Choreographies/main/shap/readme/4.png)

This plot shows the interaction between **music BPM** and **number of movements**:

- **Threshold 1 (movements)**: Below this, there’s a significant positive influence; above it, SHAP values drop and can turn negative.  
- **Threshold 2 (BPM)**: Below this, higher BPM has a negative influence; above it, the trend reverses, and higher BPM has a positive effect.

In real life, this mirrors the contrast between a prom dance (fewer movements, slower BPM) and an electronic dance (more movements, higher BPM). It raises questions about audience preference vs. robot performance capability—questions that our data-driven approach helps to illuminate.

---

## Phase 3: Robustness & Validity Analysis

### Bootstrap Stability Analysis

We assess whether feature importance rankings remain consistent across different data samples by performing a bootstrap analysis:

- Generate 50 bootstrap samples (each with 100 instances).  
- Recompute SHAP values and rank features by mean absolute SHAP value.  
- Compare rankings across bootstraps using Spearman’s ρ.  
- Features with average ρ > 0.8 are deemed stable.

This confirms that top features—like **nMovements**—are not statistical flukes but exhibit consistent importance.

   ![Waterfall plot showing baseline vs. feature contributions for one positive and one negative sample](https://raw.githubusercontent.com/rzsoli/Evaluations_on_Robotic_Choreographies/main/shap/readme/5.png)


### Subgroup SHAP Analysis

We examine whether feature importance varies across subgroups (e.g., folk vs. electronic music):

- For each feature and target (e.g., Rhythm, Storytelling), compute the mean absolute SHAP value within each subgroup.  
- Compare these values to understand context‑specific importance.

This ensures our explanations are tailored to different design contexts.

   ![Waterfall plot showing baseline vs. feature contributions for one positive and one negative sample](https://raw.githubusercontent.com/rzsoli/Evaluations_on_Robotic_Choreographies/main/shap/readme/6.png)

---

## Phase 4: Synthesis & Reporting

This final phase synthesizes the entire analysis into human-interpretable explanations for stakeholders; turning insights into actionable stories.
It acts as the delivery layer of our pipeline.
 
1. **Prototypical Decision Plots** (`decision_<target>.png`)  

These are detailed waterfall plots for individual predictions.
They visualize the baseline output and how each feature contributes
 to the final score for a selected choreography.
For each model and target, we include two representative samples: one 
with predominantly positive SHAP values, and one with negative values.

   ![Waterfall plot showing baseline vs. feature contributions for one positive and one negative sample](https://raw.githubusercontent.com/rzsoli/Evaluations_on_Robotic_Choreographies/main/shap/readme/7.png)


2. **Master Summary Table** (`final_summary.csv`)  

3. **Report Snippets** (`report_snippets.md`)  
   Auto‑generated plain‑English explanations drawn from the summary table.

All outputs are included in the project repository for easy access and review.

---

## Conclusion

- **Key Features**: We found that **nMovements**, **musicBPM**, and **timeDuration** are the most important features. Even when considering other features, their behavior often depends on interactions with these three.  
   ![Waterfall plot showing baseline vs. feature contributions for one positive and one negative sample](https://raw.githubusercontent.com/rzsoli/Evaluations_on_Robotic_Choreographies/main/shap/readme/8.png)


- **Non‑linear & Context‑dependent Effects**: For example, too many movements combined with a long duration can lead to worse outcomes.

- **Robustness**: Our results do not depend on sampling randomness, as validated by the bootstrap analysis.  
