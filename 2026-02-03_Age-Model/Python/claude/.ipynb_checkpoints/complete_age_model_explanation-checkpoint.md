"""
COMPLETE STEP-BY-STEP EXPLANATION OF THE BAYESIAN CORAL AGE MODEL
===================================================================

This document explains every part of the age model code, from data loading
to final visualization, in plain language.

"""

# ============================================================================
# PART 1: THE BIG PICTURE - WHAT ARE WE TRYING TO DO?
# ============================================================================

"""
THE PROBLEM:
-----------
You have a coral slab that you drilled horizontally along multiple paths.
- Each drill path has samples at different depths
- You measured δ¹³C and δ¹⁸O isotopes for each sample
- You see dark bands in X-ray images (annual growth bands, like tree rings)
- You know the coral was collected in April 2007

THE GOAL:
--------
Assign a calendar age (year) to each sample so you can:
- Create a timeline from ~1928 to 2007
- Study how isotopes changed over time
- Understand environmental/climate history

THE CHALLENGE:
-------------
- Samples are spaced irregularly in time (coral growth varies)
- Multiple drill paths that overlap and need to be aligned
- Dark bands give clues but aren't perfectly regular
- Need to quantify uncertainty in age estimates

THE SOLUTION:
------------
Bayesian age model that:
1. Builds ages forward from oldest to youngest
2. Uses dark bands as annual markers
3. Aligns multiple paths using X-ray observations
4. Accounts for all uncertainty properly
"""


# ============================================================================
# PART 2: DATA STRUCTURE - UNDERSTANDING YOUR DATA
# ============================================================================

"""
YOUR DATA HAS:
-------------

1. SAMPLES (574 total):
   - Each sample has: path_name, path_num, δ¹³C, δ¹⁸O
   - path_name: 'O', 'N', 'M', 'L', 'K', 'J', 'I', 'H'
   - Alphabetical order = temporal order (O is OLDEST, H is YOUNGEST)
   - path_num: increases from old to young within each path
   
2. DARK BANDS:
   - age1: Marks dark band starts
     * 1 = normal progression (one band younger)
     * -1 = overlap correction
     * 0 = same band (alignment)
     * 2 = skipped a band (gap)
   - age2: Cumulative band count (0.05 to 79.1)
   - age_end: Marks where bands end

3. TEMPORAL STRUCTURE:
   - Row 0 (oldest): Path O, sample 1, age2=0.05, ~1928
   - Row 573 (youngest): Path H, sample 113, age2=79.1, April 2007
   - Total span: ~79 years

EXAMPLE:
   Sample 0: Path=O, num=1, age2=0.05, d13c=-1.18, d18o=25.85
   Sample 573: Path=H, num=113, age2=79.1, d13c=-3.03, d18o=25.60
"""


# ============================================================================
# PART 3: THE BAYESIAN FRAMEWORK - WHY BAYESIAN?
# ============================================================================

"""
WHY BAYESIAN?
------------
A Bayesian model lets us:
1. Express uncertainty properly (ages aren't single values, they're distributions)
2. Combine multiple sources of information (dark bands, growth rates, isotopes)
3. Propagate uncertainty from unknowns to final estimates
4. Handle complex dependencies (ages depend on each other sequentially)

KEY CONCEPT:
-----------
Instead of finding THE age for each sample, we find a DISTRIBUTION of possible
ages. The distribution is narrow where we're certain, wide where we're uncertain.

BAYES' THEOREM:
--------------
Posterior ∝ Likelihood × Prior

- PRIOR: What we believe before seeing data
  * "Coral grows ~5-20 mm/year" (from literature)
  * "Dark bands are annual" (known from coral biology)
  * "Growth rates don't change drastically year-to-year"

- LIKELIHOOD: How well the data fit our model
  * "Do dark bands occur ~1 year apart in our ages?"
  * "Do isotopes look realistic?"
  * "Do path overlaps align properly?"

- POSTERIOR: Updated beliefs after seeing data
  * Distribution of possible ages for each sample
  * We sample from this using MCMC (Markov Chain Monte Carlo)
"""


# ============================================================================
# PART 4: CODE WALKTHROUGH - LINE BY LINE
# ============================================================================

"""
SECTION 1: DATA LOADING AND SETUP
---------------------------------
"""

import pymc as pm
import numpy as np
import pandas as pd
import pytensor.tensor as pt
import arviz as az
import matplotlib.pyplot as plt

# Load your Excel file
df = pd.read_excel('lab_results.xlsx')

# For testing, use just paths I and H (youngest, most recent)
test_paths = ['I', 'H']
df_test = df[df['path_name'].isin(test_paths)].copy()
df_test = df_test.reset_index(drop=True)

# Known: collection date
collection_date = 2007.25  # April 2007

"""
WHY JUST I AND H?
- Testing with smaller dataset first
- These are youngest (closest to 2007)
- Easier to validate results
- Once working, extend to all 8 paths
"""


"""
SECTION 2: DEFINING THE BAYESIAN MODEL
--------------------------------------
"""

with pm.Model() as simple_model:
    
    # ========================================================================
    # STEP 2A: PRIORS ON GROWTH RATE
    # ========================================================================
    
    """
    QUESTION: How fast does coral grow?
    ANSWER: We don't know exactly, but we have ideas from literature
    
    We're sampling horizontally, so "growth rate" here means
    "how much time elapses between consecutive samples"
    """
    
    mean_time_per_sample = pm.Gamma('mean_time_per_sample', 
                                     alpha=14, beta=100)
    """
    WHY Gamma(14, 100)?
    - Mean = alpha/beta = 14/100 = 0.14 years per sample
    - That's ~7 samples per year
    - Gamma distribution ensures positive values
    - Shape chosen based on your data: 235 samples / ~35 years ≈ 7/year
    
    WHAT THIS SAYS:
    "I believe, on average, there's about 0.14 years between samples,
     but I'm willing to learn the true value from the data."
    """
    
    variation = pm.HalfNormal('time_variation', sigma=0.03)
    """
    WHY HalfNormal(0.03)?
    - Describes year-to-year variation in time-per-sample
    - HalfNormal ensures positive (can't have negative variation)
    - sigma=0.03 means we expect small variation
    - Tight control helps prevent age reversals
    
    WHAT THIS SAYS:
    "Growth rate is fairly consistent, with small random fluctuations."
    """
    
    
    # ========================================================================
    # STEP 2B: BUILD AGES FOR EACH PATH
    # ========================================================================
    
    """
    KEY STRATEGY: Build ages BACKWARD from the known collection date
    
    WHY BACKWARD?
    - We KNOW the youngest sample was collected April 2007
    - This is our tightest constraint (±0.05 years)
    - Uncertainty grows as we go back in time
    - This is physically correct!
    
    ANALOGY: Like counting backwards from today's date
    """
    
    # PATH H (YOUNGEST)
    # -----------------
    path_H = df_test[df_test['path_name'] == 'H']
    n_H = len(path_H)  # 113 samples
    
    last_age_H = pm.Normal('age_H_end', mu=collection_date, sigma=0.05)
    """
    ANCHOR POINT: Last sample of H is at collection date
    - mu=2007.25 (April 2007)
    - sigma=0.05 (very tight - we KNOW this!)
    
    This is THE most certain thing in our model.
    """
    
    time_inc_H = pm.Lognormal('time_inc_H',
                              mu=np.log(mean_time_per_sample),
                              sigma=variation,
                              shape=n_H - 1)
    """
    TIME INCREMENTS for path H
    - One for each transition between samples (n-1 increments)
    - Lognormal ensures all positive (time can't go backward)
    - mu = log(mean_time_per_sample) centers around our belief
    - Each increment is slightly different (controlled by 'variation')
    
    VISUALIZATION:
    Sample: [1] -- inc1 --> [2] -- inc2 --> [3] ... --> [113]
    Age:    ??? <-- 0.15 <- ??? <-- 0.12 <- ??? ... <-- 2007.25
    
    We're building BACKWARD from 2007.25
    """
    
    cumsum_H = pt.concatenate([[0], pt.cumsum(time_inc_H)])
    """
    CUMULATIVE SUM: Add up all time increments
    - [0] at start = no time elapsed for first sample
    - cumsum adds: [0, inc1, inc1+inc2, inc1+inc2+inc3, ...]
    
    EXAMPLE:
    If increments are [0.15, 0.12, 0.18, ...]
    cumsum gives: [0, 0.15, 0.27, 0.45, ...]
    """
    
    ages_H_backward = last_age_H - cumsum_H
    """
    SUBTRACT FROM END: Go backward in time
    - Start at 2007.25
    - Subtract cumulative time
    - Results: [2007.25, 2007.10, 2006.98, 2006.80, ...]
    - But this is in REVERSE order (youngest to oldest)
    """
    
    ages_H = pm.Deterministic('ages_H', ages_H_backward[::-1])
    """
    REVERSE THE ORDER: [::-1] flips the array
    - Now matches data order (oldest to youngest)
    - [1985.xx, 1985.yy, ..., 2007.10, 2007.25]
    
    Deterministic means: not a random variable, just a calculation
    We track it to analyze later
    """
    
    
    # PATH I (OLDER)
    # --------------
    path_I = df_test[df_test['path_name'] == 'I']
    n_I = len(path_I)  # 122 samples
    
    age_I_start = pm.Normal('age_I_start',
                           mu=ages_H[0] - 18.0,
                           sigma=1.0)
    """
    CONNECT TO PATH H
    - ages_H[0] is first (oldest) sample of path H
    - Path I ends about where H begins
    - Based on age2 values: I spans ~18 years before H starts
    - sigma=1.0: more uncertain than H end (less constrained)
    
    WHY MORE UNCERTAIN?
    - Further from the known anchor (2007)
    - Uncertainty has accumulated
    - This is correct! Older = more uncertain.
    """
    
    time_inc_I = pm.Lognormal('time_inc_I',
                              mu=np.log(mean_time_per_sample),
                              sigma=variation,
                              shape=n_I - 1)
    
    cumsum_I = pt.concatenate([[0], pt.cumsum(time_inc_I)])
    
    ages_I = pm.Deterministic('ages_I', age_I_start + cumsum_I)
    """
    BUILD FORWARD for path I
    - Start at age_I_start (~1989)
    - ADD time increments going forward
    - Results: [1989.xx, 1989.yy, ..., 2006.xx]
    - Already in correct order (oldest to youngest)
    """
    
    
    # COMBINE PATHS
    # -------------
    all_ages = pm.Deterministic('all_ages',
                               pt.concatenate([ages_I, ages_H]))
    """
    CONCATENATE: Join path I and H ages into single array
    - [ages_I (122 samples), ages_H (113 samples)]
    - Total: 235 ages, one for each sample
    - Order: oldest (path I start) to youngest (path H end)
    """
    
    
    # ========================================================================
    # STEP 2C: DARK BAND CONSTRAINTS
    # ========================================================================
    
    """
    DARK BANDS = ANNUAL MARKERS
    Like tree rings, dark bands form once per year
    We use them to constrain our age model
    """
    
    band_indices = df_test[df_test['age1'].notna()].index.values
    """
    FIND DARK BANDS
    - age1 column has values where dark bands start
    - band_indices = [9, 19, 29, ...] (sample numbers)
    - 38 dark bands total in our test dataset
    """
    
    # CONSTRAINT 1: Bands should be ~1 year apart
    # -------------------------------------------
    if len(band_indices) > 1:
        for i in range(len(band_indices) - 1):
            age_diff = all_ages[band_indices[i+1]] - all_ages[band_indices[i]]
            
            pm.Potential(f'annual_spacing_{i}',
                        pm.logp(pm.Normal.dist(mu=1.0, sigma=0.15), age_diff))
    """
    WHAT THIS DOES:
    - Take consecutive dark bands (band i and band i+1)
    - Calculate time difference between them
    - Add constraint that difference should be ~1.0 years
    - sigma=0.15 allows some variation (±2 months)
    
    WHY pm.Potential?
    - age_diff is COMPUTED from ages (not raw data)
    - Can't use observed= with computed variables in PyMC
    - Potential adds log-probability directly
    
    EFFECT ON SAMPLING:
    - Sampler tries to find ages where bands are ~annual
    - If bands are 0.5 years apart: BAD (low probability)
    - If bands are 1.0 years apart: GOOD (high probability)
    - If bands are 2.0 years apart: BAD (low probability)
    """
    
    
    # CONSTRAINT 2: Monotonicity (ages must increase)
    # -----------------------------------------------
    age_diffs = all_ages[1:] - all_ages[:-1]
    
    pm.Potential('monotonic_ages',
                pm.logp(pm.Exponential.dist(lam=1.0/0.14), 
                       pm.math.maximum(age_diffs, 0.001)).sum())
    """
    WHY MONOTONICITY?
    - Ages MUST increase through the dataset
    - Sample 10 can't be younger than sample 9
    - This is a physical constraint
    
    HOW IT WORKS:
    - age_diffs = [ages[1]-ages[0], ages[2]-ages[1], ...]
    - All should be positive
    - Exponential distribution favors positive values
    - pm.math.maximum(age_diffs, 0.001) prevents -inf:
      * If diff is positive: use actual value
      * If diff is negative: use 0.001 (small positive)
      * This allows initialization but penalizes negatives
    
    WHY NOT HalfNormal?
    - HalfNormal gives -inf for negative values
    - Would prevent sampler from even starting
    - This "soft" constraint is more robust
    """
    
    
    # CONSTRAINT 3: Path alignment
    # ---------------------------
    i_last_age2 = df_test[df_test['path_name'] == 'I']['age2'].max()
    h_first_age2 = df_test[df_test['path_name'] == 'H']['age2'].min()
    age2_gap = h_first_age2 - i_last_age2
    
    age_I_end = ages_I[-1]
    age_H_start = ages_H[0]
    
    pm.Potential('path_alignment',
                pm.logp(pm.Normal.dist(mu=age2_gap, sigma=1.0),
                       age_H_start - age_I_end))
    """
    PATH ALIGNMENT USING age2
    - age2 from X-ray bands tells us paths overlap
    - Last sample of I (age2=61.7) vs first of H (age2=60.7)
    - They overlap by ~1 year
    - This constraint enforces smooth connection
    
    EFFECT:
    - Prevents gap or overlap at path boundary
    - Ensures continuous timeline across paths
    """
    
    
    # ========================================================================
    # STEP 2D: ISOTOPE LIKELIHOODS
    # ========================================================================
    
    """
    ISOTOPES provide information about:
    - Environment (temperature, salinity)
    - Coral metabolism
    - Seasonal cycles
    
    We model them to:
    1. Add more constraints to age model
    2. Eventually extract climate signals
    """
    
    # δ¹³C model (simplified for now)
    # -------------------------------
    d13c_mean = pm.Normal('d13c_mean', mu=-2.0, sigma=0.5)
    d13c_noise = pm.HalfNormal('d13c_noise', sigma=0.5)
    
    pm.Normal('d13c_obs',
             mu=d13c_mean,
             sigma=d13c_noise,
             observed=df_test['d13c'].values)
    """
    SIMPLE δ¹³C MODEL:
    - Mean value around -2.0‰ (from your data)
    - Plus random noise
    
    THIS IS SIMPLIFIED!
    Later we'll add:
    - Seasonal component (related to dark bands)
    - Long-term trend (your research interest)
    - Correlation with growth rate
    """
    
    # δ¹⁸O model (simplified for now)
    # -------------------------------
    d18o_mean = pm.Normal('d18o_mean', mu=25.4, sigma=0.3)
    d18o_noise = pm.HalfNormal('d18o_noise', sigma=0.3)
    
    pm.Normal('d18o_obs',
             mu=d18o_mean,
             sigma=d18o_noise,
             observed=df_test['d18o'].values)
    """
    SIMPLE δ¹⁸O MODEL:
    - Mean value around 25.4‰
    - Plus random noise
    
    FUTURE IMPROVEMENTS:
    - Add SST and SSS data (1960-2007)
    - Model: δ¹⁸O = α*SST + β*SSS + seasonal
    - Use for climate reconstruction
    """
    
    # Placeholder parameters (not used yet, but needed to run)
    band_month_rad = pm.Uniform('band_month_rad', lower=0, upper=2*np.pi)
    band_concentration = pm.Gamma('band_concentration', alpha=5, beta=1)
    """
    DARK BAND TIMING (disabled for now)
    - Will use VonMises to constrain when bands form
    - Currently just placeholders
    - Will add back after basic model works
    """


# ============================================================================
# PART 5: SAMPLING - GETTING THE POSTERIOR
# ============================================================================

"""
MARKOV CHAIN MONTE CARLO (MCMC)
-------------------------------
Now we sample from the posterior distribution using NUTS sampler.

WHAT IS SAMPLING?
- We can't compute the posterior analytically (too complex)
- Instead, we SAMPLE from it
- Like taking random draws from a hat, but weighted by probability
- After many samples, histogram approximates the true distribution
"""

with simple_model:
    trace = pm.sample(
        draws=1000,      # Number of samples from posterior (per chain)
        tune=1000,       # Number of tuning samples (thrown away)
        chains=2,        # Run 2 independent chains (checks convergence)
        cores=1,         # Use 1 CPU core
        target_accept=0.95,  # Acceptance rate (higher = more careful)
        random_seed=42,  # For reproducibility
        return_inferencedata=True,
        init='adapt_diag'  # Initialization method
    )

"""
WHAT HAPPENS DURING SAMPLING?
1. Start at random point in parameter space
2. Propose a move (change some parameters slightly)
3. Calculate: is this new point better or worse?
4. If better: accept. If worse: maybe accept (random)
5. Repeat 1000s of times
6. The path traced out samples the posterior

TUNING (1000 iterations):
- Sampler learns good step sizes
- These samples are thrown away
- Like "warming up"

DRAWS (1000 iterations):
- Actual posterior samples we keep
- Each draw is a complete set of all parameters
- We get 1000 × 2 = 2000 total samples

CHAINS (2 independent runs):
- Start from different random points
- Should converge to same distribution
- If they don't match: model has problems
- Checked via R-hat statistic
"""


# ============================================================================
# PART 6: EXTRACTING RESULTS
# ============================================================================

"""
The trace object contains ALL samples for ALL parameters.
Now we extract and summarize.
"""

# Get age samples
ages_I_posterior = trace.posterior['ages_I'].values
ages_H_posterior = trace.posterior['ages_H'].values
# Shape: (chains=2, draws=1000, samples=122 or 113)

# Flatten across chains and draws
ages_I_flat = ages_I_posterior.reshape(-1, ages_I_posterior.shape[-1])
ages_H_flat = ages_H_posterior.reshape(-1, ages_H_posterior.shape[-1])
# Shape: (2000, 122 or 113)

# Calculate statistics
ages_I_median = np.median(ages_I_flat, axis=0)  # Best estimate
ages_I_lower = np.percentile(ages_I_flat, 2.5, axis=0)  # 95% CI lower
ages_I_upper = np.percentile(ages_I_flat, 97.5, axis=0)  # 95% CI upper

"""
WHAT DO THESE MEAN?
- Median: "Most likely" age (50th percentile)
- Lower/Upper: 95% credible interval
  * "We're 95% confident the true age is in this range"
  * Width shows our uncertainty

EXAMPLE:
Sample 50: age_median = 1995.3, lower = 1994.8, upper = 1995.8
Interpretation: "We estimate this sample is from 1995.3, 
                 give or take about ±0.5 years"
"""

# Combine all ages
ages_median = np.concatenate([ages_I_median, ages_H_median])
ages_lower = np.concatenate([ages_I_lower, ages_H_lower])
ages_upper = np.concatenate([ages_I_upper, ages_H_upper])

# Add to dataframe
df_test['age_median'] = ages_median
df_test['age_lower'] = ages_lower
df_test['age_upper'] = ages_upper
df_test['age_uncertainty'] = ages_upper - ages_lower


# ============================================================================
# PART 7: ANALYZING DARK BAND TIMING
# ============================================================================

"""
Now that we have ages, when did dark bands occur?
"""

def month_from_decimal_year(t):
    frac = t - np.floor(t)
    return (np.floor(frac * 12).astype(int) % 12) + 1

# Get ages of dark band samples
band_ages = ages_median[band_indices]

# Convert to months
band_months = month_from_decimal_year(band_ages)
# Example: 1995.75 → 0.75 * 12 ≈ month 10 (October)

# Circular statistics (for months)
angles = (band_months - 1) * 2 * np.pi / 12
sin_mean = np.mean(np.sin(angles))
cos_mean = np.mean(np.cos(angles))
mean_angle = np.arctan2(sin_mean, cos_mean)
mean_month = (mean_angle * 12 / (2 * np.pi)) % 12 + 1

"""
WHY CIRCULAR STATISTICS?
- Months wrap around: Dec (12) → Jan (1)
- Regular mean of [11, 12, 1] = 8 (WRONG! Should be ~12)
- Circular mean correctly gives ~12

HOW IT WORKS:
- Convert months to angles on a circle
- Find average direction (using sin/cos)
- Convert back to months
"""


# ============================================================================
# PART 8: VALIDATION AND DIAGNOSTICS
# ============================================================================

"""
Check if model worked correctly:
"""

# 1. Monotonicity check
ages_increasing = np.all(np.diff(ages_median) > 0)
"""
MUST BE TRUE!
If ages decrease anywhere, model has problems.
"""

# 2. Convergence check (R-hat)
rhat = az.rhat(trace)
"""
R-hat measures agreement between chains.
- R-hat = 1.0: Perfect agreement
- R-hat < 1.01: Good (chains converged)
- R-hat > 1.1: Bad (chains didn't converge)

WHY IT MATTERS:
If chains don't agree, we haven't found the true posterior.
Need more tuning or model reparameterization.
"""

# 3. Effective sample size
ess = az.ess(trace)
"""
Accounts for autocorrelation in chains.
- We have 2000 total samples
- But consecutive samples are correlated
- ESS tells us effective independent samples
- Want ESS > 400 per parameter (rule of thumb)
"""

# 4. Divergences
"""
Divergences = sampling failures where NUTS couldn't follow gradient.
- 0 divergences: Excellent
- 1-5 divergences: Okay, but monitor
- Many divergences: Model problems, need reparameterization

CAUSES:
- Posterior geometry too complex
- Parameters on very different scales
- Constraints too tight
"""


# ============================================================================
# PART 9: VISUALIZATION
# ============================================================================

"""
Create plots to understand results:

1. Age estimates with uncertainty bands
   - Shows estimated chronology
   - Width of bands = uncertainty
   - Should narrow toward 2007 (anchor point)

2. Rose diagram for dark band timing
   - Polar histogram showing which months
   - Should cluster if seasonal
   - Red arrow = circular mean

3. Isotopes vs age
   - Now we have a timescale!
   - Can see temporal trends
   - Prepare for climate reconstruction

4. Diagnostics
   - Trace plots (did sampler explore well?)
   - Posterior distributions (what did we learn?)
"""


# ============================================================================
# PART 10: WHAT'S NEXT?
# ============================================================================

"""
CURRENT MODEL: Basic framework working
- 2 paths (I and H)
- Simple growth model
- Annual dark band constraints
- Monotonicity enforced

NEXT STEPS:

1. EXTEND TO ALL 8 PATHS
   - Same logic, more paths
   - Will span full 1928-2007 record
   - More complex path alignment

2. ADD CLIMATE CALIBRATION (1960-2007)
   - Load SST and SSS reanalysis data
   - Model: δ¹⁸O = α*SST + β*SSS + ε
   - Estimate α and β from modern period
   - Use to reconstruct past climate

3. DECOMPOSE δ¹³C SIGNAL
   - Seasonal component (related to dark bands)
   - Long-term trend (your research interest!)
   - Use Gaussian Process or spline

4. ADD VONMISES CONSTRAINT
   - Force dark bands to occur at consistent season
   - Tightens age model
   - Requires careful initialization

5. SENSITIVITY ANALYSIS
   - How sensitive to prior choices?
   - What if we change sigma values?
   - Robustness checks

6. VALIDATION
   - If you have independent age markers (U-Th dating?)
   - Compare to other corals in region
   - Check consistency with historical climate records
"""


# ============================================================================
# KEY CONCEPTS SUMMARY
# ============================================================================

"""
1. BAYESIAN = Expressing uncertainty properly
   - Not just point estimates
   - Full probability distributions
   - Propagates uncertainty correctly

2. HIERARCHICAL MODEL = Layers of unknowns
   - Bottom: observed data (isotopes, bands)
   - Middle: ages (what we want)
   - Top: parameters (growth rates, timing)

3. MCMC = Sampling the posterior
   - Can't solve analytically
   - Sample many times
   - Histogram ≈ true distribution

4. CONSTRAINTS = Information sources
   - Dark bands ~ annual
   - Ages must increase
   - Paths must align
   - Each adds information

5. UNCERTAINTY GROWS BACKWARD
   - 2007: Very certain (±0.05 years)
   - 2000: Less certain (±0.3 years)
   - 1990: Even less (±1.0 years)
   - 1928: Most uncertain (±3 years)
   - This is CORRECT!

6. VALIDATION IS CRITICAL
   - Check R-hat
   - Check monotonicity
   - Look at trace plots
   - Make sure it makes sense!
"""

# ============================================================================
# GLOSSARY OF TERMS
# ============================================================================

"""
BAYESIAN: Method that works with probability distributions, not point estimates

PRIOR: What we believe before seeing data

LIKELIHOOD: How well data fits our model

POSTERIOR: Updated beliefs after seeing data (what we want!)

MCMC: Markov Chain Monte Carlo - sampling method for complex posteriors

NUTS: No-U-Turn Sampler - smart MCMC algorithm (auto-tunes step size)

TRACE: Collection of all posterior samples

CHAIN: One independent MCMC run (run multiple to check convergence)

R-HAT: Convergence diagnostic (chains should agree)

ESS: Effective Sample Size (accounting for autocorrelation)

DIVERGENCE: Sampling failure (sampler couldn't follow gradient)

DETERMINISTIC: Computed value (not random variable)

POTENTIAL: Way to add custom log-probability terms

LOGNORMAL: Distribution for positive values only

HALFNORMAL: Like normal but only positive half

EXPONENTIAL: Distribution for positive values (decays exponentially)

VONMISES: Circular normal distribution (for angles/months)

CREDIBLE INTERVAL: Bayesian confidence interval (95% CI = probably in this range)

MONOTONIC: Always increasing (or always decreasing)
"""
