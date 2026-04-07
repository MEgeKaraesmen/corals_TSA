# Bayesian Age Model Pseudo Code for Solomon Islands Coral
## Based on Actual Data Structure

```python
# ============================================================================
# DATA STRUCTURE UNDERSTANDING
# ============================================================================
# 
# - 574 samples across 8 paths (O, N, M, L, K, J, I, H)
# - Paths are in REVERSE alphabetical order (O is youngest, H is oldest)
# - path_num increases with decreasing age within each path (1 is youngest)
# - Collection date: April 2007 (2007.25)
#
# age1: Dark band start markers
#   - 1: Normal progression (one year younger than previous sample)
#   - -1: Band is older than previous sample (path overlap/correction)
#   - 0: Same band as previous (alignment across paths)
#   - 2: Skipped a ring (gap between non-overlapping paths)
#
# age2: Cumulative band count
#   - Simple running sum of age1
#   - Manual entries like row 81 (7.40) indicate interpolation (~70% to next band)
#   - Range: 0.05 (youngest) to 79.1 (oldest)
#   - Represents ~79 years of record (1920s to 2007)
#
# age_end: Marks end of dark bands (when detected)
#   - Numbers correspond to which band (from age2) is ending
#
# Isotope observations:
#   - d13C at band starts: mean=-1.253, std=0.606
#   - d13C non-band: mean=-1.304, std=0.495
#   - Minimal difference in d18O between band/non-band
#   - Confirms d13C is more correlated with bands than d18O


# ============================================================================
# BAYESIAN MODEL SPECIFICATION
# ============================================================================

import pymc as pm
import numpy as np
import pandas as pd

# Load data
df = pd.read_excel('lab_results.xlsx')
n_samples = len(df)
paths = df['path_name'].unique()  # ['O', 'N', 'M', 'L', 'K', 'J', 'I', 'H']
n_paths = len(paths)

# Known anchor: collection date
collection_date = 2007.25  # April 2007

# Estimated total span from age2 max
total_band_count = df['age2'].max()  # ~79 bands
estimated_oldest_age = collection_date - total_band_count  # ~1928


# ============================================================================
# LEVEL 1: AGE MODEL PARAMETERS
# ============================================================================

with pm.Model() as age_model:
    
    # --------------------------------------------------------------------
    # 1.1: Sample-level ages (what we're ultimately solving for)
    # --------------------------------------------------------------------
    # ages[i] = calendar year for sample i
    # We'll build this up path by path
    
    ages = pm.MutableData('ages', np.zeros(n_samples), dims='sample')
    
    
    # --------------------------------------------------------------------
    # 1.2: Growth/sampling rate parameters
    # --------------------------------------------------------------------
    # How much time elapses between consecutive samples?
    # This varies by path and potentially within paths
    
    # Prior on mean samples per year (across all paths)
    mean_samples_per_year = pm.Normal('mean_samples_per_year', mu=7, sigma=2)
    # Based on data: ~574 samples / 79 years ≈ 7.3 samples/year
    
    # Path-specific sampling rates (some paths might be more densely sampled)
    samples_per_year_by_path = pm.Normal('samples_per_year_by_path',
                                         mu=mean_samples_per_year,
                                         sigma=2,
                                         shape=n_paths)
    
    # Variability within a path (some periods might be more densely sampled)
    within_path_variation = pm.HalfNormal('within_path_variation', sigma=0.3)
    
    
    # --------------------------------------------------------------------
    # 1.3: Dark band seasonal timing parameters
    # --------------------------------------------------------------------
    # When in the year do dark bands form?
    
    dark_band_peak_month = pm.Uniform('dark_band_peak_month', lower=0, upper=12)
    # Month of year (0-12) when dark band is most prominent
    
    dark_band_duration_months = pm.Normal('dark_band_duration_months', mu=4, sigma=1)
    # How many months the dark band spans (constrained to be positive)
    dark_band_duration_months = pm.math.maximum(dark_band_duration_months, 1)
    
    
    # ============================================================================
    # LEVEL 2: BUILD AGE MODEL PATH BY PATH
    # ============================================================================
    
    # Strategy: Start from youngest path (O) at collection date, work backwards
    # Use age1 markers to align paths and establish chronology
    
    for path_idx, path_name in enumerate(paths):
        
        # Get samples for this path
        path_mask = df['path_name'] == path_name
        path_samples = df[path_mask]
        path_indices = path_samples.index.values
        n_path_samples = len(path_samples)
        
        # --------------------------------------------------------------
        # 2.1: First sample in path (youngest sample in this path)
        # --------------------------------------------------------------
        
        if path_name == 'O':  # Youngest path, anchor to collection date
            first_age = pm.Normal(f'age_{path_name}_start',
                                 mu=collection_date,
                                 sigma=0.05)  # Very tight: we know collection date
        
        else:
            # For other paths, first sample should align with previous path
            # via age2 values at path transition
            
            prev_path_idx = path_idx - 1
            prev_path_name = paths[prev_path_idx]
            
            # Get age2 at end of previous path and start of this path
            prev_path_end_age2 = df[df['path_name'] == prev_path_name]['age2'].max()
            this_path_start_age2 = path_samples['age2'].iloc[0]
            
            # These should represent the same calendar time
            # Use previous path's age model to infer start of this path
            first_age = pm.Normal(f'age_{path_name}_start',
                                 mu=get_age_at_age2(prev_path_name, this_path_start_age2),
                                 sigma=0.5)  # Some uncertainty in alignment
        
        
        # --------------------------------------------------------------
        # 2.2: Subsequent samples in path
        # --------------------------------------------------------------
        
        # Time increments between consecutive samples
        # These are NEGATIVE because path_num increases = age decreases
        
        time_per_sample = pm.Lognormal(f'time_per_sample_{path_name}',
                                       mu=np.log(1.0 / samples_per_year_by_path[path_idx]),
                                       sigma=within_path_variation,
                                       shape=n_path_samples-1)
        
        # Build cumulative age for this path
        # Start from first_age, subtract time increments
        path_ages = pm.Deterministic(f'ages_{path_name}',
                                     first_age - pm.math.cumsum(
                                         pm.math.concatenate([[0], time_per_sample])
                                     ))
        
        # Assign to global age array
        pm.Deterministic(f'assign_ages_{path_name}',
                        pm.math.set_subtensor(ages[path_indices], path_ages))
    
    
    # ============================================================================
    # LEVEL 3: DARK BAND CONSTRAINTS
    # ============================================================================
    
    # Strategy: Use age1 markers to enforce that dark band starts occur
    # at consistent times of year across all paths
    
    # Get all samples with dark band start markers
    band_start_mask = df['age1'].notna()
    band_start_indices = df[band_start_mask].index.values
    band_start_age1_values = df.loc[band_start_mask, 'age1'].values
    
    # --------------------------------------------------------------
    # 3.1: Dark bands should be roughly annual
    # --------------------------------------------------------------
    
    # For age1 = 1 (normal progression), consecutive bands should be ~1 year apart
    normal_progression_mask = (df['age1'] == 1.0).values[band_start_mask]
    
    for i in range(len(band_start_indices) - 1):
        if normal_progression_mask[i]:
            age_diff = ages[band_start_indices[i]] - ages[band_start_indices[i+1]]
            # Remember: ages decrease with increasing index
            
            pm.Normal(f'annual_band_spacing_{i}',
                     mu=age_diff,
                     sigma=0.15,  # Allow ±~2 months variation
                     observed=1.0)
    
    
    # --------------------------------------------------------------
    # 3.2: Dark bands should occur at consistent season
    # --------------------------------------------------------------
    
    # Extract fractional year for each dark band start
    for i, idx in enumerate(band_start_indices):
        age_frac = ages[idx] % 1.0  # Fractional part of year
        month_of_year = age_frac * 12
        
        # This should match the dark band peak month
        # Use circular/wrapped normal for months (to handle Dec-Jan wrapping)
        
        expected_month = dark_band_peak_month
        
        # Circular distance between observed and expected month
        month_diff = pm.math.minimum(
            pm.math.abs(month_of_year - expected_month),
            12 - pm.math.abs(month_of_year - expected_month)
        )
        
        pm.Normal(f'band_seasonal_timing_{i}',
                 mu=0,
                 sigma=1.5,  # Allow ±1.5 months variation
                 observed=month_diff)
    
    
    # --------------------------------------------------------------
    # 3.3: Handle special age1 cases
    # --------------------------------------------------------------
    
    # age1 = -1: Band is older than expected (overlap correction)
    # These indicate path overlap where band was already counted
    # No additional constraint needed - just a marker
    
    # age1 = 0: Same band as previous (alignment marker)
    # The samples with age1=0 should have similar ages to nearby age1=1
    # Already handled by path alignment in Level 2
    
    # age1 = 2: Skipped a ring (gap between paths)
    # Age difference should be ~2 years instead of 1
    skip_mask = (df['age1'] == 2.0).values
    # Handle in band spacing constraints above
    
    
    # ============================================================================
    # LEVEL 4: ISOTOPE LIKELIHOOD
    # ============================================================================
    
    # --------------------------------------------------------------
    # 4.1: δ13C correlation with dark bands
    # --------------------------------------------------------------
    
    # Model δ13C as having three components:
    # 1. Seasonal cycle (related to dark bands)
    # 2. Long-term trend (the signal of interest)
    # 3. Observation noise
    
    # Seasonal component: different mean in vs out of dark band
    d13c_in_band = pm.Normal('d13c_in_band_mean', mu=-1.25, sigma=0.3)
    d13c_out_band = pm.Normal('d13c_out_band_mean', mu=-1.30, sigma=0.3)
    d13c_seasonal_sd = pm.HalfNormal('d13c_seasonal_sd', sigma=0.3)
    
    # For each sample, determine if it's in dark band season
    d13c_seasonal_expected = []
    
    for i in range(n_samples):
        age_frac = ages[i] % 1.0
        month = age_frac * 12
        
        # Is this month within dark band season?
        in_band = is_in_season(month, dark_band_peak_month, dark_band_duration_months)
        
        expected = pm.math.switch(in_band, d13c_in_band, d13c_out_band)
        d13c_seasonal_expected.append(expected)
    
    d13c_seasonal = pm.math.stack(d13c_seasonal_expected)
    
    
    # Long-term trend: use Gaussian Process or spline
    # This captures the multi-decadal signal you're interested in
    
    # Option A: Gaussian Process (flexible but computationally expensive)
    # Lengthscale should be long (years to decades)
    gp_lengthscale = pm.InverseGamma('gp_lengthscale', alpha=2, beta=10)
    # Prior suggests ~5-10 year lengthscale
    
    gp_amplitude = pm.HalfNormal('gp_amplitude', sigma=1.0)
    
    # Define GP over ages
    cov_func = gp_amplitude**2 * pm.gp.cov.ExpQuad(1, ls=gp_lengthscale)
    gp = pm.gp.Latent(cov_func=cov_func)
    
    d13c_longterm = gp.prior('d13c_longterm', X=ages[:, None])
    
    
    # Option B: Spline (faster, but less flexible)
    # n_knots = 10  # ~8 years between knots for 79 year record
    # knot_ages = np.linspace(estimated_oldest_age, collection_date, n_knots)
    # knot_values = pm.Normal('d13c_knot_values', mu=0, sigma=1.0, shape=n_knots)
    # d13c_longterm = interpolate_spline(ages, knot_ages, knot_values)
    
    
    # Combined model
    d13c_expected = d13c_seasonal + d13c_longterm
    
    d13c_obs_noise = pm.HalfNormal('d13c_obs_noise', sigma=0.3)
    
    d13c_likelihood = pm.Normal('d13c_obs',
                                mu=d13c_expected,
                                sigma=d13c_obs_noise,
                                observed=df['d13c'].values)
    
    
    # --------------------------------------------------------------
    # 4.2: δ18O calibration to climate data (1960-2007)
    # --------------------------------------------------------------
    
    # Only use samples from calibration period
    calibration_mask = ages >= 1960
    
    # Load SST and SSS data (monthly reanalysis)
    # sst_data = load_sst_reanalysis()  # shape: (n_months, )
    # sss_data = load_sss_reanalysis()  # shape: (n_months, )
    
    # For each sample in calibration period, match to climate data
    # δ18O = α * SST + β * SSS + ε
    
    alpha_sst = pm.Normal('alpha_sst', mu=-0.2, sigma=0.1)  # Per ‰/°C
    beta_sss = pm.Normal('beta_sss', mu=0.3, sigma=0.1)     # Per ‰/psu
    d18o_intercept = pm.Normal('d18o_intercept', mu=25.6, sigma=0.5)
    
    # For calibration samples
    calibration_indices = np.where(calibration_mask)[0]
    
    for i in calibration_indices:
        # Get SST and SSS at this sample's age
        sample_age = ages[i]
        sst_value = get_sst_at_age(sample_age)
        sss_value = get_sss_at_age(sample_age)
        
        d18o_expected[i] = (d18o_intercept + 
                           alpha_sst * sst_value + 
                           beta_sss * sss_value)
    
    d18o_calib_noise = pm.HalfNormal('d18o_calib_noise', sigma=0.2)
    
    d18o_calib_likelihood = pm.Normal('d18o_calib_obs',
                                     mu=d18o_expected[calibration_indices],
                                     sigma=d18o_calib_noise,
                                     observed=df.loc[calibration_indices, 'd18o'].values)
    
    
    # --------------------------------------------------------------
    # 4.3: δ18O for pre-1960 period
    # --------------------------------------------------------------
    
    # Without direct climate data, constrain δ18O to be within plausible range
    # and show realistic seasonal cycling
    
    pre1960_mask = ages < 1960
    pre1960_indices = np.where(pre1960_mask)[0]
    
    # Seasonal cycle amplitude (should be similar to calibration period)
    d18o_seasonal_amplitude = pm.HalfNormal('d18o_seasonal_amplitude', sigma=0.3)
    d18o_seasonal_phase = pm.Uniform('d18o_seasonal_phase', lower=0, upper=12)
    
    for i in pre1960_indices:
        age_frac = ages[i] % 1.0
        month = age_frac * 12
        
        # Sinusoidal seasonal cycle
        seasonal_component = (d18o_seasonal_amplitude * 
                            pm.math.sin(2 * np.pi * (month - d18o_seasonal_phase) / 12))
        
        d18o_expected[i] = d18o_intercept + seasonal_component
    
    d18o_pre1960_noise = pm.HalfNormal('d18o_pre1960_noise', sigma=0.25)
    
    d18o_pre1960_likelihood = pm.Normal('d18o_pre1960_obs',
                                       mu=d18o_expected[pre1960_indices],
                                       sigma=d18o_pre1960_noise,
                                       observed=df.loc[pre1960_indices, 'd18o'].values)
    
    
    # ============================================================================
    # LEVEL 5: PATH ALIGNMENT CONSTRAINTS (from X-ray)
    # ============================================================================
    
    # Use age2 values to constrain alignment between paths
    # Samples with same age2 should have same calendar age
    
    # Group samples by age2 value
    age2_values = df['age2'].dropna().unique()
    
    for age2_val in age2_values:
        # Find all samples with this age2 value
        samples_at_age2 = df[df['age2'] == age2_val].index.values
        
        if len(samples_at_age2) > 1:
            # All these samples should have similar ages
            reference_age = ages[samples_at_age2[0]]
            
            for idx in samples_at_age2[1:]:
                pm.Normal(f'age2_alignment_{age2_val}_{idx}',
                         mu=reference_age,
                         sigma=0.3,  # Allow ±0.3 year alignment uncertainty
                         observed=ages[idx])
    
    
    # ============================================================================
    # HELPER FUNCTIONS
    # ============================================================================
    
    def is_in_season(month, season_center, season_duration):
        """
        Check if a month falls within a season
        Handles wrapping (e.g., Nov-Feb season)
        """
        season_start = (season_center - season_duration/2) % 12
        season_end = (season_center + season_duration/2) % 12
        
        if season_start < season_end:
            return (month >= season_start) & (month <= season_end)
        else:  # Season wraps around year
            return (month >= season_start) | (month <= season_end)
    
    
    def get_age_at_age2(path_name, age2_value):
        """
        For a given path and age2 value, return the estimated age
        Used for path alignment
        """
        path_data = df[df['path_name'] == path_name]
        
        # Find samples bracketing this age2 value
        # Interpolate between them
        # Return age estimate
        pass  # Implementation depends on how we store path ages
    
    
    def get_sst_at_age(age):
        """
        Get SST value from reanalysis at a given calendar age
        """
        # Convert age to month index in reanalysis
        # Return interpolated SST value
        pass
    
    
    def get_sss_at_age(age):
        """
        Get SSS value from reanalysis at a given calendar age
        """
        # Convert age to month index in reanalysis
        # Return interpolated SSS value
        pass


# ============================================================================
# INFERENCE
# ============================================================================

# Sample from posterior
with age_model:
    # Use NUTS sampler (good for continuous parameters)
    trace = pm.sample(
        draws=2000,
        tune=1000,
        cores=4,
        target_accept=0.95,  # High acceptance for complex model
        return_inferencedata=True
    )

# ============================================================================
# POSTERIOR ANALYSIS
# ============================================================================

# Extract age estimates with uncertainty
age_posterior = trace.posterior['ages']  # shape: (chains, draws, samples)

# Get median age and credible intervals for each sample
age_median = age_posterior.median(dim=['chain', 'draw'])
age_lower = age_posterior.quantile(0.025, dim=['chain', 'draw'])
age_upper = age_posterior.quantile(0.975, dim=['chain', 'draw'])

# Add to dataframe
df['age_median'] = age_median
df['age_lower_95'] = age_lower
df['age_upper_95'] = age_upper

# Extract long-term δ13C trend
d13c_trend_posterior = trace.posterior['d13c_longterm']
d13c_trend_median = d13c_trend_posterior.median(dim=['chain', 'draw'])

# This is your research signal!
df['d13c_longterm_trend'] = d13c_trend_median

# ============================================================================
# VALIDATION
# ============================================================================

# 1. Check that dark bands align seasonally
band_start_months = (age_median[band_start_indices] % 1.0) * 12
print(f"Dark band timing: {band_start_months.mean():.1f} ± {band_start_months.std():.1f} months")

# 2. Check path alignments
for transition in path_transitions:
    check_alignment_at_transition(trace, transition)

# 3. Compare δ18O to climate data in calibration period
calibration_correlation = correlate_d18o_with_sst_sss(
    df[calibration_mask],
    sst_data,
    sss_data
)
print(f"Calibration period correlation: {calibration_correlation:.3f}")

# 4. Check growth rate plausibility
growth_rates = estimate_growth_rates(age_median, df)
print(f"Mean growth rate: {growth_rates.mean():.1f} mm/year")

```

## Key Features of This Approach

1. **Hierarchical structure**: Ages → Dark band timing → Isotopes
2. **Path alignment**: Uses age2 markers and X-ray alignment
3. **Seasonal constraints**: Dark bands occur at consistent time of year
4. **Isotope decomposition**: Separates seasonal signal from long-term trend
5. **Climate calibration**: Uses 1960-2007 SST/SSS to calibrate δ18O
6. **Uncertainty quantification**: Full posterior distribution for all parameters
7. **Flexible trend**: GP or spline for long-term δ13C signal

## Next Steps for Implementation

1. Load and format the external climate data (SST, SSS)
2. Implement the helper functions for interpolation
3. Choose between GP and spline for long-term trend
4. Set appropriate priors based on literature values
5. Test on subset of data first (e.g., just paths O and N)
6. Gradually add complexity and validate at each step
