"""
Simplified Bayesian Age Model for Coral Data
Working implementation to test core concepts
"""

import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from scipy import stats

# Load data
df = pd.read_excel('/mnt/user-data/uploads/lab_results.xlsx')

print("="*80)
print("SIMPLIFIED BAYESIAN AGE MODEL - TESTING VERSION")
print("="*80)
print(f"Total samples: {len(df)}")
print(f"Paths: {sorted(df['path_name'].unique())}")
print("="*80)

# ============================================================================
# STEP 1: Start with just TWO paths to test the concept
# ============================================================================

# Focus on paths O and N (youngest, most recent)
test_paths = ['O', 'N']
df_test = df[df['path_name'].isin(test_paths)].copy()
df_test = df_test.reset_index(drop=True)

print(f"\nWorking with paths {test_paths}")
print(f"Number of samples: {len(df_test)}")
print(f"Number of dark bands: {df_test['age1'].notna().sum()}")

# Collection date
collection_date = 2007.25  # April 2007

# ============================================================================
# STEP 2: Simple Age Model
# ============================================================================

with pm.Model() as simple_model:
    
    # ------------------------------------------------------------------
    # Prior: Mean time per sample
    # ------------------------------------------------------------------
    # From data: paths O and N together span ~8.7 bands (years)
    # with 81 samples → ~10 samples per year
    
    mean_time_per_sample = pm.Gamma('mean_time_per_sample', 
                                     alpha=10, beta=100)
    # Mean ≈ 0.1 year per sample (10 samples/year)
    
    variation = pm.HalfNormal('time_variation', sigma=0.05)
    
    # ------------------------------------------------------------------
    # Build ages for each path
    # ------------------------------------------------------------------
    
    path_ages_dict = {}
    
    for path_name in test_paths:
        path_mask = df_test['path_name'] == path_name
        path_data = df_test[path_mask]
        n_samples = len(path_data)
        
        if path_name == 'O':  # Youngest path
            # Anchor to collection date
            first_age = collection_date - 0.05  # Slightly before collection
            
        else:  # Path N
            # Should connect to path O
            # Get age2 at transition: Path O ends at ~2.0, Path N starts at ~2.05
            # So path N should start ~2 years before collection
            first_age = pm.Normal('age_N_start', mu=collection_date - 2.0, sigma=0.2)
        
        # Time increments (negative because we go back in time)
        time_increments = pm.Lognormal(f'time_inc_{path_name}',
                                        mu=np.log(mean_time_per_sample),
                                        sigma=variation,
                                        shape=n_samples-1)
        
        # Cumulative ages (going backwards in time)
        ages = pm.Deterministic(f'ages_{path_name}',
                               first_age - pm.math.concatenate([[0], 
                                                                pm.math.cumsum(time_increments)]))
        
        path_ages_dict[path_name] = ages
    
    # Concatenate all ages
    all_ages = pm.math.concatenate([path_ages_dict['O'], path_ages_dict['N']])
    
    # ------------------------------------------------------------------
    # Dark band seasonal constraint
    # ------------------------------------------------------------------
    
    # When in the year do dark bands form?
    band_month = pm.Uniform('band_month', lower=0, upper=12)
    
    # Get samples with dark band markers
    band_indices = df_test[df_test['age1'].notna()].index.values
    
    if len(band_indices) > 0:
        # Extract fractional year
        band_ages = all_ages[band_indices]
        band_months_frac = band_ages % 1.0
        band_months = band_months_frac * 12
        
        # These should cluster around a particular month
        # Use wrapped normal for circular data
        # Can't use VonMises with computed values, so use Normal with penalty
        
        # For each band, compute circular distance to expected month
        for i, idx in enumerate(band_indices):
            month_i = band_months[i]
            
            # Circular distance: minimum of forward/backward distance
            dist_forward = pm.math.abs(month_i - band_month)
            dist_backward = 12 - dist_forward
            circular_dist = pm.math.minimum(dist_forward, dist_backward)
            
            # This distance should be small (near zero)
            pm.Normal(f'band_timing_{i}',
                     mu=0,
                     sigma=1.5,  # Allow ±1.5 months
                     observed=0,
                     testval=0)
            
            # Add potential (soft constraint) instead
            pm.Potential(f'band_season_{i}', 
                        -0.5 * (circular_dist / 1.5)**2)
    
    # ------------------------------------------------------------------
    # Dark bands should be roughly annual
    # ------------------------------------------------------------------
    
    if len(band_indices) > 1:
        for i in range(len(band_indices) - 1):
            age_diff = all_ages[band_indices[i]] - all_ages[band_indices[i+1]]
            
            # Use Potential instead of observed Normal
            # Add log-probability directly to the model
            pm.Potential(f'annual_spacing_{i}',
                        pm.logp(pm.Normal.dist(mu=1.0, sigma=0.15), age_diff))
    
    # ------------------------------------------------------------------
    # Isotope likelihood - simplified
    # ------------------------------------------------------------------
    
    # d13C: Just check that it matches observations with some noise
    d13c_noise = pm.HalfNormal('d13c_noise', sigma=0.5)
    
    # Simple model: d13C has a mean value plus noise
    d13c_mean = pm.Normal('d13c_mean', mu=-1.3, sigma=0.3)
    
    pm.Normal('d13c_obs',
             mu=d13c_mean,
             sigma=d13c_noise,
             observed=df_test['d13c'].values)
    
    # d18O: similarly simple for now
    d18o_noise = pm.HalfNormal('d18o_noise', sigma=0.3)
    d18o_mean = pm.Normal('d18o_mean', mu=25.65, sigma=0.2)
    
    pm.Normal('d18o_obs',
             mu=d18o_mean,
             sigma=d18o_noise,
             observed=df_test['d18o'].values)


# ============================================================================
# STEP 3: Sample from the model
# ============================================================================

print("\nStarting MCMC sampling...")
print("This may take a few minutes...\n")

with simple_model:
    # Start with prior predictive check
    prior_pred = pm.sample_prior_predictive(samples=500, random_seed=42)
    print("✓ Prior predictive sampling successful")
    
    # Sample from posterior
    trace = pm.sample(
        draws=1000,
        tune=500,
        cores=2,
        target_accept=0.9,
        random_seed=42,
        return_inferencedata=True
    )
    print("\n✓ Posterior sampling successful")

# ============================================================================
# STEP 4: Analyze results
# ============================================================================

print("\n" + "="*80)
print("RESULTS")
print("="*80)

# Extract ages
ages_O_posterior = trace.posterior['ages_O'].values  # shape: (chains, draws, samples)
ages_N_posterior = trace.posterior['ages_N'].values

# Combine
ages_O_flat = ages_O_posterior.reshape(-1, ages_O_posterior.shape[-1])
ages_N_flat = ages_N_posterior.reshape(-1, ages_N_posterior.shape[-1])

# Get median and credible intervals
ages_O_median = np.median(ages_O_flat, axis=0)
ages_O_lower = np.percentile(ages_O_flat, 2.5, axis=0)
ages_O_upper = np.percentile(ages_O_flat, 97.5, axis=0)

ages_N_median = np.median(ages_N_flat, axis=0)
ages_N_lower = np.percentile(ages_N_flat, 2.5, axis=0)
ages_N_upper = np.percentile(ages_N_flat, 97.5, axis=0)

# Combine into dataframe
ages_median = np.concatenate([ages_O_median, ages_N_median])
ages_lower = np.concatenate([ages_O_lower, ages_N_lower])
ages_upper = np.concatenate([ages_O_upper, ages_N_upper])

df_test['age_median'] = ages_median
df_test['age_lower'] = ages_lower
df_test['age_upper'] = ages_upper
df_test['age_uncertainty'] = ages_upper - ages_lower

print(f"\nAge estimates for paths {test_paths}:")
print(f"  Youngest sample: {ages_median.max():.2f} (±{df_test.iloc[ages_median.argmax()]['age_uncertainty']/2:.2f})")
print(f"  Oldest sample: {ages_median.min():.2f} (±{df_test.iloc[ages_median.argmin()]['age_uncertainty']/2:.2f})")
print(f"  Total span: {ages_median.max() - ages_median.min():.2f} years")
print(f"  Mean uncertainty: ±{df_test['age_uncertainty'].mean()/2:.2f} years")

# Dark band timing
band_month_posterior = trace.posterior['band_month'].values.flatten()
print(f"\nDark band timing:")
print(f"  Forms around month: {np.median(band_month_posterior):.1f} ± {np.std(band_month_posterior):.1f}")
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
month_idx = int(np.median(band_month_posterior))
if 0 <= month_idx < 12:
    print(f"  Approximately: {month_names[month_idx]}")

# Check convergence
print("\nConvergence diagnostics:")
print(f"  R-hat (should be < 1.01):")
rhat = az.rhat(trace)
for var in ['mean_time_per_sample', 'band_month']:
    if var in rhat.data_vars:
        print(f"    {var}: {float(rhat[var].values):.4f}")

# ============================================================================
# STEP 5: Visualize
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Age estimates with uncertainty
ax1 = axes[0, 0]
for path in test_paths:
    mask = df_test['path_name'] == path
    path_data = df_test[mask]
    
    ax1.fill_between(path_data.index, 
                     path_data['age_lower'], 
                     path_data['age_upper'],
                     alpha=0.3, label=f'Path {path} (95% CI)')
    ax1.plot(path_data.index, path_data['age_median'], 'o-', markersize=3)
    
    # Mark dark bands
    bands = path_data[path_data['age1'].notna()]
    ax1.scatter(bands.index, bands['age_median'], s=100, marker='|', 
               color='red', linewidths=2, zorder=10)

ax1.set_xlabel('Sample Index')
ax1.set_ylabel('Estimated Age (years CE)')
ax1.set_title('Age Model with Uncertainty')
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Isotopes vs estimated age
ax2 = axes[0, 1]
ax2.scatter(df_test['age_median'], df_test['d13c'], alpha=0.5, s=30, label='δ¹³C')
ax2.set_xlabel('Estimated Age (years CE)')
ax2.set_ylabel('δ¹³C (‰)')
ax2.set_title('δ¹³C vs Age')
ax2.grid(alpha=0.3)
ax2.invert_xaxis()

ax3 = axes[1, 0]
ax3.scatter(df_test['age_median'], df_test['d18o'], alpha=0.5, s=30, 
           color='orange', label='δ¹⁸O')
ax3.set_xlabel('Estimated Age (years CE)')
ax3.set_ylabel('δ¹⁸O (‰)')
ax3.set_title('δ¹⁸O vs Age')
ax3.grid(alpha=0.3)
ax3.invert_xaxis()

# Plot 3: Dark band seasonal timing
ax4 = axes[1, 1]
ax4.hist(band_month_posterior, bins=30, alpha=0.7, edgecolor='black')
ax4.axvline(np.median(band_month_posterior), color='red', 
           linestyle='--', linewidth=2, label='Median')
ax4.set_xlabel('Month of Year')
ax4.set_ylabel('Posterior Density')
ax4.set_title('Dark Band Seasonal Timing')
ax4.set_xticks(np.arange(0, 13, 1))
ax4.set_xticklabels([''] + month_names, rotation=45)
ax4.legend()
ax4.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/home/claude/simple_age_model_results.png', dpi=150, bbox_inches='tight')
print("\n✓ Results figure saved")

# ============================================================================
# STEP 6: Save results
# ============================================================================

# Save the age estimates
df_test.to_csv('/home/claude/age_estimates_simple.csv', index=False)
print("✓ Age estimates saved to CSV")

# Summary statistics
summary = az.summary(trace, var_names=['mean_time_per_sample', 'band_month'])
print("\nParameter estimates:")
print(summary)

print("\n" + "="*80)
print("SIMPLE MODEL COMPLETE")
print("="*80)
print("\nNext steps:")
print("1. Examine the results above")
print("2. Check if age estimates are reasonable")
print("3. Verify dark band timing makes sense")
print("4. If good, extend to more paths")
print("5. Add climate data constraints")
print("6. Implement long-term trend decomposition")
