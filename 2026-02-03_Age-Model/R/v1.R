# coral_age_engine.R
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(stats)
})

# ---------------------------
# Helpers
# ---------------------------

decimal_year_to_month <- function(t) {
  # t in decimal years
  frac <- t - floor(t)
  m <- (floor(frac * 12) %% 12) + 1
  m
}

rotate_month <- function(m, phi) {
  ((m - 1 + phi) %% 12) + 1
}

log_norm <- function(x, mu, sigma) {
  dnorm(x, mean = mu, sd = sigma, log = TRUE)
}

circ_month_dist <- function(m1, m2) {
  d <- abs(m1 - m2)
  pmin(d, 12 - d)
}

# ---------------------------
# Priors / proposal draws
# ---------------------------

draw_year_durations <- function(K, sigma_year) {
  d <- rnorm(K, mean = 1.0, sd = sigma_year)
  pmax(pmin(d, 1.20), 0.80)
}

draw_dirichlet <- function(alpha_vec) {
  # simple Dirichlet sampler
  x <- rgamma(length(alpha_vec), shape = alpha_vec, rate = 1)
  x / sum(x)
}

draw_within_year_warps <- function(K, Mseg, alpha) {
  # returns K x Mseg matrix of increments summing to 1
  mat <- matrix(NA_real_, nrow = K, ncol = Mseg)
  for (k in seq_len(K)) {
    mat[k, ] <- draw_dirichlet(rep(alpha, Mseg))
  }
  mat
}

# ---------------------------
# Build times within a transect using age2 segmentation
# ---------------------------

build_times_one_transect <- function(df_tr, d, delta, Mseg) {
  # df_tr must already be sorted oldest->young within transect
  # df_tr must contain year_id (0..K-1)
  N <- nrow(df_tr)
  K <- length(d)
  
  # band start offsets in relative time
  B <- numeric(K + 1)
  B[2:(K+1)] <- cumsum(d)
  
  knot_s <- seq(0, 1, length.out = Mseg + 1)
  
  t_rel <- numeric(N)
  for (k in 0:(K-1)) {
    idx <- which(df_tr$year_id == k)
    nk <- length(idx)
    if (nk == 0) next
    
    # normalized within-band position by order
    s <- seq(0, 1, length.out = nk + 1)[1:nk]  # [0,1)
    knot_u <- c(0, cumsum(delta[k+1, ]))       # 0..1
    u <- approx(knot_s, knot_u, xout = s, rule = 2)$y
    
    t_rel[idx] <- B[k+1] + u * d[k+1]
  }
  
  t_rel
}

# ---------------------------
# Likelihood terms
# ---------------------------

loglik_climatology <- function(y, t, mu_month, phi, sigma_y) {
  m <- decimal_year_to_month(t)
  m_shift <- rotate_month(m, phi)
  mu <- mu_month[m_shift]
  sum(log_norm(y, mu, sigma_y))
}

loglik_instrumental <- function(y, t, is_inst, clim_fn, sigma_inst) {
  if (is.null(clim_fn)) return(0)
  idx <- which(is_inst)
  if (length(idx) == 0) return(0)
  yhat <- clim_fn(t[idx])
  sum(log_norm(y[idx], yhat, sigma_inst))
}

loglik_band_starts <- function(t, age1, phi, warm_month, sigma_phase_months) {
  idx <- which(age1 == 1)
  if (length(idx) == 0) return(0)
  m <- rotate_month(decimal_year_to_month(t[idx]), phi)
  dmon <- circ_month_dist(m, warm_month)
  # penalty centered at 0 months distance
  sum(log_norm(dmon, 0, sigma_phase_months))
}

loglik_tiepoints <- function(df, t, tiepoints) {
  if (is.null(tiepoints) || nrow(tiepoints) == 0) return(0)
  # tiepoints columns: transect, s, t_star, sigma_t
  ll <- 0
  for (j in seq_len(nrow(tiepoints))) {
    tr <- tiepoints$transect[j]
    sj <- tiepoints$s[j]
    idx <- which(df$transect == tr & df$s == sj)
    if (length(idx) != 1) next
    ll <- ll + log_norm(t[idx], tiepoints$t_star[j], tiepoints$sigma_t[j])
  }
  ll
}

loglik_overlaps <- function(df, t, overlaps) {
  if (is.null(overlaps) || nrow(overlaps) == 0) return(0)
  # overlaps columns: transect_i, s_i, transect_j, s_j, sigma_dt
  ll <- 0
  for (j in seq_len(nrow(overlaps))) {
    ti <- overlaps$transect_i[j]; si <- overlaps$s_i[j]
    tj <- overlaps$transect_j[j]; sj <- overlaps$s_j[j]
    idx_i <- which(df$transect == ti & df$s == si)
    idx_j <- which(df$transect == tj & df$s == sj)
    if (length(idx_i) != 1 || length(idx_j) != 1) next
    ll <- ll + log_norm(t[idx_i] - t[idx_j], 0, overlaps$sigma_dt[j])
  }
  ll
}

loglik_anchor <- function(df, t, anchor) {
  if (is.null(anchor)) return(0)
  idx <- which(df$transect == anchor$transect & df$s == anchor$s)
  if (length(idx) != 1) return(0)
  log_norm(t[idx], anchor$t_anchor, anchor$sigma_anchor)
}

# ---------------------------
# Main sampler
# ---------------------------

run_coral_age_sampler <- function(samples,
                                  mu_month,             # numeric length 12
                                  clim_fn = NULL,       # function(t)->d18o_clim (optional)
                                  tiepoints = NULL,
                                  overlaps = NULL,
                                  anchor = NULL,
                                  warm_month = 8,
                                  phi_grid = 0:11,
                                  n_particles = 5000,
                                  sigma_year = 0.05,
                                  Mseg = 6,
                                  alpha = 20,
                                  sigma_y = 0.06,
                                  sigma_inst = 0.08,
                                  sigma_phase_months = 1.5) {
  
  # Ensure canonical sorting within transect: oldest->young
  df <- samples %>%
    mutate(year_id = age2 - min(age2)) %>%  # map age2 to 0..K-1 (within the whole dataset)
    arrange(transect, age2, s)
  
  # Determine K per dataset: assume age2 is global band index
  K <- max(df$year_id) + 1
  
  # We'll draw a single (d, delta) for each particle and use it for all transects
  # (If you want transect-specific warps, we can extend to draw per transect.)
  # Overlaps will then encourage registration across transects through likelihood terms.
  
  all_T <- list()
  all_logw <- numeric()
  all_phi <- integer()
  
  # Pre-split by transect for speed
  split_tr <- split(df, df$transect)
  
  for (phi in phi_grid) {
    for (pp in seq_len(n_particles)) {
      
      d <- draw_year_durations(K, sigma_year)
      delta <- draw_within_year_warps(K, Mseg, alpha)
      
      # Build relative times for each transect, then stack
      t_rel <- numeric(nrow(df))
      for (tr_name in names(split_tr)) {
        tr_df <- split_tr[[tr_name]]
        idx <- match(tr_df$s, df$s[df$transect == tr_name]) # not safe; we will use row indices
        # safer: use original row indices
        rows <- which(df$transect == tr_name)
        t_rel[rows] <- build_times_one_transect(df[rows, ], d, delta, Mseg)
      }
      
      # Shift to absolute time using anchor if provided (hard shift)
      t <- t_rel
      if (!is.null(anchor)) {
        idxA <- which(df$transect == anchor$transect & df$s == anchor$s)
        if (length(idxA) == 1) {
          t <- t_rel - t_rel[idxA] + anchor$t_anchor
        }
      }
      
      # Compute joint log-likelihood (sum of terms)
      ll <- 0
      ll <- ll + loglik_climatology(df$y, t, mu_month, phi, sigma_y)
      ll <- ll + loglik_instrumental(df$y, t, df$is_inst, clim_fn, sigma_inst)
      ll <- ll + loglik_band_starts(t, df$age1, phi, warm_month, sigma_phase_months)
      ll <- ll + loglik_tiepoints(df, t, tiepoints)
      ll <- ll + loglik_overlaps(df, t, overlaps)
      ll <- ll + loglik_anchor(df, t, anchor)  # soft anchor too (optional)
      
      all_T[[length(all_T) + 1]] <- t
      all_logw[length(all_logw) + 1] <- ll
      all_phi[length(all_phi) + 1] <- phi
    }
  }
  
  Tmat <- do.call(rbind, all_T)
  logw <- all_logw
  phi_vec <- all_phi
  
  # Normalize weights stably
  logw <- logw - max(logw)
  w <- exp(logw)
  w <- w / sum(w)
  
  # Posterior over phi
  post_phi <- sapply(phi_grid, function(ph) sum(w[phi_vec == ph]))
  post_phi <- post_phi / sum(post_phi)
  
  list(df = df, T = Tmat, w = w, phi = phi_vec, post_phi = post_phi, phi_grid = phi_grid)
}

# ---------------------------
# Posterior summaries
# ---------------------------

weighted_quantile <- function(x, w, probs = c(0.025, 0.5, 0.975)) {
  o <- order(x)
  x <- x[o]; w <- w[o]
  cw <- cumsum(w) / sum(w)
  sapply(probs, function(p) approx(cw, x, xout = p, rule = 2)$y)
}

summarize_ages <- function(fit) {
  df <- fit$df
  T <- fit$T
  w <- fit$w
  N <- nrow(df)
  
  qs <- t(sapply(seq_len(N), function(i) weighted_quantile(T[, i], w)))
  out <- df %>%
    mutate(t_lo = qs[,1], t_med = qs[,2], t_hi = qs[,3])
  out
}
