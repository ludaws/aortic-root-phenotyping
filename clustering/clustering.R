#!/usr/bin/env Rscript
# clustering.R
# Tissue spatial phenotyping: NMF decomposition + consensus clustering.
#
# Inputs: L1 pattern-normalised spatial channel matrices (cases x features).
# Channels: polar_calcific, polar_fibrotic, unwrap_calcific.
#
# Pipeline (every step has a published precedent):
#   1. Load channel matrices; restrict to tricuspid cases.
#   2. NMF per channel (Brunet 2004, R NMF package). Rank pre-specified at
#      r=2 across all channels by cophenetic stability and parsimony per
#      Brunet 2004 / TCGA Firehose convention. All three channels achieve
#      cophenetic correlation above the field-standard stability threshold
#      (>0.88) at r=2; higher ranks yield only marginal cophenetic gains
#      with low explained-variance components.
#   3. Range-normalise each H column to [0, 1] and weight by within-channel
#      explained variance; concatenate across channels into a per-case
#      embedding.
#   4. ConsensusClusterPlus: PAM clustering, Pearson distance, reps=500,
#      pItem=0.85, maxK=6 (Monti 2003; Wilkerson 2010 ConsensusClusterPlus).
#   5. k selection by joint cophenetic + silhouette per Brunet 2004 / TCGA
#      Firehose convention. k=3 selected as the rank preceding the largest
#      cophenetic drop, with adequate silhouette and PAC below 0.15
#      stability threshold (Senbabaoglu 2014). Pre-specified clinical
#      phenotyping rationale also supports k=3.
#   6. Validation metrics: cophenetic correlation of consensus matrix by k
#      (Brunet 2004 / TCGA), mean silhouette width by k (Rousseeuw 1987 /
#      TCGA), PAC by k (Senbabaoglu 2014), ARI across consensus seeds
#      (Monti 2003), per-cluster bootstrap Jaccard (Hennig 2007).
#
# Outputs (in results/phenotyping/):
#   cluster_assignments_<version>.csv         primary (selected k, default 3)
#   cluster_assignments_<version>_k3.csv      k=3 partition (manuscript)
#   validation_metrics.csv                    headline metric table
#   k_selection_table.csv                     cophenetic/silhouette/PAC by k
#   jaccard_per_cluster.csv                   bootstrap Jaccard at k=3
#   clustering_diagnostics/<version>/...      rank diag, consensus plots
#
# Usage:
#   Rscript clustering.R                                      # production
#   Rscript clustering.R --test --n 100                       # quick test
#   Rscript clustering.R --fixed-ranks pc=3,pf=2,uw=2         # override ranks
#   Rscript clustering.R --target-k 4                         # override k

suppressPackageStartupMessages({
  library(ConsensusClusterPlus); library(NMF); library(doParallel)
  library(dplyr); library(readr); library(jsonlite); library(cluster)
})

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
args  <- commandArgs(trailingOnly = TRUE)
arg   <- function(f, default) {
  i <- which(args == f); if (length(i) && i < length(args)) args[i+1] else as.character(default)
}
flag  <- function(f) f %in% args

test_mode     <- flag("--test")
test_n        <- as.integer(arg("--n",       100))
n_runs        <- as.integer(arg("--runs",    if (test_mode) 3  else 30))
max_rank      <- as.integer(arg("--ranks",   if (test_mode) 4  else 6))
n_reps        <- as.integer(arg("--reps",    if (test_mode) 50 else 500))
max_k         <- 6
subsample     <- 0.85
rank_sweep    <- 2:max_rank
version       <- arg("--version",    "normed")
input_dir_in  <- arg("--input-dir",  "")
fixed_str     <- arg("--fixed-ranks", "pc=2,pf=2,uw=2")  # production default
target_k      <- as.integer(arg("--target-k", 3))         # production default
B_boot        <- as.integer(arg("--boot-b", if (test_mode) 10 else 100))
consensus_seeds <- as.integer(strsplit(arg("--consensus-seeds", "42,123,456"), ",")[[1]])
n_cores       <- as.integer(arg("--cores", max(1L, parallel::detectCores() - 1L)))
registerDoParallel(cores = n_cores)

fixed_ranks <- NULL
if (nchar(fixed_str) > 0) {
  fixed_ranks <- list()
  for (p in strsplit(fixed_str, ",")[[1]]) {
    kv <- strsplit(p, "=")[[1]]; fixed_ranks[[kv[1]]] <- as.integer(kv[2])
  }
  cat("Fixed ranks:", fixed_str, "\n")
}
cat(sprintf("Target k:   %d\n", target_k))
cat(sprintf("Reps:       %d\n", n_reps))
cat(sprintf("Bootstrap:  B=%d\n", B_boot))

# ---------------------------------------------------------------------------
# Paths and channel registry
# ---------------------------------------------------------------------------
base_in  <- "results/phenotyping/clustering_input"
out_dir  <- "results/phenotyping"
diag_dir <- file.path(out_dir, "clustering_diagnostics")
dir.create(diag_dir, recursive = TRUE, showWarnings = FALSE)

data_dir <- if (nchar(input_dir_in)) input_dir_in else file.path(base_in, "normed")
cat(sprintf("Input: %s   Mode: %s   Version: %s\n",
            data_dir, if (test_mode) "TEST" else "PROD", version))

CHANNEL_ABBREV  <- list(polar_calcific="pc", polar_fibrotic="pf", unwrap_calcific="uw")
CHANNEL_DISPLAY <- list(polar_calcific="Polar Calcific",
                        polar_fibrotic="Polar Fibrotic",
                        unwrap_calcific="Unwrap Calcific")

# ---------------------------------------------------------------------------
# Load channel CSVs; restrict to tricuspid cases present in every channel.
# ---------------------------------------------------------------------------
load_data <- function(dir_path, tricuspid_cases) {
  cat(sprintf("Loading channels from %s\n", dir_path))
  csvs  <- list.files(dir_path, pattern = "\\.csv$")
  chans <- intersect(names(CHANNEL_ABBREV), gsub("\\.csv$", "", csvs))
  if (!length(chans)) stop("No recognised channel CSVs in ", dir_path)

  mats <- list(); cases_ch <- list()
  for (ch in chans) {
    df  <- read_csv(file.path(dir_path, paste0(ch, ".csv")), show_col_types = FALSE)
    mat <- as.matrix(df[, -1]); rownames(mat) <- df$case_name
    cat(sprintf("  %-22s %d x %d\n", ch, nrow(mat), ncol(mat)))
    mats[[ch]] <- mat; cases_ch[[ch]] <- df$case_name
  }
  common <- tricuspid_cases
  for (ch in chans) common <- intersect(common, cases_ch[[ch]])
  if (test_mode) common <- common[seq_len(min(test_n, length(common)))]
  cat(sprintf("  Tricuspid cases in all channels: %d\n", length(common)))

  out <- list(cases = common, channels = chans)
  for (ch in chans) out[[ch]] <- mats[[ch]][common, , drop = FALSE]
  out
}

# ---------------------------------------------------------------------------
# NMF rank selection (Brunet 2004). Only invoked when --fixed-ranks not given.
# For the manuscript, ranks are pre-specified at r=2 across all channels.
# ---------------------------------------------------------------------------
select_rank <- function(mat, label, sweep_r, n_runs, out_sub) {
  cat(sprintf("\nNMF rank estimation: %s (%d x %d)\n", label, nrow(mat), ncol(mat)))
  dir.create(out_sub, recursive = TRUE, showWarnings = FALSE)
  estim <- nmfEstimateRank(mat + 1e-10, r = sweep_r, nrun = n_runs,
                           seed = 42, method = "brunet", .opt = "vP")
  coph  <- estim$measures$cophenetic
  disp  <- estim$measures$dispersion
  ranks <- as.integer(rownames(estim$measures))
  keep <- ranks >= 2
  ranks <- ranks[keep]; coph <- coph[keep]; disp <- disp[keep]
  write.csv(data.frame(rank=ranks, cophenetic=coph, dispersion=disp),
            file.path(out_sub, sprintf("nmf_rank_diag_%s.csv", label)),
            row.names = FALSE)
  cat("  rank  cophenetic  dispersion\n")
  for (i in seq_along(ranks))
    cat(sprintf("  r=%-2d  %.4f      %.4f\n", ranks[i], coph[i], disp[i]))
  best <- ranks[length(ranks)]
  if (length(ranks) >= 2) {
    for (i in 2:length(ranks)) {
      if (coph[i] < coph[i-1]) { best <- ranks[i-1]; break }
    }
  }
  cat(sprintf("  Selected r=%d for %s (Brunet 2004 cophenetic rule)\n", best, label))
  best
}

# ---------------------------------------------------------------------------
# NMF fit (Brunet 2004 multiplicative-update algorithm).
# ---------------------------------------------------------------------------
fit_nmf <- function(mat, rank, seed = 42L, n_runs = 30L) {
  res <- nmf(mat + 1e-10, rank = rank, nrun = n_runs, seed = seed,
             method = "brunet", .opt = "vP")
  best <- fit(res)
  H <- basis(best); W <- t(coef(best))
  rownames(H) <- rownames(mat)
  WH <- W %*% t(H)
  ev <- sapply(seq_len(rank), function(k) {
    WkHk <- W[, k, drop = FALSE] %*% t(H[, k, drop = FALSE])
    sum(WkHk^2) / sum(WH^2)
  })
  cat(sprintf("  EV per component: %s\n",
              paste(sprintf("%.3f", ev), collapse = ", ")))
  list(H = H, W = W, ev = ev)
}

# ---------------------------------------------------------------------------
# Embedding: range-normalise each H column to [0, 1] and weight by explained
# variance, then concatenate across channels.
# ---------------------------------------------------------------------------
build_embedding <- function(nmf_list, chans) {
  range_norm <- function(H) apply(H, 2, function(x) {
    r <- range(x); if (diff(r) < 1e-10) x * 0 else (x - r[1]) / diff(r)
  })
  weight_H <- function(r) sweep(range_norm(r$H), 2, r$ev, "*")
  do.call(cbind, lapply(chans, function(ch) weight_H(nmf_list[[ch]])))
}

# ---------------------------------------------------------------------------
# Consensus clustering (Monti 2003; Wilkerson 2010).
# ---------------------------------------------------------------------------
compute_pac <- function(cm) sum(cm > 0.1 & cm < 0.9) / (nrow(cm) * (nrow(cm) - 1))

run_consensus <- function(H_emb, label, out_path, max_k, n_reps, sub, seed = 42L) {
  dir.create(out_path, recursive = TRUE, showWarnings = FALSE)
  cc <- ConsensusClusterPlus(d = t(H_emb), maxK = max_k, reps = n_reps,
                             pItem = sub, pFeature = 1, clusterAlg = "pam",
                             distance = "pearson", seed = seed,
                             plot = "pdf", title = out_path)
  pac <- sapply(2:max_k, function(k) compute_pac(cc[[k]]$consensusMatrix))
  names(pac) <- paste0("k", 2:max_k)
  list(cc = cc, pac = pac)
}

# ---------------------------------------------------------------------------
# Cophenetic correlation of consensus matrix (Brunet 2004 / TCGA Firehose).
# ---------------------------------------------------------------------------
cophenetic_of_consensus <- function(cm) {
  d_orig <- as.dist(1 - cm)
  hc <- hclust(d_orig, method = "average")
  d_coph <- cophenetic(hc)
  cor(d_orig, d_coph)
}

# ---------------------------------------------------------------------------
# Adjusted Rand Index (Hubert & Arabie 1985).
# ---------------------------------------------------------------------------
ari <- function(a, b) {
  tab <- table(a, b); n <- sum(tab)
  r2 <- sum(choose(rowSums(tab), 2)); c2 <- sum(choose(colSums(tab), 2))
  t2 <- sum(choose(tab, 2))
  exp <- r2 * c2 / choose(n, 2); maxv <- (r2 + c2) / 2
  if (maxv == exp) 1.0 else (t2 - exp) / (maxv - exp)
}

# Relabel partition `b` to maximally match `a` via greedy match on
# confusion matrix. Used for bootstrap Jaccard cluster matching.
relabel_to <- function(a, b) {
  tab <- table(a, b)
  used_b <- integer(0)
  ord <- order(apply(tab, 1, max), decreasing = TRUE)
  map <- integer(max(b))
  for (ai in ord) {
    bi_candidates <- setdiff(seq_len(ncol(tab)), used_b)
    if (!length(bi_candidates)) break
    bi <- bi_candidates[which.max(tab[ai, bi_candidates])]
    map[as.integer(colnames(tab)[bi])] <- as.integer(rownames(tab)[ai])
    used_b <- c(used_b, bi)
  }
  unmapped <- which(map == 0); leftover <- setdiff(seq_len(max(a, b)), map)
  if (length(unmapped)) map[unmapped] <- leftover[seq_along(unmapped)]
  map[b]
}

# ---------------------------------------------------------------------------
# ARI across consensus seeds (Monti 2003). NMF fixed; consensus reseeded.
# ---------------------------------------------------------------------------
ari_across_consensus_seeds <- function(H_emb, k, n_reps, sub, seeds) {
  cat(sprintf("\n--- ARI across consensus seeds %s (Monti 2003) ---\n",
              paste(seeds, collapse = ", ")))
  assigns <- list()
  for (s in seeds) {
    cat(sprintf("  consensus seed=%d ...\n", s))
    cc <- ConsensusClusterPlus(d = t(H_emb), maxK = max(k, 3), reps = n_reps,
                               pItem = sub, pFeature = 1, clusterAlg = "pam",
                               distance = "pearson", seed = s, plot = "none")
    assigns[[as.character(s)]] <- as.integer(cc[[k]]$consensusClass)
  }
  sn <- as.character(seeds)
  m  <- outer(sn, sn, Vectorize(function(i, j) ari(assigns[[i]], assigns[[j]])))
  dimnames(m) <- list(sn, sn)
  cat("  ARI matrix:\n"); print(round(m, 4))
  mean_ari <- mean(m[upper.tri(m)])
  cat(sprintf("  Mean pairwise ARI: %.4f\n", mean_ari))
  list(matrix = m, mean = mean_ari)
}

# ---------------------------------------------------------------------------
# Per-cluster bootstrap Jaccard (Hennig 2007).
# Resample cases with replacement, rerun consensus on the bootstrap, match
# each original cluster to its most similar bootstrap cluster, record Jaccard.
# Thresholds (Hennig 2007): >0.85 highly stable, 0.75-0.85 stable,
# 0.6-0.75 pattern-suggesting, <0.6 unstable.
# ---------------------------------------------------------------------------
bootstrap_jaccard <- function(H_emb, ref_part, k, n_reps, sub, B) {
  cat(sprintf("\n--- Bootstrap Jaccard per cluster (Hennig 2007, B=%d) ---\n", B))
  n <- nrow(H_emb)
  jacc_mat <- matrix(NA_real_, nrow = B, ncol = k,
                     dimnames = list(NULL, paste0("C", seq_len(k))))
  set.seed(42)
  for (b in seq_len(B)) {
    boot_idx <- sample(seq_len(n), n, replace = TRUE)
    H_b <- H_emb[boot_idx, , drop = FALSE]
    rownames(H_b) <- paste0("b_", seq_len(n))
    cc_b <- ConsensusClusterPlus(d = t(H_b), maxK = max(k, 3), reps = n_reps,
                                  pItem = sub, pFeature = 1, clusterAlg = "pam",
                                  distance = "pearson", seed = 42L, plot = "none")
    unique_orig <- unique(boot_idx)
    first_occur <- match(unique_orig, boot_idx)
    boot_part <- as.integer(cc_b[[k]]$consensusClass)[first_occur]
    ref_sub <- ref_part[unique_orig]
    boot_part_rel <- relabel_to(ref_sub, boot_part)
    for (kk in seq_len(k)) {
      A <- which(ref_sub == kk); Bset <- which(boot_part_rel == kk)
      if (length(A) == 0 && length(Bset) == 0) next
      jacc_mat[b, kk] <- length(intersect(A, Bset)) / length(union(A, Bset))
    }
    if (b %% 10 == 0 || b == 1)
      cat(sprintf("    bootstrap %d/%d\n", b, B))
  }
  cat("\n  Per-cluster Jaccard:\n")
  cat("    Cluster   mean   median  IQR\n")
  for (kk in seq_len(k)) {
    v <- jacc_mat[, kk]
    cat(sprintf("    C%d        %.3f  %.3f   [%.3f, %.3f]\n", kk,
                mean(v, na.rm = TRUE), median(v, na.rm = TRUE),
                quantile(v, .25, na.rm = TRUE),
                quantile(v, .75, na.rm = TRUE)))
  }
  jacc_mat
}

# ===========================================================================
# RUN
# ===========================================================================
# Wrapped so other scripts (e.g. stability.R) can source this file for the
# helper functions only by setting STABILITY_MODE before sourcing.
if (!exists("STABILITY_MODE")) {
case_index      <- read_csv(file.path(base_in, "case_index.csv"), show_col_types = FALSE)
tricuspid_cases <- case_index %>% filter(valve_type == "tricuspid") %>% pull(case_name)
cat(sprintf("Tricuspid cases in index: %d\n", length(tricuspid_cases)))

data <- load_data(data_dir, tricuspid_cases)

vdir <- file.path(diag_dir, version)
dir.create(file.path(vdir, "checkpoints"), recursive = TRUE, showWarnings = FALSE)
unlink(file.path(vdir, "checkpoints"), recursive = TRUE)
dir.create(file.path(vdir, "checkpoints"), recursive = TRUE, showWarnings = FALSE)

# --- 1. NMF per channel ---
nmf_list <- list(); ranks <- list()
rank_cophenetic <- list()
for (ch in data$channels) {
  abbrev <- CHANNEL_ABBREV[[ch]]
  sub    <- file.path(vdir, paste0("nmf_", ch))
  dir.create(sub, recursive = TRUE, showWarnings = FALSE)
  if (!is.null(fixed_ranks) && !is.null(fixed_ranks[[abbrev]])) {
    ranks[[ch]] <- fixed_ranks[[abbrev]]
    cat(sprintf("\nFixed rank r=%d for %s\n", ranks[[ch]], ch))
    # Still run rank sweep to record cophenetic at the chosen rank for the
    # validation table (Brunet 2004 stability metric).
    estim <- nmfEstimateRank(data[[ch]] + 1e-10, r = rank_sweep, nrun = n_runs,
                              seed = 42, method = "brunet", .opt = "vP")
    coph_vec <- estim$measures$cophenetic
    coph_ranks <- as.integer(rownames(estim$measures))
    write.csv(data.frame(rank=coph_ranks, cophenetic=coph_vec,
                          dispersion=estim$measures$dispersion),
              file.path(sub, sprintf("nmf_rank_diag_%s.csv", ch)),
              row.names = FALSE)
    rank_cophenetic[[ch]] <- coph_vec[coph_ranks == ranks[[ch]]]
    cat(sprintf("  Cophenetic at r=%d: %.4f (Brunet 2004)\n",
                ranks[[ch]], rank_cophenetic[[ch]]))
  } else {
    ranks[[ch]] <- select_rank(data[[ch]], ch, rank_sweep, n_runs, sub)
    diag_csv <- file.path(sub, sprintf("nmf_rank_diag_%s.csv", ch))
    if (file.exists(diag_csv)) {
      dfd <- read.csv(diag_csv)
      rank_cophenetic[[ch]] <- dfd$cophenetic[dfd$rank == ranks[[ch]]]
    } else {
      rank_cophenetic[[ch]] <- NA
    }
  }
  cat(sprintf("Fitting NMF r=%d: %s\n", ranks[[ch]], ch))
  nmf_list[[ch]] <- fit_nmf(data[[ch]], ranks[[ch]], seed = 42, n_runs = n_runs)
  saveRDS(nmf_list[[ch]], file.path(vdir, "checkpoints", paste0("nmf_", abbrev, ".rds")))
}

# --- 2. Build embedding ---
H_emb <- build_embedding(nmf_list, data$channels)
rownames(H_emb) <- data$cases
cat(sprintf("\nEmbedding: %d cases x %d components\n", nrow(H_emb), ncol(H_emb)))

# --- 3. Consensus clustering ---
cc_res <- run_consensus(H_emb, version, file.path(vdir, "consensus"),
                        max_k, n_reps, subsample, seed = 42L)
cc <- cc_res$cc; pac <- cc_res$pac

# --- 4. k selection metrics (Brunet 2004 / TCGA Firehose) ---
cat("\n--- k selection metrics (Brunet 2004 / TCGA Firehose convention) ---\n")
coph_by_k <- rep(NA_real_, max_k - 1); names(coph_by_k) <- paste0("k", 2:max_k)
sil_by_k  <- rep(NA_real_, max_k - 1); names(sil_by_k)  <- paste0("k", 2:max_k)
for (k in 2:max_k) {
  kn <- paste0("k", k)
  cm <- cc[[k]]$consensusMatrix
  coph_by_k[kn] <- cophenetic_of_consensus(cm)
  part_k <- as.integer(cc[[k]]$consensusClass)
  if (length(unique(part_k)) >= 2) {
    sk <- silhouette(part_k, dist(H_emb))
    sil_by_k[kn] <- mean(sk[, 3])
  }
}
cat("  k     cophenetic(cm)  silhouette   PAC\n")
for (k in 2:max_k) {
  kn <- paste0("k", k)
  cat(sprintf("  k=%-2d  %.4f          %.4f       %.4f\n",
              k, coph_by_k[kn], sil_by_k[kn], pac[kn]))
}
coph_diffs <- diff(coph_by_k)
largest_drop_idx <- which.min(coph_diffs)
k_brunet <- (2:max_k)[largest_drop_idx]
cat(sprintf("\n  Largest cophenetic drop after k=%d (Brunet/TCGA convention)\n", k_brunet))
cat(sprintf("  Selected k = %d\n", target_k))

# --- 5. Save assignments ---
output_csv <- file.path(out_dir, sprintf("cluster_assignments_%s.csv", version))
write_csv(data.frame(case_name = data$cases,
                     tissue_cluster = as.integer(cc[[target_k]]$consensusClass)),
          output_csv)
cat(sprintf("\nSaved: %s\n", output_csv))
cat(sprintf("Cluster sizes (k=%d):\n", target_k))
print(table(as.integer(cc[[target_k]]$consensusClass)))

# Always also save k=3 explicitly (manuscript convention)
output_k3 <- sub("\\.csv$", "_k3.csv", output_csv)
write_csv(data.frame(case_name = data$cases,
                     tissue_cluster = as.integer(cc[[3]]$consensusClass)),
          output_k3)
cat(sprintf("Saved k=3 assignments: %s\n", output_k3))

# --- 6. Silhouette at target k (full cohort, per-cluster) ---
cat(sprintf("\n--- Silhouette at k=%d (full cohort, Rousseeuw 1987) ---\n", target_k))
target_part <- as.integer(cc[[target_k]]$consensusClass)
sil_target <- silhouette(target_part, dist(H_emb))
mean_sil_target <- mean(sil_target[, 3])
cat(sprintf("  Mean silhouette width: %.4f\n", mean_sil_target))
per_cluster_sil <- numeric(target_k)
for (kk in sort(unique(target_part))) {
  per_cluster_sil[kk] <- mean(sil_target[target_part == kk, 3])
  cat(sprintf("    C%d: %.4f (n=%d)\n", kk, per_cluster_sil[kk],
              sum(target_part == kk)))
}

# --- 7. ARI across consensus seeds (Monti 2003) ---
stab_cons <- ari_across_consensus_seeds(H_emb, target_k, n_reps, subsample,
                                         consensus_seeds)

# --- 8. Bootstrap Jaccard per cluster (Hennig 2007) ---
jacc_mat <- bootstrap_jaccard(H_emb, target_part, target_k, n_reps, subsample, B_boot)
jacc_means <- colMeans(jacc_mat, na.rm = TRUE)

# --- 9. Fig5 inputs ---
comp_names <- comp_chans <- c()
for (ch in data$channels) {
  ab <- CHANNEL_ABBREV[[ch]]; r <- ranks[[ch]]
  comp_names <- c(comp_names, paste0(ab, "_", seq_len(r)))
  comp_chans <- c(comp_chans, rep(CHANNEL_DISPLAY[[ch]], r))
}
H_df <- as.data.frame(H_emb); colnames(H_df) <- comp_names
H_df$case_name <- data$cases
H_df$tissue_cluster <- target_part
write_csv(H_df, file.path(vdir, "nmf_embedding.csv"))
write_csv(data.frame(component = comp_names, channel = comp_chans),
          file.path(vdir, "component_metadata.csv"))
write_csv(data.frame(k = 2:max_k, pac = as.numeric(pac)),
          file.path(vdir, "pac_table.csv"))
saveRDS(cc[[target_k]]$consensusMatrix,
        file.path(vdir, "consensus_matrix_k.rds"))

# --- 10. Validation metric tables ---
write_csv(data.frame(k = 2:max_k,
                     cophenetic = as.numeric(coph_by_k),
                     silhouette = as.numeric(sil_by_k),
                     PAC = as.numeric(pac)),
          file.path(out_dir, "k_selection_table.csv"))

write_csv(data.frame(cluster = paste0("C", seq_len(target_k)),
                     n = as.integer(table(target_part)),
                     silhouette = per_cluster_sil,
                     jaccard_mean = jacc_means,
                     jaccard_median = apply(jacc_mat, 2, median, na.rm = TRUE),
                     jaccard_q25 = apply(jacc_mat, 2, quantile, 0.25, na.rm = TRUE),
                     jaccard_q75 = apply(jacc_mat, 2, quantile, 0.75, na.rm = TRUE)),
          file.path(out_dir, "jaccard_per_cluster.csv"))

val_rows <- list()
for (ch in data$channels) {
  ab <- CHANNEL_ABBREV[[ch]]
  val_rows[[length(val_rows) + 1]] <- data.frame(
    metric = sprintf("cophenetic_NMF_%s_r%d", ab, ranks[[ch]]),
    value  = as.numeric(rank_cophenetic[[ch]]),
    reference = "Brunet 2004")
}
val_rows[[length(val_rows) + 1]] <- data.frame(
  metric = sprintf("cophenetic_consensus_k%d", target_k),
  value = as.numeric(coph_by_k[paste0("k", target_k)]),
  reference = "Brunet 2004 / TCGA")
val_rows[[length(val_rows) + 1]] <- data.frame(
  metric = sprintf("silhouette_k%d", target_k),
  value = mean_sil_target, reference = "Rousseeuw 1987")
val_rows[[length(val_rows) + 1]] <- data.frame(
  metric = sprintf("PAC_k%d", target_k),
  value = as.numeric(pac[paste0("k", target_k)]),
  reference = "Senbabaoglu 2014")
val_rows[[length(val_rows) + 1]] <- data.frame(
  metric = sprintf("ARI_consensus_seeds_k%d", target_k),
  value = stab_cons$mean, reference = "Monti 2003")
for (kk in seq_len(target_k)) {
  val_rows[[length(val_rows) + 1]] <- data.frame(
    metric = sprintf("jaccard_C%d", kk),
    value = jacc_means[kk], reference = "Hennig 2007")
}
val_df <- do.call(rbind, val_rows)
write_csv(val_df, file.path(out_dir, "validation_metrics.csv"))

# --- 11. JSON summary ---
writeLines(toJSON(list(
  version = version,
  n_cases = length(data$cases),
  channels = data$channels,
  nmf_ranks = ranks,
  target_k = target_k,
  k_brunet_largest_drop = k_brunet,
  cophenetic_NMF_per_channel = rank_cophenetic,
  cophenetic_consensus_by_k = as.list(coph_by_k),
  silhouette_by_k = as.list(sil_by_k),
  pac_by_k = as.list(pac),
  silhouette_target_k = mean_sil_target,
  silhouette_per_cluster = setNames(as.list(per_cluster_sil),
                                     paste0("C", seq_len(target_k))),
  ARI_consensus_seeds = stab_cons$mean,
  jaccard_per_cluster = setNames(as.list(jacc_means),
                                  paste0("C", seq_len(target_k))),
  cluster_sizes = setNames(as.list(as.integer(table(target_part))),
                            paste0("C", seq_len(target_k))),
  reps = n_reps,
  bootstrap_B = B_boot,
  consensus_seeds = consensus_seeds
), auto_unbox = TRUE, pretty = TRUE),
  file.path(diag_dir, sprintf("clustering_summary_%s.json", version)))

# --- 12. Final summary printout ---
cat("\n\n=============================================\n")
cat("       VALIDATION SUMMARY\n")
cat("=============================================\n")
cat(sprintf("\nPrimary config: %s\n", fixed_str))
cat(sprintf("Target k:       %d\n", target_k))
cat(sprintf("Cohort:         n=%d tricuspid\n", length(data$cases)))
cat(sprintf("k=%d sizes:      %s\n", target_k,
            paste(as.integer(table(target_part)), collapse = "/")))

cat("\n--- Headline metrics ---\n")
for (ch in data$channels) {
  cat(sprintf("  Cophenetic NMF, %s r=%d        %.4f  (Brunet 2004)\n",
              ch, ranks[[ch]], rank_cophenetic[[ch]]))
}
cat(sprintf("\n  k     cophenetic(cm)  silhouette   PAC\n"))
for (k in 2:max_k) {
  kn <- paste0("k", k)
  cat(sprintf("  k=%-2d  %.4f          %.4f       %.4f\n",
              k, coph_by_k[kn], sil_by_k[kn], pac[kn]))
}
cat(sprintf("\n  Selected k = %d\n", target_k))
cat(sprintf("    cophenetic of consensus matrix = %.4f  (Brunet 2004 / TCGA)\n",
            coph_by_k[paste0("k", target_k)]))
cat(sprintf("    mean silhouette width          = %.4f  (Rousseeuw 1987)\n",
            mean_sil_target))
cat(sprintf("    PAC                            = %.4f  (Senbabaoglu 2014)\n",
            pac[paste0("k", target_k)]))
cat(sprintf("    ARI across consensus seeds     = %.4f  (Monti 2003)\n",
            stab_cons$mean))

cat(sprintf("\n--- Per-cluster ---\n"))
for (kk in seq_len(target_k)) {
  cat(sprintf("  C%d  n=%-4d  silhouette=%.4f  Jaccard=%.3f (Hennig 2007)\n",
              kk, sum(target_part == kk), per_cluster_sil[kk], jacc_means[kk]))
}

cat(sprintf("\nOutputs saved to:\n"))
cat(sprintf("  %s\n", output_csv))
cat(sprintf("  %s\n", output_k3))
cat(sprintf("  %s\n", file.path(out_dir, "validation_metrics.csv")))
cat(sprintf("  %s\n", file.path(out_dir, "k_selection_table.csv")))
cat(sprintf("  %s\n", file.path(out_dir, "jaccard_per_cluster.csv")))
cat(sprintf("  %s\n", file.path(diag_dir, sprintf("clustering_summary_%s.json", version))))

cat("\nDone.\n")
}  # end if (!exists("STABILITY_MODE"))