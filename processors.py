"""
Processors for Vegetation Vulnerability Analysis.
Handles metric pivoting, trend calculation, aggregation, and scoring with memory optimizations.
"""

import gc
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm.auto import tqdm

from config import VegetationConfig
from data_loaders import SpatialDataManager

logger = logging.getLogger(__name__)


class MetricProcessor:
    """
    Processes vegetation vulnerability metrics with memory-optimized operations.
    Handles z-score standardization, trend analysis, and rolling statistics.
    """

    def __init__(self, config: VegetationConfig, spatial_manager: SpatialDataManager):
        self.config = config
        self.spatial_manager = spatial_manager

    def metric_scores(self, df, sum_cols, standard_score_bins, soilmoist_bins):
        """
        Converts metric columns to integer vulnerability scores based on provided bins.
        Returns only the score columns along with UID and date. Input DataFrame is untouched.

        Args:
            df (pd.DataFrame): Input DataFrame with UID, date, and metric columns
            col_bins (dict): Mapping of column names to bin edges

        Returns:
            pd.DataFrame: DataFrame with UID, date, and score columns
        """
        import pandas as pd

        # Skeleton output with keys
        result = df[["UID", "date"]].copy()

        # TODO remove separate soilmoisture scoring
        col_bins = {
            c: (soilmoist_bins if c.startswith("soilmoist") else standard_score_bins)
            for c in sum_cols
        }

        for col, bins in col_bins.items():
            # Highest vulnerability = lowest bin = largest numeric score
            scores = list(range(len(bins) - 1, 0, -1))
            result[col + "_score"] = pd.cut(
                df[col], bins=bins, labels=scores, include_lowest=True
            ).astype(float)  # float avoids categorical overhead

        return result

    def trend_slope(
        self,
        df,
        z_cols,
        temporal_scale,
        trend_window_width,
    ):
        """
        Memory-efficient trend slope calculation.
        Returns only slope columns (does NOT modify input DataFrame).

        Args:
            df (pd.DataFrame): Input data (must contain UID, date, z_cols)
            z_cols (list): Z-score column names
            trend_window_width (int): Window size in years
            freq (str): Frequency ('Y' or 'M')

        Returns:
            pd.DataFrame: DataFrame with UID, date, and slope columns
        """
        import numpy as np

        is_monthly = temporal_scale == "monthly"
        window_size = 1 + (int(trend_window_width * 12) if is_monthly else int(trend_window_width))

        # --- Sort without mutating original ---
        df_sorted = df.sort_values(["UID", "date"])

        # --- Build result skeleton (small copy only of keys) ---
        result = df_sorted[["UID", "date"]].copy()

        # --- Create temporary time index (Series only, not added to df) ---
        time_idx = df_sorted.groupby("UID").cumcount()
        if is_monthly:
            time_idx = time_idx / 12.0

        # Precompute variance of x
        # x_var_const = np.var(np.arange(window_size), ddof=1)

        raw_range = np.arange(window_size)
        if is_monthly:
            x_var_const = np.var(raw_range / 12.0, ddof=1)
        else:
            x_var_const = np.var(raw_range, ddof=1)

        # Min periods
        min_periods = 12 if is_monthly else 2

        for col in z_cols:
            col_name = f"{col}_slope_{trend_window_width}yr"

            # Rolling covariance (no df mutation)
            rolling_cov = (
                df_sorted[col].rolling(window=window_size, min_periods=min_periods).cov(time_idx)
            )

            slope = rolling_cov / x_var_const

            # Mask cross-UID boundary leakage
            valid_group_mask = df_sorted["UID"] == df_sorted["UID"].shift(window_size - 1)
            slope = slope.where(valid_group_mask)

            result[col_name] = slope

        return result

    def rolling_z(self, df, z_cols, temporal_scale, window_years):
        """
        Memory-efficient rolling mean calculation.
        Returns only rolling mean columns (does NOT modify input DataFrame).

        Args:
            df (pd.DataFrame): Input data (must contain UID, date, z_cols)
            z_cols (list): Z-score column names
            window_years (int): Rolling window size in years
            freq (str): Frequency ('Y' or 'M')

        Returns:
            pd.DataFrame: DataFrame with UID, date, and rolling mean columns
        """
        import numpy as np

        is_monthly = temporal_scale == "monthly"
        window = window_years * 12 if is_monthly else window_years
        min_periods = 24 if is_monthly else 2

        # --- Sort without mutating original ---
        df_sorted = df.sort_values(["UID", "date"])

        # --- Result skeleton (minimal memory) ---
        result = df_sorted[["UID", "date"]].copy()

        # --- Rolling means (vectorized, no mutation) ---
        rolled_means = df_sorted[z_cols].rolling(window=window, min_periods=min_periods).mean()

        # --- Mask cross-UID boundary leakage ---
        valid_group_mask = df_sorted["UID"] == df_sorted["UID"].shift(window - 1)
        rolled_means = rolled_means.where(valid_group_mask, np.nan)

        # --- Add rolling columns to result only ---
        for col in z_cols:
            result[f"{col}_roll_{window_years}yr"] = rolled_means[col]

        return result

    def standardise_z(
        self,
        df,
        metrics,
        input_is_percentile_rank=None,
        exclude_years=None,
        date_col="date",
    ):
        """
        Compute z-scores and return ONLY the new z-score columns.

        Args:
            df (pd.DataFrame): Input DataFrame (not modified)
            metrics (list): Column names to standardize
            input_is_standardised (list or str): Columns already standardized
            exclude_years (list): Years to exclude from baseline calculation
            date_col (str): Date column name

        Returns:
            pd.DataFrame: DataFrame with only UID, date, and z-score columns
        """
        import numpy as np

        # Normalize input
        input_is_percentile_rank = (
            [input_is_percentile_rank]
            if isinstance(input_is_percentile_rank, str)
            else input_is_percentile_rank or []
        )

        # Precompute exclusion mask once
        exclude_mask = None
        if exclude_years:
            exclude_mask = df[date_col].dt.year.isin(exclude_years)

        # Start result with only keys (no full copy of df)
        result = df[["UID", date_col]].copy()

        for col in metrics:
            if col in input_is_percentile_rank:
                # Just pass through as z-score
                # result[f"{col}_z"] = df[col]
                # use scipy.stats to convert percentile rank to z-score
                result[f"{col}_z"] = norm.ppf(df[col].clip(0.001, 0.999))
                continue

            if exclude_years:
                calc_data = df[col].where(~exclude_mask)
                grouper = calc_data.groupby(df["UID"], observed=True)
            else:
                grouper = df.groupby("UID", observed=True)[col]

            mean_series = grouper.transform("mean")
            std_series = grouper.transform("std")

            # Avoid divide-by-zero
            std_series = std_series.replace(0, np.nan)

            result[f"{col}_z"] = (df[col] - mean_series) / std_series

        return result

    def monthly_to_mean_annual(self, wit_df, metrics, temporal_scale):
        """
        Aggregates monthly data to annual means with water-year or calendar-year logic.

        Args:
            wit_df (pd.DataFrame): Monthly data
            metrics (list): Columns to aggregate
            water_year (bool): Use water year (July-June) vs calendar year

        Returns:
            pd.DataFrame: Annually aggregated data
        """
        if temporal_scale == "water-year":
            # Water year: July-June, ends June 30th
            target_year = wit_df["date"].dt.year + (wit_df["date"].dt.month >= 7).astype(int)
            month, day = 6, 30
        else:
            # Calendar year: January-December, ends December 31st
            target_year = wit_df["date"].dt.year
            month, day = 12, 31

        # Create temporary annual date Series (no mutation of monthly wit_df)
        annual_dates = pd.to_datetime({"year": target_year, "month": month, "day": day})

        # Define aggregation rules
        agg_rules = dict.fromkeys(metrics, "mean")
        if "count" in wit_df.columns:
            agg_rules["count"] = "sum"

        # Group using UID + temporary annual_dates
        result = (
            wit_df.groupby([wit_df["UID"], annual_dates], observed=True)
            .agg(agg_rules)
            .reset_index()
        )

        # Rename generated column to "date"
        result.rename(columns={"level_1": "date"}, inplace=True)

        return result

    def _append_tsli_scores(self, df, tsli_col="tsli", group_field="grp"):
        """
        Assigns TSLI (Time Since Last Inundation) scores based on vegetation-specific thresholds.
        Efficient iteration over vegetation groups (small number ~18) rather than rows.

        Args:
            df (pd.DataFrame): DataFrame with TSLI data
            tsli_col (str): TSLI column name
            group_field (str): Vegetation group column name

        Returns:
            pd.DataFrame: DataFrame with tsli_score column added
        """
        df["tsli_score"] = np.nan

        # Iterate over vegetation groups (efficient - only ~18 groups)
        for group, thresholds in self.config.vegetation_tsli_stress_thresholds.items():
            mask = df[group_field] == group
            if not mask.any():
                continue

            # Create bins with infinity upper bound
            bins = thresholds + [float("inf")]
            labels = list(range(1, len(bins)))

            # Apply scoring: short TSLI = good condition = low score
            df.loc[mask, "tsli_score"] = pd.cut(
                df.loc[mask, tsli_col], bins=bins, labels=labels, include_lowest=True
            ).astype(float)

            df.loc[df[tsli_col].isna(), "tsli_score"] = (
                len(bins) - 1
            )  # "worst" score if no events in record

        return df

    def process_tsli_robust(self, temporal_scale, chunk_size=10000, use_cache=True):
        """
        Optimized version of original logic.
        Maintains historical accuracy: finds max(end_date) where start_date < cutoff.

        Processes Time Since Last Inundation with optimized memory usage.
        Pre-allocates result list size and uses chunked processing for monthly data.
        Caches results as parquet files for faster subsequent runs.

        Loads inundation data only when needed (cache miss).

        Args:
            temporal_scale (water-year, calendar-year, monthly)
            chunk_size (int): Number of UIDs to process per chunk
            use_cache (bool): Whether to use cached results if available

        Returns:
            pd.DataFrame: TSLI scores by date and UID
        """
        from data_loaders import TsliLoader

        # Check for cached results
        cache_file = Path(f"tsli_{temporal_scale}.parquet")
        if use_cache and cache_file.exists():
            logger.info(f"Loading cached TSLI data from {cache_file}")
            return pd.read_parquet(cache_file)

        logger.info(f"Processing TSLI for {temporal_scale} temporal scale...")

        # Only load inundation data if we need to process (cache miss)
        logger.info("Loading inundation data for TSLI processing...")
        inundation_df = TsliLoader(self.config).load_data(self.spatial_manager.uid_dtype)

        df = inundation_df
        anae_df = self.spatial_manager.anae_gdf

        # 1. Standardize Inputs
        df = df[["UID", "start_date", "end_date"]].copy()
        df["start_date"] = pd.to_datetime(df["start_date"])
        df["end_date"] = pd.to_datetime(df["end_date"])

        # 2. Replicate your original Date Logic
        min_date, max_date = df["start_date"].min(), df["end_date"].max()
        if temporal_scale == "monthly":
            cutoffs = pd.date_range(start=min_date, end=max_date, freq="ME")
        elif temporal_scale == "water-year":
            years = range(min_date.year, max_date.year + 1)
            candidates = pd.to_datetime([f"{y}-06-30" for y in years])
            cutoffs = candidates[(candidates >= min_date) & (candidates <= max_date)]
        else:  # calendar-year
            cutoffs = pd.date_range(start=min_date, end=max_date, freq="YE")

        cutoff_df = pd.DataFrame({"cutoff_date": cutoffs})
        unique_uids = df["UID"].unique()
        all_results = []

        # 3. Chunk by UID to prevent memory overflow
        for i in tqdm(range(0, len(unique_uids), chunk_size), desc="Processing UID Chunks"):
            subset_uids = unique_uids[i : i + chunk_size]
            chunk_events = df[df["UID"].isin(subset_uids)]

            # Cross join: Every event in this chunk meets every cutoff date
            # (5000 UIDs * ~10 events/UID * 480 cutoffs is roughly 24M rows - manageable)
            merged = chunk_events.merge(cutoff_df, how="cross")

            # ORIGINAL LOGIC STEP 1: "idf = df[df['start_date'] < cutoff_date]"
            merged = merged[merged["start_date"] < merged["cutoff_date"]]

            if merged.empty:
                continue

            # ORIGINAL LOGIC STEP 2: "idf.loc[idf['end_date'] > cutoff_date, 'end_date'] = cutoff_date"
            # We use .clip() for better performance than .loc
            merged["effective_end"] = merged["end_date"].clip(upper=merged["cutoff_date"])

            # ORIGINAL LOGIC STEP 3: "Find last inundation event per UID"
            # Group by UID and Cutoff, then find the latest effective_end
            last_inundation = (
                merged.groupby(["UID", "cutoff_date"], observed=True)["effective_end"]
                .max()
                .reset_index()
            )

            # ORIGINAL LOGIC STEP 4: "tsli = (cutoff_date - last['end_date']).dt.days"
            last_inundation["tsli"] = (
                last_inundation["cutoff_date"] - last_inundation["effective_end"]
            ).dt.days

            # Format for final collection
            last_inundation = last_inundation.rename(columns={"cutoff_date": "date"})
            all_results.append(last_inundation[["UID", "date", "tsli"]])

            # Cleanup RAM
            del merged, last_inundation, chunk_events

            gc.collect()

        # 4. Final Assemble
        if not all_results:
            return pd.DataFrame()

        final_tsli = pd.concat(all_results, ignore_index=True)

        # Fill missing effective_end with the start of observation period
        final_tsli["tsli"] = final_tsli["tsli"].fillna((final_tsli["date"] - min_date).dt.days)

        # LEFT Join with ANAE groups (keep original UID dtype)
        # left join keeps duplicate UID where polygon straddles valley boundary
        final_tsli = final_tsli.merge(anae_df[["UID", "grp"]].reset_index(), on="UID", how="left")

        result = self._append_tsli_scores(final_tsli)

        # Cache results
        if use_cache:
            logger.info(f"Caching TSLI results to {cache_file}")
            result.to_parquet(cache_file, index=False)

        return result

    def sum_rolling_and_slope(
        self, rolling_df, slope_df, z_cols, year_window_width, trend_window_width
    ):
        logger.info("Combining rolling means and slopes into summed metrics...")
        df = rolling_df.merge(slope_df, on=["UID", "date"])

        result = df[["UID", "date"]].copy()

        for z_col in z_cols:
            result[f"{z_col}_sum"] = (
                df[f"{z_col}_roll_{year_window_width}yr"]
                + df[f"{z_col}_roll_{year_window_width}yr_slope_{trend_window_width}yr"]
            )

        return result


class VulnerabilityAggregator:
    """
    Aggregates vulnerability metrics across spatial scales with memory optimizations.
    Handles area-weighted averaging and vulnerability score calculation.
    """

    def __init__(self, config: VegetationConfig, spatial_manager: SpatialDataManager):
        self.config = config
        self.spatial_manager = spatial_manager

    def _extract_scores(self, index_cols, fname, score_cols):
        """
        Extracts scores from saved CSV files for analysis.

        Args:
            index_cols (list): Index column names
            fname (str): Filename to read
            score_cols (list): Score column names

        Returns:
            pd.DataFrame: Extracted scores with renamed columns
        """
        path = self.config.data_dir / fname
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")

        scores = pd.read_csv(path, low_memory=False)
        scores = scores[index_cols + score_cols].set_index(index_cols)
        return scores.rename(columns={c: c[-4:] for c in scores.columns})

    def _sum_and_normalise_weighted(self, df_subset: pd.DataFrame) -> pd.Series:
        """
        Calculates weighted mean of condition/stress scores handling missing data.
        Normalizes result to 0-1 scale to account for varying numbers of contributing metrics.

        Args:
            df_subset (pd.DataFrame): Subset of score columns to aggregate

        Returns:
            pd.Series: Normalized vulnerability scores
        """
        # Calculate row-wise mean (automatically handles missing data)
        raw_scores = df_subset.mean(axis=1, numeric_only=True, skipna=True)
        return self._normalise(raw_scores)

    def _normalise(self, data: pd.Series) -> pd.Series:
        """
        Normalizes values to 0-1 scale using global min/max.
        Handles edge case where all values are identical (prevents division by zero).

        Args:
            data (pd.Series): Input data to normalize

        Returns:
            pd.Series: Normalized data (0-1 scale)
        """
        d_min = data.min()
        d_max = data.max()

        # Handle DataFrame input (reduce to scalar)
        if isinstance(d_min, pd.Series):
            d_min = d_min.min()
            d_max = d_max.max()

        # Prevent division by zero for constant data
        denom = d_max - d_min
        if denom == 0:
            return data * 0

        return (data - d_min) / denom

    def _aggregate(self, df, ag_name, anae_group=True, tag=""):
        """
        Memory-optimized area-weighted aggregation using vectorized operations.
        Avoids creating intermediate DataFrames for 13M+ row datasets.

        Args:
            df (pd.DataFrame): Input data with Area_Ha column
            ag_name (str): Aggregation level name
            anae_group (bool): Include vegetation group in aggregation
            tag (str): Optional tag for processing

        Returns:
            pd.DataFrame: Aggregated data with area-weighted means
        """
        print(f"Aggregating {ag_name}...")

        # Identify columns to aggregate
        columns_to_agg = [c for c in df.columns if c.endswith(("_2yr", "_5yr", "_score"))]
        ag_fields = self.spatial_manager.agg_fields[ag_name]

        # Standardize aggregation fields to list
        group_cols = [ag_fields] if isinstance(ag_fields, str) else list(ag_fields)

        # Add required grouping columns
        if "date" not in group_cols:
            group_cols.append("date")
        if anae_group and "grp" not in group_cols:
            group_cols.append("grp")

        # Validate required columns exist
        if not set(group_cols).issubset(df.columns):
            print(
                f"Cannot aggregate {ag_name}, missing columns: {set(group_cols) - set(df.columns)}"
            )
            return None

        # Identify data columns (exclude grouping and area columns)
        data_cols = [c for c in columns_to_agg if c not in group_cols and c != "Area_Ha"]
        grouper = [df[c] for c in group_cols]

        # Memory-optimized area-weighted calculation
        # Use single multiply operation instead of creating intermediate DataFrames
        area_weights = df["Area_Ha"]

        # Numerator: sum of (value * area) for each group
        numerator = (
            df[data_cols].multiply(area_weights, axis=0).groupby(grouper, observed=True).sum()
        )

        # Denominator: sum of areas where values exist
        denominator = (
            df[data_cols]
            .notna()
            .multiply(area_weights, axis=0)
            .groupby(grouper, observed=True)
            .sum()
        )

        return numerator / denominator

    def _join_dataframes(self, z_scores_df, tsli_df):
        """
        Joins z-scores and TSLI data with spatial information.
        Uses left join to prevent row inflation from mismatched dates.

        Args:
            z_scores_df (pd.DataFrame): Processed z-scores
            tsli_df (pd.DataFrame): TSLI scores

        Returns:
            pd.DataFrame: Combined dataset with spatial attributes
        """
        # Use left join to maintain z_scores_df structure and prevent row inflation

        z_score_cols = [col for col in z_scores_df.columns if col.endswith("_score")]

        df = z_scores_df[["UID", "date"] + z_score_cols].merge(
            tsli_df[["UID", "date", "tsli_score"]], on=["UID", "date"], how="left"
        )

        # Add spatial attributes (drop geometry to save memory)
        return df.merge(
            self.spatial_manager.anae_gdf.drop(columns="geometry"), on="UID", how="left"
        )

    def _calc_vulnerability_and_save(self, df, ag_name, save_file_root: str):
        """
        Calculates final vulnerability scores and saves results.
        Uses sanitized file paths to prevent directory traversal attacks.

        Args:
            df (pd.DataFrame): Aggregated vulnerability data
            ag_name (str): Aggregation level name
            save_file_root (str): Base filename for output

        Returns:
            pd.DataFrame: Vulnerability scores (condition, stress, vulnerability)
        """
        print(f"\nCalculating vulnerability scores for {ag_name}...")

        # Calculate condition score (vegetation health indicators)
        condition = self._sum_and_normalise_weighted(
            df[["npv+pv+wet_median_z_sum_score", "ndvi_z_sum_score"]]
        )

        # Calculate stress score (water availability indicators)
        stress = self._sum_and_normalise_weighted(
            df[["water+wet_median_z_sum_score", "soilmoist_z_sum_score", "tsli_score"]]
        )

        # Create vulnerability DataFrame
        vuln_df = pd.DataFrame(index=df.index)
        vuln_df["condition"] = condition
        vuln_df["stress"] = stress

        # Calculate final vulnerability: condition + stress, then normalize
        # High condition = resilient, high stress = vulnerable
        raw_vulnerability = condition + stress
        vuln_df["vulnerability"] = self._normalise(raw_vulnerability)

        # Sanitize filename to prevent path traversal attacks
        safe_root = re.sub(r"[^\w\-_.]", "_", save_file_root)
        safe_ag_name = re.sub(r"[^\w\-_.]", "_", ag_name)

        fname = Path(f"vulnerability-{safe_root}-{safe_ag_name}")
        zip_file = fname.with_suffix(".zip")

        logger.info(f"Saving vulnerability scores to {zip_file}")

        # Save as compressed CSV
        compression_opts = {"method": "zip", "archive_name": fname.with_suffix(".csv").name}
        vuln_df.sort_index(axis=1).round(2).to_csv(zip_file, compression=compression_opts)

        return vuln_df

    def detect_temporal_scale_from_df(self, df):
        """
        Infers temporal scale from date column patterns.

        Args:
            df (pd.DataFrame): DataFrame with date column

        Returns:
            str: Detected temporal scale ('monthly', 'cal-year', 'water-year', 'unknown_interval')
        """
        if df["date"].dtype == "datetime64[ns]":
            if df["date"].dt.is_month_end.all():
                return "monthly"
            elif df["date"].dt.is_year_end.all():
                return "cal-year"
            elif df["date"].dt.month.eq(6).all() & df["date"].dt.day.eq(30).all():
                return "water-year"
        return "unknown_interval"

    def aggregate_and_save_vulnerability(self, z_scores_df, tsli_df, save_file_root: str = None):
        """
        Main aggregation pipeline: joins data, aggregates across scales, calculates vulnerability.

        Args:
            z_scores_df (pd.DataFrame): Processed z-scores
            tsli_df (pd.DataFrame): TSLI scores
            save_file_root (str): Base filename for outputs
        """
        # Join all data sources
        df = self._join_dataframes(z_scores_df, tsli_df)

        # Auto-detect temporal scale if not provided
        if not save_file_root:
            save_file_root = self.detect_temporal_scale_from_df(df)

        # Save intermediate results
        df.to_parquet(f"{save_file_root}_vuln_scores.parquet", index=False)

        # Process each aggregation level
        for ag_name in self.spatial_manager.aggregators:
            agg_df = self._aggregate(df, ag_name, anae_group=False)

            # Save aggregated data
            fname = f"metric-scores-{save_file_root}-{ag_name}.csv"
            agg_df.to_csv(fname)
            logger.debug(f"Saved aggregated {ag_name} data to {fname}")

            # Calculate and save vulnerability scores
            self._calc_vulnerability_and_save(agg_df, ag_name, save_file_root)
