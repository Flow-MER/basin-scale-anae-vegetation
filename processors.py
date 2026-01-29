"""
Processors for Vegetation Vulnerability Analysis.
Handles metric pivoting, trend calculation, aggregation, and scoring.
"""
import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from config import VegetationConfig
from data_loaders import SpatialDataManager, normalise

class MetricProcessor:
    def __init__(self, config: VegetationConfig, spatial_manager: SpatialDataManager):
        self.config = config
        self.spatial_manager = spatial_manager

    def _pivot_year(self, df, metric, pkey="UID"):
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        if metric not in df.columns:
            raise KeyError(f"Column '{metric}' not found")

        all_years = df["year"].unique().tolist()
        baseline_years = [y for y in all_years if y not in self.config.MILLENNIUM_DROUGHT]

        df_pivot = df.pivot(index=pkey, columns="year", values=metric)
        baseline_data = df_pivot[baseline_years]

        df_count = df_pivot.count(axis=1, numeric_only=True).rename("count")
        df_baseline = baseline_data.mean(axis=1, numeric_only=True).rename("baseline")
        df_stddev = baseline_data.std(axis=1, numeric_only=True).rename("stddev")
        df_max = baseline_data.max(axis=1, numeric_only=True).rename("max")
        df_median = baseline_data.median(axis=1, numeric_only=True).rename("median")
        
        deviations = baseline_data.sub(df_median, axis=0).abs()
        df_mad = deviations.median(axis=1, numeric_only=True).rename("mad")

        return pd.concat([df_pivot, df_count, df_baseline, df_stddev, df_max, df_median, df_mad], axis=1)

    @staticmethod
    def _fn_slope(d):
        y_values = d.values
        if len(y_values) < 2:
            return float("NaN")
        x = np.arange(len(y_values))
        x_std = (x - x.mean()) / x.std()
        return round(np.polyfit(x_std, y_values, 1)[0], 4)

    def _fn_average_trend(self, df, period, trend_period=None, nobaseline=False):
        if trend_period is None:
            trend_period = period
        
        if nobaseline:
            tmp = df[period]
        else:
            tmp = df[period].subtract(df["baseline"], axis=0).div(df["stddev"], axis=0) + 0
        
        tmp = tmp.fillna(0)
        
        slope_col = "Trend" + str(period[-1])
        ave_col = "Ave" + str(period[-1])
        
        tmp[ave_col] = tmp[period].mean(axis=1, numeric_only=True)
        cols = [ave_col]
        
        if len(period) > 1:
            tmp[slope_col] = tmp[trend_period].apply(self._fn_slope, axis=1)
            cols.append(slope_col)
            
        return tmp[cols]

    def _deviation_from_baseline(self, df, metric, year_window_width, trend_window_width, no_baseline=False, pkey="UID"):
        allyears = df["year"].unique().tolist()
        _years = range(allyears[0] + year_window_width - 1, allyears[-1] + 1)
        metric_name = metric.replace("+", "").lower()

        pivot = self._pivot_year(df, metric, pkey).round(4)
        # Optional: Save pivot if needed, skipping for now to reduce I/O or make optional
        
        dfs = []
        for year in tqdm(_years, desc=f"{metric} in {year_window_width}y window"):
            period = list(range(year - year_window_width + 1, year + 1))
            trend_period = period[-trend_window_width:] if trend_window_width else period
            
            stats_df = self._fn_average_trend(pivot, period, trend_period, nobaseline=no_baseline)
            
            if len(stats_df.columns) > 1:
                stats_df["sum" + metric_name + str(year)] = stats_df.sum(axis=1)
            
            dfs.append(stats_df)
            
        metrics_df = pd.concat(dfs, axis=1).sort_index(axis=1)
        
        # Rename columns
        rename_map = {}
        for c in metrics_df.columns:
            c_clean = c.strip()
            c_clean = c_clean.replace("Ave", metric_name)
            c_clean = c_clean.replace("Trend", "T" + metric_name)
            rename_map[c] = c_clean
            
        return metrics_df.rename(columns=rename_map)

    def _aggregate_area_weighted(self, df, agg_name, anae_grp=["grp"]):
        agg_df = self.spatial_manager.aggregators[agg_name]
        agg_fields = self.spatial_manager.agg_fields[agg_name]
        
        joined_df = df.join(agg_df, how="inner").replace([np.inf, -np.inf], np.nan)
        cols = df.columns.values.tolist()

        if agg_name == "ANAE":
            return joined_df[["grp"] + cols + ["Area_Ha"]].set_index("grp", append=True)
        else:
            joined_df[cols] = joined_df[cols].multiply(joined_df["Area_Ha"], axis=0)
            agg_data = (
                joined_df[cols + ["Area_Ha"] + agg_fields + anae_grp]
                .groupby(agg_fields + anae_grp)
                .sum(numeric_only=True)
            )
            agg_data[cols] = agg_data[cols].div(agg_data["Area_Ha"], axis=0)
            return agg_data[cols + ["Area_Ha"]]

    def _append_metric_scores(self, df, col_list, _bins, _labels):
        dfs = [df]
        for col in col_list:
            dfs.append(
                pd.cut(df[col], bins=_bins, labels=_labels, include_lowest=True)
                .astype("float")
                .rename("sc" + col)
            )
        return pd.concat(dfs, axis=1)

    def process_metric(self, df, metrics, year_window_width=5, trend_window_width=None, 
                       _bins=[float("-inf"), -1, 0, float("inf")], reverse_scores=False, 
                       no_baseline=False, tag=""):
        if isinstance(metrics, str):
            metrics = [metrics]
        
        if trend_window_width is None:
            trend_window_width = year_window_width

        _scores = list(range(1, len(_bins)))
        if reverse_scores:
            _scores = _scores[::-1]

        for metric in metrics:
            metrics_df = self._deviation_from_baseline(
                df, metric, year_window_width, trend_window_width, no_baseline=no_baseline
            )
            
            print(f"Aggregating {metric}...")
            for ag_name in self.spatial_manager.aggregators:
                fname = f"{metric}_{ag_name}_{year_window_width}yr{tag}.csv"
                agg_result = self._aggregate_area_weighted(metrics_df, ag_name).round(4)
                
                col_list = [c for c in agg_result.columns if c != "Area_Ha"]
                scored_result = self._append_metric_scores(agg_result, col_list, _bins, _scores)
                
                out_path = self.config.DATA_DIR / fname
                scored_result.to_csv(out_path)
                print(f"Saved {fname}")

    def process_tsli_scores(self, tsli_df):
        thresholds = self.config.VEGETATION_TSLI_STRESS_THRESHOLDS
        cols = [f"tsli{y}" for y in self.config.ALL_TIME]
        scores = [f"sc_tsli{y}" for y in self.config.ALL_TIME]
        tsli_df[scores] = np.nan

        for group_name, group_df in tqdm(tsli_df.groupby('grp'), desc="Scoring TSLI"):
            if group_name in thresholds:
                _bins = thresholds[group_name] + [float("inf")]
                _labels = range(len(_bins) - 1, 0, -1)
                
                for col, score in zip(cols, scores):
                    if col in group_df.columns:
                        tsli_df.loc[group_df.index, score] = pd.cut(
                            group_df[col], bins=_bins, labels=_labels, right=False
                        ).astype('float')

        cols_to_agg = cols + scores
        
        for ag_name in self.spatial_manager.aggregators:
            fname = f"time_since_last_inundation_{ag_name}_vegetation_stress.csv"
            agg_result = self._aggregate_area_weighted(tsli_df[cols_to_agg], ag_name)
            agg_result.round(1).to_csv(self.config.DATA_DIR / fname)
            print(f"Saved {fname}")

class VulnerabilityAggregator:
    def __init__(self, config: VegetationConfig, spatial_manager: SpatialDataManager):
        self.config = config
        self.spatial_manager = spatial_manager

    def _extract_scores(self, index_cols, fname, score_cols):
        path = self.config.DATA_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
            
        scores = pd.read_csv(path, low_memory=False)
        scores = scores[index_cols + score_cols].set_index(index_cols)
        return scores.rename(columns={c: c[-4:] for c in scores.columns})

    def _sum_and_normalise_weighted(self, df):
        sum_df = df.groupby(level=df.index.names).sum()
        count_df = df.groupby(level=df.index.names).count()
        return normalise(sum_df.divide(count_df))

    def combine_and_save(self):
        years = range(self.config.ALL_TIME[0] + self.config.VEG_WINDOW_WIDTH - 1, self.config.ALL_TIME[-1] + 1)
        
        print("\nCalculating Vulnerability Scores...")
        
        for ag_name in self.spatial_manager.aggregators:
            fname = f"MDB_vegetation_vulnerability_{ag_name}.csv"
            index_cols = self.spatial_manager.agg_fields[ag_name] + ["grp"]
            print(f"Processing {ag_name}...")

            # Condition
            cond_a = self._extract_scores(
                index_cols, f"npv+pv+wet_median_{ag_name}_5yr.csv", 
                [f"scsumnpvpvwet_median{y}" for y in years]
            )
            cond_b = self._extract_scores(
                index_cols, f"ndvi_{ag_name}_5yr.csv", 
                [f"scsumndvi{y}" for y in years]
            )
            combined_cond = pd.concat([cond_a, cond_b], axis=0)
            cond_df = self._sum_and_normalise_weighted(combined_cond).round(2)

            # Stress
            stress_a = self._extract_scores(
                index_cols, f"water+wet_median_{ag_name}_5yr.csv",
                [f"scsumwaterwet_median{y}" for y in years]
            )
            stress_b = self._extract_scores(
                index_cols, f"time_since_last_inundation_{ag_name}_vegetation_stress.csv",
                [f"sc_tsli{y}" for y in years]
            )
            stress_c = self._extract_scores(
                index_cols, f"soilmoist_{ag_name}_5yr.csv",
                [f"scsumsoilmoist{y}" for y in years]
            )
            combined_stress = pd.concat([stress_a, stress_b, stress_c], axis=0)
            stress_df = self._sum_and_normalise_weighted(combined_stress).round(2)

            # Vulnerability
            tmp_df = pd.concat([cond_df, stress_df], axis=0)
            vul_df = normalise(tmp_df.groupby(tmp_df.index.names).sum()).round(2)

            # Rename and Save
            cond_df = cond_df.rename(columns={c: f"cond{c}" for c in cond_df.columns})
            stress_df = stress_df.rename(columns={c: f"stress{c}" for c in stress_df.columns})
            vul_df = vul_df.rename(columns={c: f"vul{c}" for c in vul_df.columns})

            zip_file = self.config.DATA_DIR / f"{os.path.splitext(fname)[0]}.zip"
            compression_opts = dict(method='zip', archive_name=fname)
            
            (pd.concat([cond_df, stress_df, vul_df], axis=1)
             .sort_index(axis=1)
             .to_csv(zip_file, compression=compression_opts))
            
            print(f"Saved {zip_file}")
