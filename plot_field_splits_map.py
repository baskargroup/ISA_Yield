from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import tempfile
from urllib.request import urlretrieve

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import contextily as ctx


SPLIT_FILES = {
    "train": "train_{crop}.txt",
    "val": "val_{crop}.txt",
    "test": "test_{crop}.txt",
}

SPLIT_COLORS = {
    "train": "#0072B2",
    "val": "#E69F00",
    "test": "#D55E00",
}

SPLIT_MARKERS = {
    "train": "o",
    "val": "^",
    "test": "s",
}

US_STATES_GEOJSON_URL = (
    "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
)


def _normalize_crop(value: str) -> str:
    return value.strip().lower()


def _extract_year_from_layer_id(layer_id: str) -> int | None:
    match = re.match(r"^ST(\d{4})", layer_id)
    if not match:
        return None
    return int(match.group(1))


def load_crop_centroids(raw_yield_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(raw_yield_dir.glob("Crop_Classification_*.csv")):
        year_match = re.search(r"(\d{4})$", csv_path.stem)
        if year_match is None:
            continue
        source_year = int(year_match.group(1))

        df = pd.read_csv(csv_path)
        column_map = {col: col.strip().strip('"') for col in df.columns}
        df = df.rename(columns=column_map)

        required = {"Layer_ID", "X_cent", "Y_cent", "Crop"}
        if not required.issubset(df.columns):
            continue

        subset = df[["Layer_ID", "X_cent", "Y_cent", "Crop"]].copy()
        subset["Layer_ID"] = subset["Layer_ID"].astype(str).str.replace('"', "", regex=False)
        subset["Crop"] = subset["Crop"].astype(str).str.replace('"', "", regex=False)
        subset["crop_norm"] = subset["Crop"].map(_normalize_crop)
        subset["lat"] = pd.to_numeric(subset["Y_cent"], errors="coerce")
        subset["lon"] = pd.to_numeric(subset["X_cent"], errors="coerce")
        subset["year"] = source_year
        subset = subset.dropna(subset=["lat", "lon"])
        frames.append(subset[["Layer_ID", "crop_norm", "year", "lat", "lon"]])

    if not frames:
        raise FileNotFoundError(
            f"No usable Crop_Classification_*.csv files found in {raw_yield_dir}"
        )

    all_points = pd.concat(frames, ignore_index=True)
    all_points = all_points.drop_duplicates(subset=["Layer_ID", "crop_norm", "year"], keep="first")
    return all_points


def parse_split_file(split_file: Path, crop: str, split_name: str) -> pd.DataFrame:
    rows = []
    with split_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if not value:
                continue
            if "_" in value:
                layer_id, crop_in_file = value.rsplit("_", 1)
            else:
                layer_id, crop_in_file = value, crop
            rows.append(
                {
                    "Layer_ID": layer_id,
                    "crop_norm": _normalize_crop(crop_in_file),
                    "year": _extract_year_from_layer_id(layer_id),
                    "split": split_name,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["Layer_ID", "crop_norm", "year", "split"])
    return pd.DataFrame(rows)


def build_crop_map(
    crop: str,
    split_dir: Path,
    centroids: pd.DataFrame,
 ) -> pd.DataFrame:
    crop_norm = _normalize_crop(crop)
    points = centroids[centroids["crop_norm"] == crop_norm].copy()

    split_frames = []
    for split_name, pattern in SPLIT_FILES.items():
        split_file = split_dir / pattern.format(crop=crop_norm)
        if not split_file.exists():
            raise FileNotFoundError(f"Missing split file: {split_file}")
        split_frames.append(parse_split_file(split_file, crop_norm, split_name))

    split_df = pd.concat(split_frames, ignore_index=True)
    merged = split_df.merge(points, on=["Layer_ID", "crop_norm", "year"], how="left")

    missing_year = merged[merged["year"].isna()]
    if not missing_year.empty:
        missing_year_ids = ", ".join(sorted(missing_year["Layer_ID"].unique())[:10])
        print(
            f"Warning: {len(missing_year)} rows for crop '{crop}' have invalid Layer_ID format "
            f"(cannot extract year). Examples: {missing_year_ids}"
        )

    missing = merged[merged[["lat", "lon"]].isna().any(axis=1)]
    if not missing.empty:
        missing_ids = ", ".join(sorted(missing["Layer_ID"].unique())[:10])
        print(
            f"Warning: {len(missing)} rows for crop '{crop}' have no centroid coordinates. "
            f"Examples: {missing_ids}"
        )
    merged = merged.dropna(subset=["lat", "lon"]).copy()

    if merged.empty:
        raise ValueError(f"No mappable points found for crop '{crop}' in {split_dir}")

    return merged


def save_folium_map(merged: pd.DataFrame, output_html: Path) -> None:
    center = [merged["lat"].mean(), merged["lon"].mean()]
    fmap = folium.Map(location=center, zoom_start=7, tiles="CartoDB positron")

    for split_name in ["train", "val", "test"]:
        split_points = merged[merged["split"] == split_name]
        feature_group = folium.FeatureGroup(name=f"{split_name.title()} ({len(split_points)})")

        for _, row in split_points.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=3,
                color=SPLIT_COLORS[split_name],
                fill=True,
                fill_color=SPLIT_COLORS[split_name],
                fill_opacity=0.8,
                weight=1,
                popup=f"{row['Layer_ID']} ({split_name})",
            ).add_to(feature_group)

        feature_group.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(output_html))


def _get_style_config(basemap_style: str) -> dict:
    if basemap_style == "satellite":
        return {
            "marker_edge": "#111111",
            "legend_face": (1, 1, 1, 0.9),
            "title_face": (1, 1, 1, 0.82),
            "title_color": "#111111",
            "marker_size": 48,
            "marker_alpha": 0.95,
        }
    return {
        "marker_edge": "white",
        "legend_face": (1, 1, 1, 0.96),
        "title_face": (1, 1, 1, 0.92),
        "title_color": "#111111",
        "marker_size": 44,
        "marker_alpha": 0.9,
    }


def _get_basemap_sources(style: str) -> list:
    style_norm = style.strip().lower()
    if style_norm == "satellite":
        return [
            ctx.providers.Esri.WorldImagery,
            ctx.providers.Esri.WorldStreetMap,
            ctx.providers.CartoDB.PositronNoLabels,
        ]
    if style_norm == "terrain":
        return [
            ctx.providers.Esri.WorldTopoMap,
            ctx.providers.OpenTopoMap,
            ctx.providers.OpenStreetMap.Mapnik,
        ]
    return [
        ctx.providers.CartoDB.PositronNoLabels,
        ctx.providers.OpenStreetMap.Mapnik,
    ]


def _add_basemap_with_fallback(ax: plt.Axes, style: str) -> None:
    errors: list[str] = []
    for source in _get_basemap_sources(style):
        try:
            ctx.add_basemap(
                ax,
                source=source,
                attribution=False,
                zoom="auto",
            )
            return
        except Exception as exc:
            source_name = getattr(source, "name", str(source))
            errors.append(f"{source_name}: {exc}")

    joined = " | ".join(errors[:2]) if errors else "unknown error"
    print(f"Warning: basemap tiles unavailable ({joined}). Rendering points without tiles.")


def _load_state_boundary(cache_path: Path, state_name: str) -> gpd.GeoDataFrame | None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp_file:
                temp_path = Path(tmp_file.name)
            urlretrieve(US_STATES_GEOJSON_URL, temp_path)
            shutil.copyfile(temp_path, cache_path)
        except Exception as exc:
            print(f"Warning: could not download US state boundaries ({exc}).")
            return None
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)

    try:
        states = gpd.read_file(cache_path)
    except Exception as exc:
        print(f"Warning: could not read state boundary file {cache_path} ({exc}).")
        return None

    if "name" not in states.columns:
        print("Warning: state boundary file missing 'name' column.")
        return None

    selected = states[states["name"].astype(str).str.lower() == state_name.lower()]
    if selected.empty:
        print(f"Warning: state '{state_name}' not found in state boundary file.")
        return None

    return selected.to_crs(epsg=3857)


def _build_static_figure(
    merged: pd.DataFrame,
    crop: str,
    basemap_style: str,
    state_name: str,
    state_boundary_cache: Path,
) -> tuple[plt.Figure, plt.Axes]:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(7.8, 7.0))
    style = _get_style_config(basemap_style)

    gdf = gpd.GeoDataFrame(
        merged.copy(),
        geometry=gpd.points_from_xy(merged["lon"], merged["lat"]),
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    minx, miny, maxx, maxy = gdf.total_bounds
    pad_x = max((maxx - minx) * 0.08, 5000)
    pad_y = max((maxy - miny) * 0.08, 5000)
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    _add_basemap_with_fallback(ax, basemap_style)

    state_outline = _load_state_boundary(state_boundary_cache, state_name)
    if state_outline is not None:
        state_outline.boundary.plot(ax=ax, color="white", linewidth=3.2, zorder=3)
        state_outline.boundary.plot(ax=ax, color="#111111", linewidth=1.4, zorder=4)

    for split_name in ["train", "val", "test"]:
        split_points = gdf[gdf["split"] == split_name]
        ax.scatter(
            split_points.geometry.x,
            split_points.geometry.y,
            s=style["marker_size"],
            marker=SPLIT_MARKERS[split_name],
            alpha=style["marker_alpha"],
            color=SPLIT_COLORS[split_name],
            edgecolors=style["marker_edge"],
            linewidths=0.6,
            zorder=5,
        )

    handles = [
        Line2D(
            [0],
            [0],
            marker=SPLIT_MARKERS[split_name],
            color="none",
            markerfacecolor=SPLIT_COLORS[split_name],
            markeredgecolor=style["marker_edge"],
            markeredgewidth=0.8,
            markersize=8,
            label=f"{split_name.title()} (n={len(gdf[gdf['split'] == split_name])})",
        )
        for split_name in ["train", "val", "test"]
    ]
    legend = ax.legend(
        handles=handles,
        loc="lower left",
        frameon=True,
        title="Dataset Split",
        fancybox=True,
    )
    legend.get_frame().set_facecolor(style["legend_face"])
    legend.get_frame().set_edgecolor("#888888")
    legend.get_frame().set_linewidth(0.7)

    ax.annotate(
        "N",
        xy=(0.95, 0.94),
        xytext=(0.95, 0.84),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#111111",
        bbox={"facecolor": (1, 1, 1, 0.78), "edgecolor": "none", "boxstyle": "round,pad=0.18"},
        arrowprops={"arrowstyle": "-|>", "lw": 1.2, "color": "#111111"},
        zorder=7,
    )

    ax.set_axis_off()
    fig.tight_layout()
    return fig, ax


def save_pdf_map(
    merged: pd.DataFrame,
    crop: str,
    output_pdf: Path,
    basemap_style: str,
    state_name: str,
    state_boundary_cache: Path,
) -> None:
    fig, _ = _build_static_figure(
        merged,
        crop,
        basemap_style,
        state_name=state_name,
        state_boundary_cache=state_boundary_cache,
    )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def save_png_map(
    merged: pd.DataFrame,
    crop: str,
    output_png: Path,
    dpi: int,
    basemap_style: str,
    state_name: str,
    state_boundary_cache: Path,
) -> None:
    fig, _ = _build_static_figure(
        merged,
        crop,
        basemap_style,
        state_name=state_name,
        state_boundary_cache=state_boundary_cache,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate folium maps for train/val/test field splits for corn and soybean."
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("processed_data/weekly_24/processed_data_weekly_24"),
        help="Directory containing train_corn.txt, val_corn.txt, test_corn.txt and soybean equivalents.",
    )
    parser.add_argument(
        "--raw-yield-dir",
        type=Path,
        default=Path("raw_yield"),
        help="Directory containing Crop_Classification_*.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("maps"),
        help="Output directory for generated maps.",
    )
    parser.add_argument(
        "--format",
        choices=["pdf", "png", "html", "both", "all"],
        default="pdf",
        help="Output format. Default is pdf.",
    )
    parser.add_argument(
        "--png-dpi",
        type=int,
        default=600,
        help="DPI for PNG output (journal-quality default: 600).",
    )
    parser.add_argument(
        "--basemap-style",
        choices=["light", "satellite", "terrain"],
        default="terrain",
        help="Basemap style for static PDF/PNG outputs.",
    )
    parser.add_argument(
        "--journal",
        action="store_true",
        help="Save static outputs with '_journal' suffix for publication-ready variants.",
    )
    parser.add_argument(
        "--state-name",
        type=str,
        default="Iowa",
        help="US state boundary to overlay for clearer geographic context.",
    )
    parser.add_argument(
        "--state-boundary-cache",
        type=Path,
        default=Path("maps/state_boundaries_us.geojson"),
        help="Local cache path for downloaded US states GeoJSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    centroids = load_crop_centroids(args.raw_yield_dir)

    for crop in ["corn", "soybean"]:
        merged = build_crop_map(crop=crop, split_dir=args.split_dir, centroids=centroids)
        style_suffix = "" if args.basemap_style == "light" else f"_{args.basemap_style}"
        journal_suffix = "_journal" if args.journal else ""

        if args.format in {"html", "both", "all"}:
            html_path = args.output_dir / f"{crop}_train_val_test_map.html"
            save_folium_map(merged, html_path)
            print(f"Saved: {html_path}")

        if args.format in {"pdf", "both", "all"}:
            pdf_path = args.output_dir / f"{crop}_train_val_test_map{style_suffix}{journal_suffix}.pdf"
            save_pdf_map(
                merged,
                crop,
                pdf_path,
                basemap_style=args.basemap_style,
                state_name=args.state_name,
                state_boundary_cache=args.state_boundary_cache,
            )
            print(f"Saved: {pdf_path}")

        if args.format in {"png", "all"}:
            png_path = args.output_dir / f"{crop}_train_val_test_map{style_suffix}{journal_suffix}.png"
            save_png_map(
                merged,
                crop,
                png_path,
                dpi=args.png_dpi,
                basemap_style=args.basemap_style,
                state_name=args.state_name,
                state_boundary_cache=args.state_boundary_cache,
            )
            print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
