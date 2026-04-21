from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import argparse
import csv
import os
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import opendssdirect as dss


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER_PATH = ROOT / "opendss" / "ieee123_base" / "IEEE123Master.dss"
DEFAULT_DER_OVERLAY_PATH = ROOT / "opendss" / "DER_Assets.dss"
DEFAULT_BUS_COORDS_PATH = ROOT / "opendss" / "ieee123_base" / "BusCoords.dat"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "topology_validation"

KEY_BUSES = ["13", "18", "35", "48", "60", "83", "94", "108", "114"]
CURATED_PV_BUSES = {"35", "60", "83"}
CURATED_BESS_BUSES = {"48", "108"}

NORMAL_NODE_COLOR = "#DDE3E7"
SELECTED_NODE_COLOR = "#3B73A5"
PV_NODE_COLOR = "#5F9F68"
BESS_NODE_COLOR = "#7759A6"
EDGE_COLOR = "#9AA5AF"
BACKGROUND_COLOR = "#FBFAF7"
TEXT_COLOR = "#24313B"


@dataclass(frozen=True, slots=True)
class TopologyEdge:
    line_name: str
    bus1: str
    bus2: str
    is_switch: bool


@dataclass(slots=True)
class TopologyData:
    edges: list[TopologyEdge]
    pv_by_bus: dict[str, list[str]]
    storage_by_bus: dict[str, list[str]]
    loads_by_bus: dict[str, list[str]]


def normalize_bus_name(raw_bus: str) -> str:
    return raw_bus.strip().strip('"').split(".")[0].lower()


def bus_sort_key(bus: str) -> tuple[int, str]:
    match = re.match(r"(\d+)", bus)
    if match:
        return (int(match.group(1)), bus)
    return (10**9, bus)


def compile_model(master_path: Path, der_overlay_path: Path | None = None) -> None:
    original_cwd = Path.cwd()
    dss.Basic.ClearAll()
    try:
        os.chdir(master_path.parent)
        dss.Text.Command(f"compile [{master_path.name}]")
        if der_overlay_path is not None and der_overlay_path.exists():
            dss.Text.Command(f"redirect [{der_overlay_path.resolve()}]")
        dss.Text.Command("solve")
    finally:
        os.chdir(original_cwd)


def extract_asset_buses(collection_name: str) -> dict[str, list[str]]:
    assets_by_bus: dict[str, list[str]] = defaultdict(list)
    if collection_name == "pv":
        names = dss.PVsystems.AllNames()
        setter = dss.PVsystems.Name
    elif collection_name == "storage":
        names = dss.Storages.AllNames()
        setter = dss.Storages.Name
    elif collection_name == "load":
        names = dss.Loads.AllNames()
        setter = dss.Loads.Name
    else:
        raise ValueError(f"Unsupported collection: {collection_name}")

    for name in names:
        if not name or name.lower() == "none":
            continue
        setter(name)
        bus = normalize_bus_name(dss.CktElement.BusNames()[0])
        assets_by_bus[bus].append(name)

    return {bus: sorted(names, key=str.lower) for bus, names in assets_by_bus.items()}


def extract_topology(master_path: Path, der_overlay_path: Path | None = None) -> TopologyData:
    compile_model(master_path=master_path, der_overlay_path=der_overlay_path)

    edges: list[TopologyEdge] = []
    for line_name in dss.Lines.AllNames():
        if not line_name or line_name.lower() == "none":
            continue
        dss.Lines.Name(line_name)
        bus1 = normalize_bus_name(dss.Lines.Bus1())
        bus2 = normalize_bus_name(dss.Lines.Bus2())
        edges.append(
            TopologyEdge(
                line_name=line_name,
                bus1=bus1,
                bus2=bus2,
                is_switch=line_name.lower().startswith("sw"),
            )
        )

    return TopologyData(
        edges=edges,
        pv_by_bus=extract_asset_buses("pv"),
        storage_by_bus=extract_asset_buses("storage"),
        loads_by_bus=extract_asset_buses("load"),
    )


def build_graph(edges: list[TopologyEdge]) -> nx.Graph:
    graph = nx.Graph()
    for edge in edges:
        if graph.has_edge(edge.bus1, edge.bus2):
            graph[edge.bus1][edge.bus2]["line_names"].append(edge.line_name)
            graph[edge.bus1][edge.bus2]["switch_names"].extend([edge.line_name] if edge.is_switch else [])
        else:
            graph.add_edge(
                edge.bus1,
                edge.bus2,
                line_names=[edge.line_name],
                switch_names=[edge.line_name] if edge.is_switch else [],
            )
    return graph


def load_bus_coordinates(bus_coords_path: Path) -> dict[str, tuple[float, float]]:
    coordinates: dict[str, tuple[float, float]] = {}
    if not bus_coords_path.exists():
        return coordinates

    for raw_line in bus_coords_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        bus, x_coord, y_coord = parts[0], parts[1], parts[2]
        coordinates[normalize_bus_name(bus)] = (float(x_coord), float(y_coord))
    return coordinates


def resolve_positions(graph: nx.Graph, bus_coords_path: Path) -> dict[str, tuple[float, float]]:
    coordinates = load_bus_coordinates(bus_coords_path)
    if not coordinates:
        return nx.spring_layout(graph, seed=42)

    positions: dict[str, tuple[float, float]] = {}
    for node in graph.nodes:
        if node in coordinates:
            positions[node] = coordinates[node]

    missing_nodes = [node for node in graph.nodes if node not in positions]
    if not missing_nodes:
        return positions

    if positions:
        x_values = [coord[0] for coord in positions.values()]
        y_values = [coord[1] for coord in positions.values()]
        x_mid = sum(x_values) / len(x_values)
        y_mid = sum(y_values) / len(y_values)
    else:
        x_mid = 0.0
        y_mid = 0.0

    for node in missing_nodes:
        alias = node.replace("_open", "")
        if alias in coordinates:
            positions[node] = (coordinates[alias][0] + 80.0, coordinates[alias][1] + 80.0)

    still_missing = [node for node in graph.nodes if node not in positions]
    if still_missing:
        fallback_layout = nx.spring_layout(graph.subgraph(still_missing), seed=42)
        for node, (x_coord, y_coord) in fallback_layout.items():
            positions[node] = (x_mid + x_coord * 200.0, y_mid + y_coord * 200.0)

    return positions


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor(BACKGROUND_COLOR)
    ax.set_axis_off()
    ax.set_aspect("equal")


def plot_full_graph(graph: nx.Graph, positions: dict[str, tuple[float, float]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 10.0))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    style_axes(ax)

    nx.draw_networkx_edges(graph, pos=positions, ax=ax, edge_color=EDGE_COLOR, width=0.85, alpha=0.78)
    nx.draw_networkx_nodes(
        graph,
        pos=positions,
        ax=ax,
        node_color=NORMAL_NODE_COLOR,
        node_size=20,
        edgecolors="#EEF1F3",
        linewidths=0.2,
    )

    ax.set_title(
        "IEEE 123-Bus OpenDSS Topology (Raw Line Connectivity)",
        fontsize=15,
        fontweight="bold",
        color=TEXT_COLOR,
        pad=14,
    )
    ax.text(
        0.5,
        0.98,
        "Full ground-truth structure from IEEE123Master.dss with no topology simplification.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#69757F",
    )
    ax.invert_yaxis()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def plot_highlighted_graph(
    graph: nx.Graph,
    positions: dict[str, tuple[float, float]],
    output_path: Path,
    key_buses: list[str],
    pv_buses: set[str],
    storage_buses: set[str],
) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 10.0))
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    style_axes(ax)

    nx.draw_networkx_edges(graph, pos=positions, ax=ax, edge_color=EDGE_COLOR, width=0.9, alpha=0.72)
    nx.draw_networkx_nodes(
        graph,
        pos=positions,
        ax=ax,
        node_color=NORMAL_NODE_COLOR,
        node_size=24,
        edgecolors="#F3F5F7",
        linewidths=0.25,
    )

    selected_only = sorted((set(key_buses) - pv_buses - storage_buses) & set(graph.nodes), key=bus_sort_key)
    pv_nodes = sorted(pv_buses & set(graph.nodes), key=bus_sort_key)
    storage_nodes = sorted(storage_buses & set(graph.nodes), key=bus_sort_key)

    if selected_only:
        nx.draw_networkx_nodes(
            graph,
            pos=positions,
            nodelist=selected_only,
            ax=ax,
            node_color=SELECTED_NODE_COLOR,
            node_size=135,
            edgecolors="#F8FAFB",
            linewidths=1.1,
        )
    if pv_nodes:
        nx.draw_networkx_nodes(
            graph,
            pos=positions,
            nodelist=pv_nodes,
            ax=ax,
            node_color=PV_NODE_COLOR,
            node_size=155,
            edgecolors="#F8FAFB",
            linewidths=1.2,
        )
    if storage_nodes:
        nx.draw_networkx_nodes(
            graph,
            pos=positions,
            nodelist=storage_nodes,
            ax=ax,
            node_color=BESS_NODE_COLOR,
            node_size=155,
            edgecolors="#F8FAFB",
            linewidths=1.2,
        )

    labels = {bus: bus.upper() for bus in selected_only + pv_nodes + storage_nodes}
    nx.draw_networkx_labels(
        graph,
        pos=positions,
        labels=labels,
        font_size=8,
        font_weight="bold",
        font_color=TEXT_COLOR,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "#FFFFFF", "edgecolor": "none", "alpha": 0.9},
        ax=ax,
    )

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=NORMAL_NODE_COLOR, markeredgecolor="#DDE2E6", markersize=7, label="Normal bus"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SELECTED_NODE_COLOR, markeredgecolor="white", markersize=8, label="Highlighted figure bus"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PV_NODE_COLOR, markeredgecolor="white", markersize=8, label="PV bus"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BESS_NODE_COLOR, markeredgecolor="white", markersize=8, label="BESS bus"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        facecolor="#FFFFFF",
        edgecolor="#D7DEE4",
        fontsize=8.5,
    )
    ax.set_title(
        "IEEE 123-Bus Topology with Curated-Diagram Validation Buses Highlighted",
        fontsize=15,
        fontweight="bold",
        color=TEXT_COLOR,
        pad=14,
    )
    ax.text(
        0.5,
        0.98,
        "Blue nodes come from the research figure; green and purple nodes show DER attachment buses.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="#69757F",
    )
    ax.invert_yaxis()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def save_edges_csv(edges: list[TopologyEdge], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["line_name", "bus1", "bus2", "is_switch"])
        writer.writeheader()
        for edge in sorted(edges, key=lambda item: item.line_name.lower()):
            writer.writerow(
                {
                    "line_name": edge.line_name,
                    "bus1": edge.bus1,
                    "bus2": edge.bus2,
                    "is_switch": str(edge.is_switch).lower(),
                }
            )


def print_key_adjacency(
    graph: nx.Graph,
    output_path: Path,
    key_buses: list[str],
    pv_by_bus: dict[str, list[str]],
    storage_by_bus: dict[str, list[str]],
    loads_by_bus: dict[str, list[str]],
) -> None:
    lines = [
        "Key-bus adjacency extracted from IEEE123Master.dss",
        "Ground truth is based on all Line objects after phase suffix stripping.",
        "",
    ]

    for bus in key_buses:
        if not graph.has_node(bus):
            lines.append(f"Bus {bus} -> [missing from extracted graph]")
            continue
        neighbors = sorted(graph.neighbors(bus), key=bus_sort_key)
        neighbor_text = ", ".join(neighbors)
        pv_assets = ", ".join(pv_by_bus.get(bus, [])) or "none"
        storage_assets = ", ".join(storage_by_bus.get(bus, [])) or "none"
        load_count = len(loads_by_bus.get(bus, []))
        lines.append(f"Bus {bus} -> [{neighbor_text}]")
        lines.append(f"  PV: {pv_assets}")
        lines.append(f"  Storage: {storage_assets}")
        lines.append(f"  Loads attached: {load_count}")
        lines.append("")

    text = "\n".join(lines).rstrip() + "\n"
    output_path.write_text(text, encoding="utf-8")
    print(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and visualize the raw IEEE 123-bus OpenDSS topology for validation.")
    parser.add_argument("--master-path", type=Path, default=DEFAULT_MASTER_PATH, help="Path to IEEE123Master.dss")
    parser.add_argument("--der-overlay-path", type=Path, default=DEFAULT_DER_OVERLAY_PATH, help="Optional DER overlay used only to attach PV/storage metadata")
    parser.add_argument("--bus-coords-path", type=Path, default=DEFAULT_BUS_COORDS_PATH, help="Path to BusCoords.dat for plotting")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated validation artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    der_overlay = args.der_overlay_path if args.der_overlay_path.exists() else None
    topology = extract_topology(master_path=args.master_path.resolve(), der_overlay_path=der_overlay.resolve() if der_overlay else None)
    graph = build_graph(topology.edges)
    positions = resolve_positions(graph, args.bus_coords_path.resolve())
    pv_buses = set(topology.pv_by_bus) or CURATED_PV_BUSES
    storage_buses = set(topology.storage_by_bus) or CURATED_BESS_BUSES

    plot_full_graph(graph, positions, output_dir / "topology_full.png")
    plot_highlighted_graph(
        graph,
        positions,
        output_dir / "topology_highlighted.png",
        key_buses=KEY_BUSES,
        pv_buses=pv_buses,
        storage_buses=storage_buses,
    )
    save_edges_csv(topology.edges, output_dir / "topology_edges.csv")
    print_key_adjacency(
        graph,
        output_dir / "adjacency_key_buses.txt",
        key_buses=KEY_BUSES,
        pv_by_bus=topology.pv_by_bus,
        storage_by_bus=topology.storage_by_bus,
        loads_by_bus=topology.loads_by_bus,
    )

    print(f"Saved validation outputs to: {output_dir}")
    print(f"  topology_full.png")
    print(f"  topology_highlighted.png")
    print(f"  topology_edges.csv")
    print(f"  adjacency_key_buses.txt")


if __name__ == "__main__":
    main()


# Comparison guidance:
# - Compare topology_highlighted.png against outputs/paper_figures/der_topology_full.png.
# - The OpenDSS outputs here are raw ground truth from every Line object in IEEE123Master.dss,
#   with phase suffixes stripped but no branch simplification applied.
# - The curated der_topology_full figure is intentionally a simplified abstraction that keeps
#   only representative buses, DER placements, switches, and control assets for communication.
# - Structural validation should therefore focus on whether the highlighted buses and DER anchor
#   points sit on the correct feeder regions and branch families, not whether the curated figure
#   reproduces every intermediate OpenDSS bus as a direct one-hop connection.
