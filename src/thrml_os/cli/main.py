"""THRML-OS Command Line Interface."""

from __future__ import annotations

import click
from rich.console import Console
from rich.table import Table

from thrml_os.client import THRMLClient
from thrml_os.models.device import detect_devices

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="thrml-os")
def main() -> None:
    """THRML-OS: Thermodynamic Computing Operating System."""
    pass


@main.command()
def devices() -> None:
    """List available compute devices."""
    devs = detect_devices()

    table = Table(title="Probabilistic Compute Units (PCUs)")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Type", style="green")
    table.add_column("Status")
    table.add_column("Cores", justify="right")
    table.add_column("Max Nodes", justify="right")

    for d in devs:
        table.add_row(
            d.id,
            d.name,
            d.device_type.value,
            d.status.value,
            str(d.capabilities.compute_units),
            f"{d.capabilities.max_nodes:,}",
        )

    console.print(table)


@main.command()
def stats() -> None:
    """Show system statistics."""
    import jax

    table = Table(title="THRML-OS Statistics")
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="cyan", justify="right")

    devs = detect_devices()
    table.add_row("Devices", str(len(devs)))
    table.add_row("JAX Backend", jax.default_backend())
    table.add_row("JAX Devices", str(jax.device_count()))

    console.print(table)


@main.command()
@click.option("--nodes", "-n", default=10, help="Number of spin nodes")
@click.option("--samples", "-s", default=1000, help="Number of samples")
@click.option("--warmup", "-w", default=100, help="Warmup sweeps")
@click.option("--coupling", "-c", default=1.0, help="Coupling strength")
@click.option("--beta", "-b", default=1.0, help="Inverse temperature")
def demo(nodes: int, samples: int, warmup: int, coupling: float, beta: float) -> None:
    """Run a demo Ising chain sampling job."""
    import jax.numpy as jnp
    from thrml import SpinNode
    from thrml.models import IsingEBM

    console.print(f"\n[bold green]THRML-OS Demo[/bold green]")
    console.print(f"  Ising chain: {nodes} spins, coupling={coupling}, beta={beta}")
    console.print(f"  Samples: {samples}, warmup: {warmup}\n")

    # Build model
    spin_nodes = [SpinNode() for _ in range(nodes)]
    edges = [(spin_nodes[i], spin_nodes[i + 1]) for i in range(nodes - 1)]
    biases = jnp.zeros(nodes)
    weights = jnp.ones(len(edges)) * coupling

    model = IsingEBM(spin_nodes, edges, biases, weights, jnp.array(beta))

    # Submit job via client
    with THRMLClient() as client:
        with console.status("[bold yellow]Sampling...[/bold yellow]"):
            job = client.run_simple(
                model, n_samples=samples, n_warmup=warmup, steps_per_sample=2
            )

    result = job.samples

    # Show results
    console.print(f"[green]Done![/green] Collected {result.shape[0]} samples\n")

    mag = float(jnp.mean(jnp.abs(jnp.mean(result, axis=1))))
    all_aligned = float(
        100 * jnp.mean(jnp.all(result == result[:, 0:1], axis=1))
    )

    table = Table(title="Results")
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="cyan", justify="right")
    table.add_row("Samples", str(result.shape[0]))
    table.add_row("Nodes", str(result.shape[1]))
    table.add_row("|Magnetization|", f"{mag:.4f}")
    table.add_row("% All Aligned", f"{all_aligned:.1f}%")

    console.print(table)

    # Show first few samples
    console.print("\n[bold]First 5 samples:[/bold]")
    for i in range(min(5, result.shape[0])):
        spins = ["[green]+[/green]" if x > 0 else "[red]-[/red]" for x in result[i]]
        console.print(f"  {'  '.join(spins)}")


@main.command()
def info() -> None:
    """Show THRML-OS system info."""
    import jax
    import thrml

    console.print("[bold]THRML-OS v0.1.0[/bold]\n")
    console.print(f"  THRML version:  {thrml.__version__}")
    console.print(f"  JAX version:    {jax.__version__}")
    console.print(f"  JAX backend:    {jax.default_backend()}")
    console.print(f"  JAX devices:    {jax.device_count()}")

    devs = detect_devices()
    console.print(f"  PCUs detected:  {len(devs)}")
    for d in devs:
        console.print(f"    - {d.name} ({d.device_type.value})")


if __name__ == "__main__":
    main()
