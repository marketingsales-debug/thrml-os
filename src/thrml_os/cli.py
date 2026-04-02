"""THRML-OS Command Line Interface."""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="thrml-os")
def main() -> None:
    """THRML-OS: Thermodynamic Computing Operating System.
    
    Runtime for probabilistic graphical models on Extropic hardware.
    """
    pass


@main.command()
def devices() -> None:
    """List available compute devices (PCUs)."""
    from thrml_os.models.device import detect_devices
    
    devices = detect_devices()
    
    table = Table(title="Available Compute Devices")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Type", style="magenta")
    table.add_column("Status", style="yellow")
    table.add_column("Compute Units")
    table.add_column("Max Nodes")
    
    for dev in devices:
        table.add_row(
            dev.id,
            dev.name,
            dev.device_type.value,
            dev.status.value,
            str(dev.capabilities.compute_units),
            f"{dev.capabilities.max_nodes:,}",
        )
    
    console.print(table)
    console.print(f"\n[dim]Total: {len(devices)} device(s)[/dim]")


@main.command()
@click.argument("model_file", type=click.Path(exists=True))
@click.option("--samples", "-n", default=1000, help="Number of samples to collect")
@click.option("--warmup", "-w", default=100, help="Warmup/burn-in sweeps")
@click.option("--priority", "-p", default="normal", 
              type=click.Choice(["realtime", "high", "normal", "low", "batch"]))
@click.option("--tag", "-t", multiple=True, help="Tags for the job")
def submit(
    model_file: str,
    samples: int,
    warmup: int,
    priority: str,
    tag: tuple,
) -> None:
    """Submit a sampling job.
    
    MODEL_FILE: Python file defining the THRML model.
    The file should define a variable named 'model'.
    """
    import sys
    import importlib.util
    
    # Load model from file
    with console.status("[bold green]Loading model..."):
        spec = importlib.util.spec_from_file_location("model_module", model_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules["model_module"] = module
        spec.loader.exec_module(module)
        
        if not hasattr(module, 'model'):
            console.print("[red]Error: Model file must define a 'model' variable[/red]")
            raise SystemExit(1)
        
        model = module.model
    
    # Create job
    from thrml_os.models.job import SamplingJob, JobPriority
    
    job = SamplingJob(
        model=model,
        n_samples=samples,
        n_warmup=warmup,
        priority=JobPriority[priority.upper()],
        tags=list(tag),
    )
    
    # Submit to scheduler
    from thrml_os.scheduler import Scheduler
    
    scheduler = Scheduler()
    scheduler.start()
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Submitting job...", total=None)
            job_id = scheduler.submit(job)
            progress.update(task, description=f"Job submitted: {job_id}")
        
        console.print(Panel(
            f"[green]✓[/green] Job submitted successfully\n\n"
            f"[bold]Job ID:[/bold] {job_id}\n"
            f"[bold]Samples:[/bold] {samples}\n"
            f"[bold]Warmup:[/bold] {warmup}\n"
            f"[bold]Priority:[/bold] {priority}\n\n"
            f"[dim]Use 'thrml-os status {job_id}' to check progress[/dim]",
            title="Job Submitted",
            border_style="green",
        ))
        
        # Wait for completion if small job
        if samples <= 100:
            with console.status("[bold yellow]Running..."):
                import time
                while True:
                    job = scheduler.get_job(job_id)
                    if job and job.is_terminal:
                        break
                    time.sleep(0.1)
            
            if job.status.value == "completed":
                console.print(f"[green]✓ Job completed with {job.samples_collected} samples[/green]")
            else:
                console.print(f"[red]✗ Job {job.status.value}: {job.error_message}[/red]")
    finally:
        scheduler.stop(wait=False)


@main.command()
@click.argument("job_id")
def status(job_id: str) -> None:
    """Check status of a job.
    
    JOB_ID: The job identifier returned by submit.
    """
    # For now, show a mock status since we don't have persistence
    console.print(Panel(
        f"[bold]Job ID:[/bold] {job_id}\n"
        f"[bold]Status:[/bold] [yellow]unknown[/yellow]\n\n"
        f"[dim]Note: Job persistence not yet implemented.\n"
        f"Run jobs interactively with 'thrml-os run' instead.[/dim]",
        title="Job Status",
    ))


@main.command()
@click.argument("job_id")
@click.option("--follow", "-f", is_flag=True, help="Follow logs in real-time")
def logs(job_id: str, follow: bool) -> None:
    """View logs for a job.
    
    JOB_ID: The job identifier.
    """
    console.print(f"[dim]Logs for job {job_id}:[/dim]")
    console.print("[yellow]Log streaming not yet implemented[/yellow]")


@main.command()
@click.argument("job_id")
def cancel(job_id: str) -> None:
    """Cancel a running or queued job.
    
    JOB_ID: The job identifier to cancel.
    """
    console.print(f"[yellow]Cancelling job {job_id}...[/yellow]")
    console.print("[dim]Note: Requires active scheduler connection[/dim]")


@main.command()
def stats() -> None:
    """Show system statistics."""
    from thrml_os.models.device import detect_devices
    from thrml_os.hal.registry import get_default_backend
    
    devices = detect_devices()
    
    # Try to get backend info
    try:
        backend = get_default_backend()
        backend_info = f"{backend.backend_type.value}"
        caps = backend.capabilities
    except Exception:
        backend_info = "unknown"
        caps = None
    
    console.print(Panel(
        f"[bold]THRML-OS Statistics[/bold]\n\n"
        f"[bold]Backend:[/bold] {backend_info}\n"
        f"[bold]Devices:[/bold] {len(devices)}\n"
        f"[bold]Max Graph Nodes:[/bold] {caps.max_graph_nodes:,}" if caps else "" + "\n"
        f"[bold]Est. Samples/sec:[/bold] {caps.estimated_samples_per_second:,.0f}" if caps and caps.estimated_samples_per_second else "",
        title="System Stats",
        border_style="blue",
    ))


@main.command()
@click.option("--samples", "-n", default=100, help="Number of samples")
@click.option("--nodes", default=10, help="Number of nodes in test graph")
def benchmark(samples: int, nodes: int) -> None:
    """Run a benchmark with a simple Ising model."""
    import time
    import jax
    import jax.numpy as jnp
    
    console.print(f"[bold]Running benchmark...[/bold]")
    console.print(f"  Nodes: {nodes}")
    console.print(f"  Samples: {samples}")
    console.print()
    
    # Use simulator for benchmarking
    from thrml_os.hal.simulator import SimulatorBackend
    from thrml_os.models.schedule import SamplingSchedule
    from thrml_os.hal.backend import ExecutionContext
    
    backend = SimulatorBackend(samples_per_second=100000)
    backend.initialize()
    
    # Create a mock model
    class MockModel:
        def __init__(self, n: int):
            self.nodes = list(range(n))
            self.edges = [(i, i+1) for i in range(n-1)]
    
    model = MockModel(nodes)
    compiled = backend.compile_model(model)
    
    schedule = SamplingSchedule(
        n_warmup=10,
        n_samples=samples,
        steps_per_sample=1,
    )
    
    key = jax.random.key(42)
    context = ExecutionContext(key=key)
    
    # Run benchmark
    start = time.time()
    total_samples = 0
    
    with Progress(console=console) as progress:
        task = progress.add_task("Sampling...", total=samples)
        
        for batch in backend.sample(compiled, schedule, None, context):
            total_samples += batch.n_samples
            progress.update(task, advance=batch.n_samples)
    
    elapsed = time.time() - start
    rate = total_samples / elapsed if elapsed > 0 else 0
    
    console.print()
    console.print(Panel(
        f"[green]✓ Benchmark complete[/green]\n\n"
        f"[bold]Samples:[/bold] {total_samples:,}\n"
        f"[bold]Time:[/bold] {elapsed:.3f}s\n"
        f"[bold]Rate:[/bold] {rate:,.0f} samples/sec",
        title="Benchmark Results",
        border_style="green",
    ))
    
    backend.shutdown()


if __name__ == "__main__":
    main()
