#!/usr/bin/env python3
"""Launch the local CARLA server and then run a project Python script."""

from __future__ import annotations

import argparse
import ast
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_EDITOR = Path(
    "/home/rug/UnrealEngine_4.26/Engine/Binaries/Linux/UE4Editor"
)
DEFAULT_CARLA_PROJECT = Path(
    "/home/rug/carla/Unreal/CarlaUE4/CarlaUE4.uproject"
)
DEFAULT_LOG = PROJECT_DIR / "carla_server.log"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 2000
DEFAULT_MAP = "MyFigure8"


class LauncherError(RuntimeError):
    """An actionable launcher error."""


def positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def nonnegative_float(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Start this computer's source-built CARLA server, wait until it is "
            "ready, optionally load a map, and run one Python script from this "
            "directory. Put launcher options before SCRIPT."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--editor",
        type=Path,
        default=DEFAULT_EDITOR,
        help="UE4Editor executable",
    )
    parser.add_argument(
        "--carla-project",
        type=Path,
        default=DEFAULT_CARLA_PROJECT,
        help="CarlaUE4.uproject path",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used for the selected script",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="CARLA RPC host used by the readiness check",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=(
            "CARLA RPC port; the existing project scripts are hard-coded to "
            "127.0.0.1:2000"
        ),
    )
    parser.add_argument(
        "--map",
        default=DEFAULT_MAP,
        metavar="NAME",
        help=(
            "map to load; 'auto' reads CARLA_MAP_NAME from the selected script, "
            "and 'current' keeps the server's current map"
        ),
    )
    parser.add_argument(
        "--spectator-height",
        type=nonnegative_float,
        default=0.0,
        metavar="METERS",
        help=(
            "top-down spectator height above the highest road point; zero "
            "chooses a height from the map dimensions"
        ),
    )
    parser.add_argument(
        "--no-top-down-view",
        action="store_true",
        help="do not reposition the spectator after loading the map",
    )
    parser.add_argument(
        "--startup-timeout",
        type=positive_float,
        default=240.0,
        help="seconds to wait for CARLA's RPC server",
    )
    parser.add_argument(
        "--script-timeout",
        type=nonnegative_float,
        default=0.0,
        help="maximum script runtime in seconds; zero disables the limit",
    )
    parser.add_argument(
        "--quality-level",
        choices=("Low", "Epic"),
        default="Low",
        help="CARLA rendering quality",
    )
    parser.add_argument(
        "--offscreen",
        action="store_true",
        help="run CARLA without an on-screen spectator window",
    )
    parser.add_argument(
        "--opengl",
        action="store_true",
        help="use OpenGL instead of Vulkan",
    )
    parser.add_argument(
        "--no-sound",
        action="store_true",
        help="disable CARLA audio",
    )
    parser.add_argument(
        "--keep-carla",
        action="store_true",
        help="leave a CARLA process started by this launcher running afterward",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=DEFAULT_LOG,
        help="file for CARLA stdout and stderr",
    )
    parser.add_argument(
        "--carla-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="extra CARLA argument; repeat as needed (use --carla-arg=-flag)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print commands without starting anything",
    )
    parser.add_argument(
        "--list-scripts",
        action="store_true",
        help="list scripts in this directory that create a carla.Client",
    )
    parser.add_argument("script", nargs="?", metavar="SCRIPT")
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        metavar="SCRIPT_ARG",
        help="arguments forwarded unchanged to SCRIPT",
    )
    return parser


def list_client_scripts() -> list[Path]:
    scripts: list[Path] = []
    for path in sorted(PROJECT_DIR.glob("*.py")):
        if path.name == Path(__file__).name:
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if "carla.Client(" in source:
            scripts.append(path)
    return scripts


def resolve_project_script(value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = PROJECT_DIR / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise LauncherError(f"script does not exist: {candidate}") from exc

    if resolved.parent != PROJECT_DIR:
        raise LauncherError("SCRIPT must be a Python file directly inside figure8_project")
    if resolved.suffix != ".py":
        raise LauncherError(f"SCRIPT is not a Python file: {resolved.name}")
    if resolved == Path(__file__).resolve():
        raise LauncherError("the launcher cannot launch itself")
    return resolved


def resolve_executable(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() or path.parent != Path("."):
        resolved = path.resolve()
        if not resolved.is_file() or not os.access(resolved, os.X_OK):
            raise LauncherError(f"Python executable is missing or not executable: {path}")
        return resolved

    found = shutil.which(value)
    if not found:
        raise LauncherError(f"Python executable was not found on PATH: {value}")
    return Path(found).resolve()


def validate_server_paths(editor: Path, project: Path) -> tuple[Path, Path]:
    editor = editor.expanduser().resolve()
    project = project.expanduser().resolve()
    if not editor.is_file() or not os.access(editor, os.X_OK):
        raise LauncherError(f"UE4Editor is missing or not executable: {editor}")
    if not project.is_file():
        raise LauncherError(f"CARLA project file is missing: {project}")
    return editor, project


def infer_map_from_script(script: Path) -> str | None:
    """Return a literal CARLA_MAP_NAME without importing or running the script."""
    try:
        tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    except (OSError, SyntaxError):
        return None

    for node in tree.body:
        value: ast.expr | None = None
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            value = node.value
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            value = node.value
            targets = [node.target]
        if any(isinstance(target, ast.Name) and target.id == "CARLA_MAP_NAME" for target in targets):
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return value.value

    if script.name.startswith("8ring_") or script.name.startswith("randommap_"):
        return "8ring"
    return None


def requested_map(map_option: str, script: Path) -> str | None:
    if map_option.lower() in {"current", "none"}:
        return None
    if map_option.lower() == "auto":
        return infer_map_from_script(script)
    return map_option


def build_carla_command(args: argparse.Namespace, editor: Path, project: Path) -> list[str]:
    command = [
        str(editor),
        str(project),
        "-game",
        "-opengl" if args.opengl else "-vulkan",
        f"-carla-rpc-port={args.port}",
        f"-quality-level={args.quality_level}",
    ]
    if args.offscreen:
        command.append("-RenderOffScreen")
    if args.no_sound:
        command.append("-nosound")
    command.extend(args.carla_arg)
    return command


def import_carla_api() -> ModuleType:
    try:
        import carla  # type: ignore
    except ImportError as exc:
        raise LauncherError(
            "the selected Python environment cannot import carla; use --python "
            "for the project script and install the CARLA API for this launcher"
        ) from exc
    return carla


def tcp_port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def try_connect(carla: ModuleType, host: str, port: int, timeout: float):
    try:
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        client.get_server_version()
        return client
    except Exception:
        return None


def tail_log(path: Path, line_count: int = 30) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-line_count:])


def wait_for_server(
    carla: ModuleType,
    host: str,
    port: int,
    process: subprocess.Popen[bytes],
    timeout: float,
    log_path: Path,
):
    deadline = time.monotonic() + timeout
    last_notice = 0.0
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            details = tail_log(log_path)
            suffix = f"\n\nLast CARLA log lines:\n{details}" if details else ""
            raise LauncherError(
                f"CARLA exited before becoming ready (exit code {return_code}).{suffix}"
            )

        client = try_connect(carla, host, port, timeout=2.0)
        if client is not None:
            return client

        now = time.monotonic()
        if now - last_notice >= 10.0:
            remaining = max(0, int(deadline - now))
            print(f"Waiting for CARLA on {host}:{port} ({remaining}s remaining)...")
            last_notice = now
        time.sleep(1.0)

    details = tail_log(log_path)
    suffix = f"\n\nLast CARLA log lines:\n{details}" if details else ""
    raise LauncherError(
        f"CARLA did not become ready within {timeout:.0f} seconds.{suffix}"
    )


def load_map(client, map_name: str):
    client.set_timeout(90.0)
    current_world = client.get_world()
    current_name = current_world.get_map().name.rsplit("/", 1)[-1]
    if current_name == map_name:
        print(f"CARLA map is already '{map_name}'.")
        return current_world

    print(f"Loading CARLA map '{map_name}' (current: '{current_name}')...")
    try:
        world = client.load_world(map_name)
    except Exception as exc:
        raise LauncherError(f"could not load CARLA map '{map_name}': {exc}") from exc
    loaded_name = world.get_map().name.rsplit("/", 1)[-1]
    if loaded_name != map_name:
        raise LauncherError(
            f"CARLA returned map '{loaded_name}' after requesting '{map_name}'"
        )
    print(f"CARLA map '{map_name}' is ready.")
    return world


def set_top_down_spectator(carla: ModuleType, world, requested_height: float) -> None:
    """Place the spectator above the center of the map's road network."""
    carla_map = world.get_map()
    waypoints = carla_map.generate_waypoints(5.0)
    locations = [waypoint.transform.location for waypoint in waypoints]
    if not locations:
        locations = [transform.location for transform in carla_map.get_spawn_points()]
    if not locations:
        raise LauncherError("cannot center the spectator: the loaded map has no road points")

    min_x = min(location.x for location in locations)
    max_x = max(location.x for location in locations)
    min_y = min(location.y for location in locations)
    max_y = max(location.y for location in locations)
    max_z = max(location.z for location in locations)
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    map_span = max(max_x - min_x, max_y - min_y)
    height = requested_height or max(150.0, map_span * 1.1)

    desired = carla.Transform(
        carla.Location(x=center_x, y=center_y, z=max_z + height),
        carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
    )
    spectator = world.get_spectator()
    horizontal_tolerance = max(5.0, map_span * 0.01)

    transform = None
    for _ in range(3):
        spectator.set_transform(desired)
        try:
            if world.get_settings().synchronous_mode:
                world.tick()
            else:
                world.wait_for_tick(5.0)
        except RuntimeError:
            time.sleep(0.5)

        transform = spectator.get_transform()
        pitch_error = abs(((transform.rotation.pitch + 90.0 + 180.0) % 360.0) - 180.0)
        if (
            abs(transform.location.x - center_x) <= horizontal_tolerance
            and abs(transform.location.y - center_y) <= horizontal_tolerance
            and abs(transform.location.z - (max_z + height)) <= 2.0
            and pitch_error <= 2.0
        ):
            break
    else:
        assert transform is not None
        raise LauncherError(
            "CARLA did not apply the requested top-down spectator transform; "
            f"actual=({transform.location.x:.1f}, {transform.location.y:.1f}, "
            f"{transform.location.z:.1f}, pitch={transform.rotation.pitch:.1f})"
        )

    print(
        "Spectator top-down view: "
        f"center=({transform.location.x:.1f}, {transform.location.y:.1f}), "
        f"z={transform.location.z:.1f}m, pitch={transform.rotation.pitch:.1f}deg, "
        f"road span={map_span:.1f}m"
    )


def stop_process_group(process: subprocess.Popen, label: str, grace_seconds: float = 15.0) -> None:
    if process.poll() is not None:
        return
    print(f"Stopping {label} (PID {process.pid})...")
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=grace_seconds)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        print(f"{label} did not stop after {grace_seconds:.0f}s; sending SIGKILL.")
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=5.0)


def run_script(command: list[str], timeout: float) -> int:
    print(f"Running project script: {shlex.join(command)}")
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        cwd=PROJECT_DIR,
        env=environment,
        start_new_session=True,
    )
    try:
        return process.wait(timeout=timeout if timeout > 0 else None)
    except subprocess.TimeoutExpired:
        print(f"Project script exceeded its {timeout:.0f}s timeout.", file=sys.stderr)
        stop_process_group(process, "project script", grace_seconds=10.0)
        return 124
    except KeyboardInterrupt:
        print("Interrupted; stopping the project script...", file=sys.stderr)
        stop_process_group(process, "project script", grace_seconds=10.0)
        return 130


def main(argv: list[str] | None = None) -> int:
    # Keep launcher status messages ordered even when output is piped to a log.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_scripts:
        scripts = list_client_scripts()
        print("Scripts that create a CARLA client:")
        for path in scripts:
            print(f"  {path.name}")
        return 0

    if not args.script:
        parser.error("SCRIPT is required unless --list-scripts is used")

    try:
        script = resolve_project_script(args.script)
        editor, project = validate_server_paths(args.editor, args.carla_project)
        python = resolve_executable(args.python)
        map_name = requested_map(args.map, script)
        script_args = list(args.script_args)
        if script_args and script_args[0] == "--":
            script_args.pop(0)

        if args.port != DEFAULT_PORT and "carla.Client(" in script.read_text(encoding="utf-8"):
            print(
                "Warning: project scripts currently connect to 127.0.0.1:2000; "
                f"the requested launcher port is {args.port}.",
                file=sys.stderr,
            )

        carla_command = build_carla_command(args, editor, project)
        script_command = [str(python), "-u", str(script), *script_args]

        print(f"Project directory: {PROJECT_DIR}")
        print(f"CARLA command: {shlex.join(carla_command)}")
        print(f"Map: {map_name if map_name else 'keep current map'}")
        if args.no_top_down_view:
            print("Spectator: keep current view")
        elif args.spectator_height > 0:
            print(f"Spectator: top-down at map center, height {args.spectator_height:.1f}m")
        else:
            print("Spectator: top-down at map center, automatic height")
        print(f"Script command: {shlex.join(script_command)}")
        if args.dry_run:
            print("Dry run complete; no process was started.")
            return 0

        if not args.offscreen and not os.environ.get("DISPLAY"):
            raise LauncherError(
                "DISPLAY is not set; add --offscreen or run from a graphical session"
            )

        carla = import_carla_api()
        existing_client = try_connect(carla, args.host, args.port, timeout=2.0)
        if existing_client is None and tcp_port_is_open(args.host, args.port):
            raise LauncherError(
                f"TCP port {args.port} is occupied, but it is not responding as CARLA"
            )

        carla_process: subprocess.Popen[bytes] | None = None
        log_handle = None
        try:
            if existing_client is not None:
                client = existing_client
                print(
                    f"CARLA is already running on {args.host}:{args.port}; reusing it."
                )
            else:
                log_path = args.log.expanduser().resolve()
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_handle = log_path.open("w", encoding="utf-8")
                print(f"Starting CARLA; server log: {log_path}")
                carla_process = subprocess.Popen(
                    carla_command,
                    cwd=project.parent,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                client = wait_for_server(
                    carla,
                    args.host,
                    args.port,
                    carla_process,
                    args.startup_timeout,
                    log_path,
                )
                print(
                    "CARLA is ready: "
                    f"server {client.get_server_version()}, client {client.get_client_version()}"
                )

            if map_name:
                world = load_map(client, map_name)
            else:
                client.set_timeout(30.0)
                world = client.get_world()

            if not args.no_top_down_view:
                set_top_down_spectator(carla, world, args.spectator_height)

            return run_script(script_command, args.script_timeout)
        finally:
            if carla_process is not None:
                if args.keep_carla and carla_process.poll() is None:
                    print(f"Leaving CARLA running (PID {carla_process.pid}).")
                else:
                    stop_process_group(carla_process, "CARLA")
            if log_handle is not None:
                log_handle.close()

    except LauncherError as exc:
        print(f"Launcher error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
